# Project MUSE - test_lightweight_sam.py
# (C) 2025 MUSE Corp. All rights reserved.
# Purpose: Real-time Teacher Combination Test (ViTPose + SAM 2.1 Teacher)
# "ViTPose가 눈(Eye)이 되어주고, SAM 2.1(Teacher)가 손(Hand)이 되어 잘라냅니다."

import cv2
import numpy as np
import sys
import os
import time
import torch
import traceback

# 프로젝트 루트 경로 설정
current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

# CUDA 설정
try:
    from utils.cuda_helper import setup_cuda_environment
    setup_cuda_environment()
except ImportError:
    pass

# 모듈 로드
try:
    from ai.tracking.vitpose_trt import VitPoseTrt
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
except ImportError as e:
    print(f"[ERROR] 필수 모듈 로드 실패: {e}")
    sys.exit(1)

def apply_mask_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    """마스크 영역에 색상을 입혀서 오버레이"""
    if mask is None: return image
    
    mask = mask.astype(bool)
    overlay = image.copy()
    overlay[mask] = np.array(color, dtype=np.uint8)
    
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

def main():
    print("========================================================")
    print("   MUSE Real-time Teacher Test (ViTPose + SAM 2.1 Large)")
    print("========================================================")

    # 1. 모델 경로 설정
    # (1) ViTPose (Pose)
    engine_path = os.path.join(root_dir, "assets", "models", "tracking", "vitpose_huge.engine")
    if not os.path.exists(engine_path):
        print("❌ ViTPose Engine 파일이 없습니다.")
        print("   👉 'python tools/trt_converter.py'를 먼저 실행해주세요.")
        return

    # (2) SAM 2.1 Teacher (Large) - [Modified]
    # 가장 정확도가 높은 최신 모델 (SAM 2.1 Hiera Large)
    sam2_checkpoint = os.path.join(root_dir, "assets", "models", "segment_anything", "sam2.1_hiera_large.pt")
    
    # Config 설정 (SAM 2.1) - [Fixed]
    # 모델 버전과 Config 버전을 2.1로 일치시킴
    sam2_config_dir = os.path.join(root_dir, "assets", "sam2_configs", "sam2.1")
    sam2_config_name = "sam2.1_hiera_l.yaml" 

    if not os.path.exists(sam2_checkpoint):
        print("❌ SAM 2.1 Large 체크포인트가 없습니다.")
        print("   👉 'python tools/download_models.py'를 실행해서 다운로드해주세요.")
        return

    # 2. 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")

    # [Load ViTPose]
    print("⏳ [1/2] Loading ViTPose (TensorRT)...")
    try:
        pose_model = VitPoseTrt(engine_path)
    except Exception as e:
        print(f"❌ ViTPose 로드 실패: {e}")
        return

    # [Load SAM 2.1 Large]
    print(f"⏳ [2/2] Loading SAM 2.1 Large (Most Accurate)...")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    try:
        with initialize_config_dir(config_dir=sam2_config_dir, version_base="1.2"):
            # 확장자를 뺀 이름으로 빌드 시도 (Hydra 특성)
            cfg_name = sam2_config_name.replace(".yaml", "")
            sam2_model = build_sam2(cfg_name, sam2_checkpoint, device=device)
            predictor = SAM2ImagePredictor(sam2_model)
    except Exception as e:
        print(f"❌ SAM 2 로드 실패: {e}")
        traceback.print_exc()
        return

    print("✅ 모든 모델 준비 완료.")

    # 3. 카메라 실행
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n🎥 [Start] Loop 시작 (Press 'q' to quit)")
    
    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        loop_start = time.time()

        # -----------------------------------------------------------
        # [Step A] ViTPose: 사람 위치 찾기
        # -----------------------------------------------------------
        kpts = pose_model.inference(frame)
        
        box_prompt = None
        has_person = False

        if kpts is not None:
            # 신뢰도 0.4 이상인 키포인트만 필터링
            valid_kpts = kpts[kpts[:, 2] > 0.4]
            
            if len(valid_kpts) > 3: # 점이 3개 이상 보여야 사람으로 인정
                has_person = True
                
                # Bounding Box 계산 (여유 공간 Padding 추가)
                x_min = np.min(valid_kpts[:, 0])
                x_max = np.max(valid_kpts[:, 0])
                y_min = np.min(valid_kpts[:, 1])
                y_max = np.max(valid_kpts[:, 1])
                
                # 박스를 조금 더 크게 잡아서(Padding) SAM이 사람 전체를 잘 잡도록 유도
                pad = 20
                h, w = frame.shape[:2]
                box_prompt = np.array([
                    max(0, x_min - pad), 
                    max(0, y_min - pad), 
                    min(w, x_max + pad), 
                    min(h, y_max + pad)
                ])

        # -----------------------------------------------------------
        # [Step B] SAM 2: 찾은 위치를 기반으로 정밀 분리
        # -----------------------------------------------------------
        mask_final = None
        
        if has_person and box_prompt is not None:
            # 1. 이미지 인코딩 (Large 모델은 여기서 시간이 좀 걸릴 수 있습니다)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictor.set_image(frame_rgb)
            
            # 2. 박스 프롬프트로 마스크 예측
            masks, scores, _ = predictor.predict(
                box=box_prompt,
                multimask_output=False # 가장 확실한 마스크 1개만 요청
            )
            mask_final = masks[0]

        # -----------------------------------------------------------
        # [Step C] 시각화
        # -----------------------------------------------------------
        display = frame.copy()
        
        # 박스 그리기 (노란색)
        if box_prompt is not None:
            x1, y1, x2, y2 = box_prompt.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(display, "ViTPose Detection", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 마스크 오버레이 (초록색)
        if mask_final is not None:
            display = apply_mask_overlay(display, mask_final, color=(0, 255, 0), alpha=0.4)
            
            # 외곽선 그리기
            contours, _ = cv2.findContours(mask_final.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

        # FPS 계산
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        infer_time = (time.time() - loop_start) * 1000
        
        # [Modified] Info Text Update
        info_text = f"FPS: {fps:.1f} | Latency: {infer_time:.1f}ms | Model: SAM 2.1 Large"
        cv2.rectangle(display, (0, 0), (650, 40), (0, 0, 0), -1)
        cv2.putText(display, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("MUSE Teacher Test", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()