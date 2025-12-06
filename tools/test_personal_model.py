# Project MUSE - test_personal_model.py
# (C) 2025 MUSE Corp. All rights reserved.
# Purpose: 개인화 모델(Student Model) 성능 검증 도구
# 기능: Segmentation Mask 오버레이 확인 + Skeleton 시각화

import cv2
import numpy as np
import sys
import os
import argparse
import time
import logging  # [Added] 로깅 모듈 추가

# [Added] 상세 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

# [Fix] 경로 설정 (프로젝트 루트 및 src 폴더 추가)
current_file = os.path.abspath(__file__)
tools_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(tools_dir)
src_dir = os.path.join(root_dir, "src")

# 'muse' 루트와 'src' 폴더를 모두 경로에 추가해야 내부 모듈(ai, utils 등)이 서로를 찾을 수 있음
sys.path.append(root_dir)
sys.path.append(src_dir)

# [Fix] CUDA DLL 경로 설정 (Windows 호환성)
try:
    from utils.cuda_helper import setup_cuda_environment
    setup_cuda_environment()
except ImportError:
    logging.warning("utils.cuda_helper not found. Skipping CUDA setup.") # 로그 추가
    pass

# Import MUSE Modules
# [Fix] 'src.ai...' 대신 'ai...'로 import (내부 모듈과 일관성 유지)
try:
    from ai.tracking.body_tracker import BodyTracker
    logging.info("Module 'ai.tracking.body_tracker' imported successfully.")
except ImportError as e:
    logging.error(f"Failed to import BodyTracker: {e}")
    sys.exit(1)

# [New] 디버그용 스켈레톤 그리기 함수 (낮은 신뢰도용)
def draw_skeleton_debug(img, keypoints, conf_thresh=0.1):
    """
    BodyTracker의 기본 draw_debug 대신 사용할 커스텀 함수.
    낮은 신뢰도(0.1)에서도 억지로 선을 그려서 디버깅을 돕습니다.
    """
    if keypoints is None: return img
    
    # COCO Keypoint Indices
    # 0:Nose, 1:L-Eye, 2:R-Eye, 3:L-Ear, 4:R-Ear
    # 5:L-Shldr, 6:R-Shldr, 7:L-Elbow, 8:R-Elbow, 9:L-Wrist, 10:R-Wrist
    # 11:L-Hip, 12:R-Hip, 13:L-Knee, 14:R-Knee, 15:L-Ankle, 16:R-Ankle
    
    # 연결 관계 (Skeleton Edges)
    edges = [
        (0,1), (0,2), (1,3), (2,4),         # Face
        (5,6), (5,11), (6,12), (11,12),     # Torso
        (5,7), (7,9),                       # Left Arm
        (6,8), (8,10),                      # Right Arm
        (11,13), (13,15),                   # Left Leg
        (12,14), (14,16)                    # Right Leg
    ]
    
    # 색상 (BGR)
    color_point = (0, 0, 255)   # Red Points
    color_line = (0, 255, 0)    # Green Lines

    # 1. 점 그리기
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > conf_thresh:
            cv2.circle(img, (int(x), int(y)), 4, color_point, -1)

    # 2. 선 그리기
    for i, j in edges:
        if i < len(keypoints) and j < len(keypoints):
            xi, yi, ci = keypoints[i]
            xj, yj, cj = keypoints[j]
            if ci > conf_thresh and cj > conf_thresh:
                cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)), color_line, 2)
                
    return img

def main():
    parser = argparse.ArgumentParser(description="MUSE Personal Model Tester")
    parser.add_argument("--profile", type=str, default="front", help="Target profile name (e.g., front, top)")
    parser.add_argument("--cam", type=int, default=0, help="Camera Index")
    args = parser.parse_args()

    print("========================================================")
    print(f"   MUSE Personal Model Tester - Profile: [{args.profile}]")
    print("========================================================")
    logging.info(f"Test Started. Target Profile: {args.profile}, Camera ID: {args.cam}")

    # 1. BodyTracker 초기화 (모든 프로필 스캔)
    try:
        logging.info("Initializing BodyTracker (Scanning for dual models)...")
        tracker = BodyTracker()
        logging.info(f"BodyTracker Initialized. Loaded Profiles: {list(tracker.models.keys())}")
    except Exception as e:
        print(f"❌ Tracker 초기화 실패: {e}")
        logging.error(f"Tracker Init Error: {e}", exc_info=True)
        return

    # 2. 프로필 선택
    logging.info(f"Selecting profile: {args.profile}")
    if not tracker.set_profile(args.profile):
        print(f"❌ 프로필 '{args.profile}'을 찾을 수 없습니다.")
        print("   [Check List]")
        print(f"   1. assets/models/personal/ 폴더에 다음 두 파일이 모두 있어야 합니다:")
        print(f"      - student_seg_{args.profile}.engine")
        print(f"      - student_pose_{args.profile}.engine")
        print("   2. 학습 후 변환(Convert) 과정을 수행했는지 확인하세요.")
        print("      -> 실행: python tools/convert_student_to_trt.py --profile {args.profile}")
        
        logging.warning(f"Profile '{args.profile}' not found in tracker. Using fallback.")
        print("   -> (테스트를 위해 기본(default) 프로필 또는 로드된 첫 번째 모델로 진행합니다.)")
        
        if len(tracker.models) > 0:
            fallback_profile = list(tracker.models.keys())[0]
            tracker.set_profile(fallback_profile)
            print(f"   -> Fallback Profile: {fallback_profile}")
        else:
            print("   ❌ 로드된 모델이 전혀 없습니다. 종료합니다.")
            return
    else:
        logging.info(f"Profile '{args.profile}' selected successfully.")

    # 3. 카메라 연결
    logging.info(f"Opening Camera {args.cam}...")
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print(f"❌ 카메라 {args.cam}번을 열 수 없습니다.")
        logging.error(f"Failed to open Camera {args.cam}")
        return
    
    # 카메라 실제 설정 확인 로그
    real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Camera Opened: {int(real_w)}x{int(real_h)} @ {real_fps:.1f} FPS")

    print("\n🎥 [Start] 테스트 시작 (Press 'q' to quit)")
    print("   - Green Lines: Skeleton (Joints)")
    print("   - Red Overlay: Person Mask (Background Removal)")

    prev_time = time.time()
    frame_count = 0 

    while True:
        ret, frame = cap.read()
        if not ret: 
            logging.warning("Frame read failed or stream ended.")
            break

        frame_count += 1
        
        # ---------------------------------------------------------
        # [Step 1] Inference (추론)
        # ---------------------------------------------------------
        
        t_infer_start = time.perf_counter() 
        try:
            keypoints = tracker.process(frame)
        except Exception as e:
            print(f"[Error] Inference Failed: {e}")
            logging.error(f"Inference Exception: {e}", exc_info=True)
            break
        t_infer_end = time.perf_counter()
        infer_ms = (t_infer_end - t_infer_start) * 1000.0
        
        # GPU 메모리에 있는 Mask 가져오기 (CuPy -> Numpy)
        mask_gpu = tracker.get_mask()
        mask_cpu = None
        
        if mask_gpu is not None:
            if hasattr(mask_gpu, 'get'):
                mask_cpu = mask_gpu.get() # GPU -> CPU
            elif hasattr(mask_gpu, 'cpu'):
                mask_cpu = mask_gpu.cpu().numpy() # Torch -> Numpy
            else:
                mask_cpu = mask_gpu 

        # [Added] 상세 로그 출력 (60프레임마다)
        if frame_count % 60 == 0:
            valid_kpts = 0
            avg_conf = 0.0
            if keypoints is not None:
                valid_list = [k for k in keypoints if k[2] > 0.0]
                valid_kpts = len(valid_list)
                if valid_kpts > 0:
                    avg_conf = sum(k[2] for k in valid_list) / valid_kpts
            
            mask_fill_ratio = 0.0
            if mask_cpu is not None:
                mask_fill_ratio = np.count_nonzero(mask_cpu) / mask_cpu.size * 100
                
            logging.info(f"[F{frame_count}] Infer: {infer_ms:.2f}ms | Valid Kpts: {valid_kpts} (Conf: {avg_conf:.2f}) | Mask: {mask_fill_ratio:.1f}%")

        # ---------------------------------------------------------
        # [Step 2] Visualization (시각화)
        # ---------------------------------------------------------
        display = frame.copy()

        # 1. Mask Overlay (Segmentation 확인)
        if mask_cpu is not None:
            if mask_cpu.dtype != np.uint8:
                mask_u8 = (mask_cpu * 255).astype(np.uint8)
            else:
                mask_u8 = mask_cpu

            zeros = np.zeros_like(mask_u8)
            mask_color = cv2.merge([zeros, zeros, mask_u8]) 
            display = cv2.addWeighted(display, 1.0, mask_color, 0.5, 0)
            
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 255), 2) 

        # 2. Skeleton Draw (관절 확인) - [Modified] 커스텀 함수 사용 (Threshold 0.1)
        if keypoints is not None:
            # 기존: display = tracker.draw_debug(display, keypoints)
            display = draw_skeleton_debug(display, keypoints, conf_thresh=0.1)

        # ---------------------------------------------------------
        # [Step 3] Info Display
        # ---------------------------------------------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        cv2.putText(display, f"Profile: {args.profile}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"FPS: {fps:.1f} (Infer: {infer_ms:.1f}ms)", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("MUSE Personal Model Test", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("User requested quit.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Test terminated cleanly.")

if __name__ == "__main__":
    main()