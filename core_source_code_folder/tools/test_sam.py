# Project MUSE - test_sam.py
# (C) 2025 MUSE Corp. All rights reserved.
# Purpose: SAM (Segment Anything Model) 성능 및 속도 체험 (Teacher Model Test)

import os
import sys
import cv2
import numpy as np
import torch
import time

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("❌ 'segment_anything' 라이브러리가 없습니다.")
    print("👉 설치: pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    # 마스크 오버레이 생성
    img_shape = (sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3)
    mask_overlay = np.zeros(img_shape, dtype=np.uint8)
    
    for ann in sorted_anns:
        m = ann['segmentation']
        # 랜덤 색상 생성
        color_mask = np.random.randint(0, 255, (1, 3)).tolist()[0]
        
        mask_overlay[m] = color_mask

    return mask_overlay

def main():
    print("========================================================")
    print("   MUSE Teacher Model Test: SAM (ViT-Huge)")
    print("========================================================")

    # 1. 모델 경로 설정
    # assets/models/segment_anything 폴더에 모델이 있어야 합니다.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "assets", "models", "segment_anything", "sam_vit_h_4b8939.pth")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("👉 다운로드: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return

    # 2. CUDA 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device} (CUDA가 아니면 매우 느립니다)")

    # 3. 모델 로딩 (시간이 좀 걸립니다)
    print("⏳ SAM 모델(ViT-Huge)을 메모리에 적재 중... (약 1~2분 소요)")
    sam = sam_model_registry["vit_h"](checkpoint=MODEL_PATH)
    sam.to(device=device)
    
    # 마스크 생성기 초기화
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # 너무 작은 영역 무시
    )
    print("✅ SAM 모델 로드 완료! 선생님 준비 끝.")

    # 4. 카메라 실행
    cap = cv2.VideoCapture(1) # 장치 ID (0 또는 1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n🎥 [조작 방법]")
    print("   - SPACE: 현재 화면 캡처 후 SAM 분석 (몇 초 걸림)")
    print("   - Q: 종료")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        cv2.putText(display_frame, "Press SPACE to Segment", (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("MUSE - SAM Tester", display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '): # 스페이스바
            print("\n📸 캡처됨! 선생님(SAM)이 분석을 시작합니다...")
            start_t = time.time()
            
            # BGR -> RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # [핵심] SAM 추론 (오래 걸림)
            masks = mask_generator.generate(frame_rgb)
            
            end_t = time.time()
            elapsed = end_t - start_t
            
            print(f"✨ 분석 완료! 소요 시간: {elapsed:.2f}초")
            print(f"   -> 찾은 객체 수: {len(masks)}개")

            # 결과 시각화
            mask_overlay = show_anns(masks)
            
            # 원본과 마스크 합성 (5:5 비율)
            result = cv2.addWeighted(frame, 0.6, mask_overlay, 0.4, 0)
            
            cv2.imshow("MUSE - SAM Result", result)
            print("   (결과 창을 확인하세요. 아무 키나 누르면 카메라로 돌아갑니다.)")
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()