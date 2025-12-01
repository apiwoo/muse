# Project MUSE - run_labeling.py
# The Teacher's Workshop: Automatic Data Annotation (Multi-Video Support)
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import cv2
import numpy as np
import torch
import json
import glob
from tqdm import tqdm

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# [Teachers]
from ai.tracking.vitpose_trt import VitPoseTrt # Teacher B
try:
    from segment_anything import sam_model_registry, SamPredictor # Teacher A
except ImportError:
    print("❌ 'segment_anything' 모듈이 없습니다. 'pip install git+https://github.com/facebookresearch/segment-anything.git' 실행 필요")
    sys.exit(1)

class AutoLabeler:
    def __init__(self, session_name):
        """
        [자동 라벨링 시스템]
        녹화된 영상을 분석하여 학습용 데이터셋(Image + Mask + Keypoints)을 생성합니다.
        V2.0 Update: 폴더 내 모든 MP4 파일을 처리하도록 변경.
        """
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.root_dir, "recorded_data", session_name)
        
        # 폴더 내 모든 MP4 파일 검색
        self.video_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.mp4")))
        
        # 출력 디렉토리 (Dataset Structure)
        self.out_imgs = os.path.join(self.data_dir, "images")
        self.out_masks = os.path.join(self.data_dir, "masks")
        self.out_labels = os.path.join(self.data_dir, "labels") # Keypoints JSON
        
        for d in [self.out_imgs, self.out_masks, self.out_labels]:
            os.makedirs(d, exist_ok=True)

        if not self.video_paths:
            print(f"❌ 폴더 내에 MP4 영상이 없습니다: {self.data_dir}")
            print("   -> 영상을 녹화하거나 복사해 넣으세요.")
            sys.exit(1)

        # 1. Load Teacher B (ViTPose - TensorRT)
        print("👨‍🏫 [Teacher B] ViTPose(Keypoints) 로딩 중...")
        try:
            self.pose_model = VitPoseTrt(engine_path=os.path.join(self.root_dir, "assets/models/tracking/vitpose_huge.engine"))
        except Exception as e:
            print(f"❌ ViTPose 로드 실패: {e}")
            sys.exit(1)

        # 2. Load Teacher A (SAM - PyTorch)
        print("👩‍🏫 [Teacher A] SAM(Segmentation) 로딩 중...")
        sam_checkpoint = os.path.join(self.root_dir, "assets/models/segment_anything/sam_vit_h_4b8939.pth")
        if not os.path.exists(sam_checkpoint):
            print(f"❌ SAM 모델 없음: {sam_checkpoint}")
            print("   -> 'tools/download_models.py'를 확인하거나 모델을 다운로드하세요.")
            sys.exit(1)
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
        print("✅ 선생님들 준비 완료.")

    def process(self, frame_interval=5):
        """
        :param frame_interval: 5프레임마다 1장 추출 (30fps -> 6fps)
        """
        print(f"🚀 라벨링 시작 (총 {len(self.video_paths)}개 영상)")
        
        global_saved_count = 0
        
        for vid_idx, video_path in enumerate(self.video_paths):
            vid_name = os.path.basename(video_path)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"\n[{vid_idx+1}/{len(self.video_paths)}] 처리 중: {vid_name} ({total_frames} frames)")
            
            frame_idx = 0
            pbar = tqdm(total=total_frames)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # 지정된 간격마다 추출
                if frame_idx % frame_interval == 0:
                    # 파일명이 겹치지 않게 전체 카운트 사용
                    success = self._annotate_frame(frame, global_saved_count)
                    if success:
                        global_saved_count += 1
                
                frame_idx += 1
                pbar.update(1)
                
            pbar.close()
            cap.release()
            
        print(f"\n🎉 전체 완료! 총 {global_saved_count}장의 정답 데이터 생성.")
        print(f"👉 경로: {self.data_dir}")

    def _annotate_frame(self, frame, idx):
        # 1. Pose Estimation (Teacher B)
        # ViTPose는 BGR 이미지를 받습니다.
        keypoints = self.pose_model.inference(frame) # [17, 3] (x, y, conf)
        
        if keypoints is None:
            return False

        # 유효한 키포인트(신뢰도 > 0.4) 필터링
        valid_kpts = []
        for kp in keypoints:
            x, y, conf = kp
            if conf > 0.4:
                valid_kpts.append([x, y])
        
        if len(valid_kpts) < 3: # 사람이 거의 안 보이면 스킵
            return False

        # 2. Segmentation (Teacher A)
        # SAM은 RGB 이미지를 받습니다.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(frame_rgb)
        
        # ViTPose의 좌표를 힌트(Point Prompt)로 제공
        input_points = np.array(valid_kpts)
        input_labels = np.ones(len(input_points)) # 1 = Foreground
        
        # Box Prompt 추가 (몸 전체를 감싸는 박스) - 안정성 향상
        x_min = np.min(input_points[:, 0])
        y_min = np.min(input_points[:, 1])
        x_max = np.max(input_points[:, 0])
        y_max = np.max(input_points[:, 1])
        
        # 박스에 여유(Padding) 주기
        h, w = frame.shape[:2]
        pad = 20
        box = np.array([
            max(0, x_min - pad), max(0, y_min - pad),
            min(w, x_max + pad), min(h, y_max + pad)
        ])

        masks, _, _ = self.sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=box[None, :], # Box 힌트 추가
            multimask_output=False # 모호함 없이 하나만 출력
        )
        
        final_mask = masks[0] # (H, W) bool array

        # 3. Save Data
        filename = f"{idx:06d}"
        
        # (1) 원본 이미지
        cv2.imwrite(os.path.join(self.out_imgs, f"{filename}.jpg"), frame)
        
        # (2) 마스크 (0 or 255)
        mask_uint8 = (final_mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.out_masks, f"{filename}.png"), mask_uint8)
        
        # (3) 라벨 (Keypoints JSON)
        label_data = {
            "keypoints": keypoints.tolist(), # JSON 직렬화 가능하게 변환
            "box": box.tolist()
        }
        with open(os.path.join(self.out_labels, f"{filename}.json"), "w") as f:
            json.dump(label_data, f)
            
        return True

if __name__ == "__main__":
    # 사용자가 입력한 세션 이름 (폴더명)
    if len(sys.argv) < 2:
        print("사용법: python run_labeling.py <SESSION_NAME>")
        print("예: python run_labeling.py 20231025_143000")
        sys.exit(1)
        
    session_name = sys.argv[1]
    labeler = AutoLabeler(session_name)
    
    # 5프레임 간격 (30fps 영상 -> 6fps 데이터셋)
    labeler.process(frame_interval=5)