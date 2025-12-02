# Project MUSE - run_labeling.py
# The Teacher's Workshop: Multi-Profile Automatic Data Annotation
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
    print("❌ 'segment_anything' 모듈이 없습니다.")
    sys.exit(1)

class AutoLabeler:
    def __init__(self, root_session="personal_data"):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.root_data_dir = os.path.join(self.root_dir, "recorded_data", root_session)
        
        # [Multi-Profile Search]
        # personal_data/ 아래에 있는 모든 하위 폴더(프로파일)를 찾습니다.
        if not os.path.exists(self.root_data_dir):
            print(f"❌ 데이터 폴더를 찾을 수 없습니다: {self.root_data_dir}")
            sys.exit(1)
            
        self.profiles = [
            d for d in os.listdir(self.root_data_dir) 
            if os.path.isdir(os.path.join(self.root_data_dir, d))
        ]
        
        if not self.profiles:
            print(f"❌ 프로파일 폴더가 없습니다. recorder.py를 먼저 실행하세요.")
            sys.exit(1)
            
        print(f"📂 발견된 프로파일: {', '.join(self.profiles)}")

        # Teacher Load
        print("👨‍🏫 [Teacher B] ViTPose(Keypoints) 로딩 중...")
        try:
            self.pose_model = VitPoseTrt(engine_path=os.path.join(self.root_dir, "assets/models/tracking/vitpose_huge.engine"))
        except Exception as e:
            print(f"❌ ViTPose 로드 실패: {e}")
            sys.exit(1)

        print("👩‍🏫 [Teacher A] SAM(Segmentation) 로딩 중...")
        sam_checkpoint = os.path.join(self.root_dir, "assets/models/segment_anything/sam_vit_h_4b8939.pth")
        if not os.path.exists(sam_checkpoint):
            print(f"❌ SAM 모델 없음: {sam_checkpoint}")
            sys.exit(1)
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
        print("✅ 선생님들 준비 완료.")

    def process_all_profiles(self, frame_interval=5):
        """모든 프로파일을 순회하며 라벨링을 수행합니다."""
        for profile in self.profiles:
            print(f"\n==================================================")
            print(f"   Running Labeling for Profile: [{profile}]")
            print(f"==================================================")
            self._process_single_profile(profile, frame_interval)

    def _process_single_profile(self, profile, frame_interval):
        profile_dir = os.path.join(self.root_data_dir, profile)
        video_paths = sorted(glob.glob(os.path.join(profile_dir, "*.mp4")))
        
        if not video_paths:
            print(f"   ⚠️ 경고: '{profile}' 프로파일에 영상이 없습니다. 스킵합니다.")
            return

        # 출력 디렉토리 (프로파일 폴더 내부)
        out_imgs = os.path.join(profile_dir, "images")
        out_masks = os.path.join(profile_dir, "masks")
        out_labels = os.path.join(profile_dir, "labels")
        
        for d in [out_imgs, out_masks, out_labels]:
            os.makedirs(d, exist_ok=True)

        # [Append Logic] 기존 인덱스 확인
        existing_imgs = glob.glob(os.path.join(out_imgs, "*.jpg"))
        max_idx = -1
        if existing_imgs:
            for p in existing_imgs:
                try:
                    name = os.path.splitext(os.path.basename(p))[0]
                    idx = int(name)
                    if idx > max_idx: max_idx = idx
                except: pass
        
        current_idx = max_idx + 1
        print(f"   🚀 시작 인덱스: {current_idx}")
        processed_count = 0
        
        for vid_idx, video_path in enumerate(video_paths):
            vid_name = os.path.basename(video_path)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"   [{vid_idx+1}/{len(video_paths)}] {vid_name} ({total_frames} frames)")
            
            frame_idx = 0
            pbar = tqdm(total=total_frames, leave=False)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if frame_idx % frame_interval == 0:
                    success = self._annotate_frame(frame, current_idx, out_imgs, out_masks, out_labels)
                    if success:
                        current_idx += 1
                        processed_count += 1
                
                frame_idx += 1
                pbar.update(1)
                
            pbar.close()
            cap.release()
            
        print(f"   🎉 [{profile}] 완료! 추가된 데이터: {processed_count}장")

    def _annotate_frame(self, frame, idx, out_imgs, out_masks, out_labels):
        # 1. Pose
        keypoints = self.pose_model.inference(frame)
        if keypoints is None: return False

        valid_kpts = [kp[:2] for kp in keypoints if kp[2] > 0.4]
        if len(valid_kpts) < 3: return False

        # 2. SAM
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(frame_rgb)
        
        input_points = np.array(valid_kpts)
        input_labels = np.ones(len(input_points))
        
        x_min, y_min = np.min(input_points, axis=0)
        x_max, y_max = np.max(input_points, axis=0)
        h, w = frame.shape[:2]
        pad = 20
        box = np.array([
            max(0, x_min - pad), max(0, y_min - pad),
            min(w, x_max + pad), min(h, y_max + pad)
        ])

        masks, _, _ = self.sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=box[None, :],
            multimask_output=False
        )
        final_mask = masks[0]

        # 3. Save
        filename = f"{idx:06d}"
        cv2.imwrite(os.path.join(out_imgs, f"{filename}.jpg"), frame)
        cv2.imwrite(os.path.join(out_masks, f"{filename}.png"), (final_mask * 255).astype(np.uint8))
        
        label_data = {"keypoints": keypoints.tolist(), "box": box.tolist()}
        with open(os.path.join(out_labels, f"{filename}.json"), "w") as f:
            json.dump(label_data, f)
            
        return True

if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "personal_data"
    labeler = AutoLabeler(session)
    labeler.process_all_profiles(frame_interval=5)