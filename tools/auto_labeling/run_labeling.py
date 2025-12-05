# Project MUSE - run_labeling.py
# The Teacher's Workshop: Smart Auto-Labeling (Append Support)
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import cv2
import numpy as np
import torch
import json
import glob
from tqdm import tqdm

# 프로젝트 루트 경로 확보
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src")) # [Fix] src 폴더를 경로에 추가하여 ai 모듈 인식

from ai.tracking.vitpose_trt import VitPoseTrt
from ai.distillation.teacher.sam_wrapper import Sam2VideoWrapper

class AutoLabeler:
    def __init__(self, root_session="personal_data"):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.root_data_dir = os.path.join(self.root_dir, "recorded_data", root_session)
        
        if not os.path.exists(self.root_data_dir):
            print("[Error] 데이터 폴더가 없습니다.")
            self.profiles = []
            return
            
        self.profiles = [d for d in os.listdir(self.root_data_dir) if os.path.isdir(os.path.join(self.root_data_dir, d))]

        print("[Teacher B] ViTPose (Keypoints) Loading...")
        self.pose_model = VitPoseTrt(engine_path=os.path.join(self.root_dir, "assets/models/tracking/vitpose_huge.engine"))

        print("[Teacher A] SAM 2 (Video Segmentation) Loading...")
        self.sam_wrapper = Sam2VideoWrapper(model_root=os.path.join(self.root_dir, "assets/models/segment_anything"))

    def process_all_profiles(self):
        total_profiles = len(self.profiles)
        for i, profile in enumerate(self.profiles):
            print(f"\n==================================================")
            print(f"   Labeling Profile ({i+1}/{total_profiles}): [{profile}]")
            print(f"==================================================")
            self._process_single_profile(profile, i, total_profiles)

    def _process_single_profile(self, profile, profile_idx, total_profiles):
        profile_dir = os.path.join(self.root_data_dir, profile)
        video_paths = sorted(glob.glob(os.path.join(profile_dir, "train_video_*.mp4")))
        
        out_imgs = os.path.join(profile_dir, "images")
        out_masks = os.path.join(profile_dir, "masks")
        out_labels = os.path.join(profile_dir, "labels")
        for d in [out_imgs, out_masks, out_labels]: os.makedirs(d, exist_ok=True)

        # [Append Logic] 이미 처리된 비디오 목록 로드
        processed_log_path = os.path.join(profile_dir, "processed_videos.json")
        processed_videos = []
        if os.path.exists(processed_log_path):
            with open(processed_log_path, "r") as f:
                processed_videos = json.load(f)
        
        # 다음 이미지 인덱스 계산 (이어쓰기)
        global_idx = self._get_next_index(out_imgs)
        
        newly_processed = []

        for v_idx, video_path in enumerate(video_paths):
            vid_name = os.path.basename(video_path)
            
            # [Smart Check] 이미 처리된 영상이면 스킵
            if vid_name in processed_videos:
                # 단, 이미지가 실제로 있는지 확인은 필요할 수 있음 (여기서는 로그 신뢰)
                print(f"   [Skip] Skipping processed video: {vid_name}")
                continue

            print(f"   [Video] Processing New Video: {vid_name}")
            
            # [GUI Log] 비디오 단위 진행률
            current_progress = int(((profile_idx * len(video_paths) + v_idx) / (total_profiles * len(video_paths))) * 100)
            print(f"[PROGRESS] {current_progress}")
            
            try:
                self.sam_wrapper.init_state(video_path)
            except Exception as e:
                print(f"      [Error] SAM Init Failed: {e}")
                continue

            cap = cv2.VideoCapture(video_path)
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                continue
            
            keypoints = self.pose_model.inference(first_frame)
            if keypoints is None:
                print("      [Warn] No pose detected in first frame.")
                cap.release()
                self.sam_wrapper.reset()
                continue

            valid_kpts = [kp[:2] for kp in keypoints if kp[2] > 0.4]
            if len(valid_kpts) < 3:
                print("      [Warn] Not enough keypoints.")
                cap.release()
                self.sam_wrapper.reset()
                continue

            points = np.array(valid_kpts, dtype=np.float32)
            labels = np.ones(len(points), dtype=np.int32)
            
            self.sam_wrapper.add_prompt(frame_idx=0, points=points, labels=labels)

            print("      [SAM] Propagating masks...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            video_masks = {}
            for frame_idx, obj_ids, mask_logits in self.sam_wrapper.propagate():
                mask = (mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
                video_masks[frame_idx] = mask

            # Save Data
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            curr_f_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # 프레임 저장 여부 결정 (6프레임당 1장 등 샘플링 가능하나, 여기선 전부 저장하고 Trainer에서 셔플)
                # 용량 절약을 위해 3프레임당 1장 저장 (옵션)
                # 여기서는 풀 데이터를 위해 전부 저장
                
                kpts = self.pose_model.inference(frame)
                mask = video_masks.get(curr_f_idx, None)
                
                if kpts is not None and mask is not None:
                    fname = f"{global_idx:06d}"
                    cv2.imwrite(os.path.join(out_imgs, f"{fname}.jpg"), frame)
                    cv2.imwrite(os.path.join(out_masks, f"{fname}.png"), mask * 255)
                    
                    y_indices, x_indices = np.where(mask > 0)
                    if len(x_indices) > 0:
                        box = [int(np.min(x_indices)), int(np.min(y_indices)), 
                               int(np.max(x_indices)), int(np.max(y_indices))]
                    else:
                        box = [0, 0, 0, 0]

                    label_data = {"keypoints": kpts.tolist(), "box": box}
                    with open(os.path.join(out_labels, f"{fname}.json"), "w") as f:
                        json.dump(label_data, f)
                    
                    global_idx += 1
                
                curr_f_idx += 1

            cap.release()
            self.sam_wrapper.reset()
            
            # 처리 완료 목록에 추가
            newly_processed.append(vid_name)
        
        # 로그 업데이트
        if newly_processed:
            processed_videos.extend(newly_processed)
            with open(processed_log_path, "w") as f:
                json.dump(processed_videos, f, indent=4)
            print(f"   [OK] Added {len(newly_processed)} videos to processed log.")
        else:
            print("   [Info] No new videos to process.")

    def _get_next_index(self, dir_path):
        files = glob.glob(os.path.join(dir_path, "*.jpg"))
        max_idx = -1
        for f in files:
            try:
                idx = int(os.path.splitext(os.path.basename(f))[0])
                if idx > max_idx: max_idx = idx
            except: pass
        return max_idx + 1

if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "personal_data"
    labeler = AutoLabeler(session)
    labeler.process_all_profiles()