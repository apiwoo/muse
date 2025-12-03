# Project MUSE - run_labeling.py
# The Teacher's Workshop: Auto-Labeling with SAM 2 & ViTPose
# Update: Video Propagation Logic
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.tracking.vitpose_trt import VitPoseTrt
from ai.distillation.teacher.sam_wrapper import Sam2VideoWrapper

class AutoLabeler:
    def __init__(self, root_session="personal_data"):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.root_data_dir = os.path.join(self.root_dir, "recorded_data", root_session)
        self.profiles = [d for d in os.listdir(self.root_data_dir) if os.path.isdir(os.path.join(self.root_data_dir, d))]

        # 1. Load Teacher B (ViTPose) - Frame by Frame
        print("👨‍🏫 [Teacher B] ViTPose (Keypoints) Loading...")
        self.pose_model = VitPoseTrt(engine_path=os.path.join(self.root_dir, "assets/models/tracking/vitpose_huge.engine"))

        # 2. Load Teacher A (SAM 2) - Video Batch
        print("👩‍🏫 [Teacher A] SAM 2 (Video Segmentation) Loading...")
        self.sam_wrapper = Sam2VideoWrapper(model_root=os.path.join(self.root_dir, "assets/models/segment_anything"))

    def process_all_profiles(self):
        for profile in self.profiles:
            print(f"\n==================================================")
            print(f"   Labeling Profile: [{profile}]")
            print(f"==================================================")
            self._process_single_profile(profile)

    def _process_single_profile(self, profile):
        profile_dir = os.path.join(self.root_data_dir, profile)
        video_paths = sorted(glob.glob(os.path.join(profile_dir, "train_video_*.mp4")))
        
        out_imgs = os.path.join(profile_dir, "images")
        out_masks = os.path.join(profile_dir, "masks")
        out_labels = os.path.join(profile_dir, "labels")
        for d in [out_imgs, out_masks, out_labels]: os.makedirs(d, exist_ok=True)

        global_idx = self._get_next_index(out_imgs)
        
        for video_path in video_paths:
            print(f"   🎥 Processing Video: {os.path.basename(video_path)}")
            
            # 1. Initialize SAM 2 Session
            try:
                self.sam_wrapper.init_state(video_path)
            except Exception as e:
                print(f"      ❌ SAM Init Failed: {e}")
                continue

            # 2. Keyframe Analysis (First Frame)
            cap = cv2.VideoCapture(video_path)
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                continue
            
            # Pose Detection on First Frame
            keypoints = self.pose_model.inference(first_frame)
            if keypoints is None:
                print("      ⚠️ No pose detected in first frame. Skipping video.")
                cap.release()
                self.sam_wrapper.reset()
                continue

            valid_kpts = [kp[:2] for kp in keypoints if kp[2] > 0.4]
            if len(valid_kpts) < 3:
                print("      ⚠️ Not enough keypoints. Skipping.")
                cap.release()
                self.sam_wrapper.reset()
                continue

            # 3. Prompting SAM 2
            # Use points from ViTPose as positive prompts
            points = np.array(valid_kpts, dtype=np.float32)
            labels = np.ones(len(points), dtype=np.int32)
            
            self.sam_wrapper.add_prompt(frame_idx=0, points=points, labels=labels)

            # 4. Propagation & Saving
            print("      🌊 Propagating masks...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Map frame_idx -> Mask
            video_masks = {}
            for frame_idx, obj_ids, mask_logits in self.sam_wrapper.propagate():
                # mask_logits: (1, H, W) -> binary mask
                mask = (mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
                video_masks[frame_idx] = mask

            # 5. Save Data Loop
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar = tqdm(total=total_frames, leave=False, desc="Saving")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            curr_f_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Pose for current frame (Teacher B needs to run every frame for labels)
                kpts = self.pose_model.inference(frame)
                
                # Get Mask from SAM Propagation
                mask = video_masks.get(curr_f_idx, None)
                
                if kpts is not None and mask is not None:
                    # Save
                    fname = f"{global_idx:06d}"
                    cv2.imwrite(os.path.join(out_imgs, f"{fname}.jpg"), frame)
                    cv2.imwrite(os.path.join(out_masks, f"{fname}.png"), mask * 255)
                    
                    # Compute Box from Mask for label
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
                pbar.update(1)

            pbar.close()
            cap.release()
            self.sam_wrapper.reset()

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