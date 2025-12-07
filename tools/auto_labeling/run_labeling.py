# Project MUSE - run_labeling.py
# The Teacher's Workshop: Smart Auto-Labeling (Dual Mode: Preview & Full)
# (C) 2025 MUSE Corp. All rights reserved.

import os
import sys
import cv2
import numpy as np
import torch
import json
import glob
import argparse
import gc
from tqdm import tqdm

# Project Root Setup
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

from ai.tracking.vitpose_trt import VitPoseTrt
from ai.distillation.teacher.sam_wrapper import Sam2VideoWrapper

class AutoLabeler:
    def __init__(self, root_session="personal_data"):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.root_data_dir = os.path.join(self.root_dir, "recorded_data", root_session)
        
        if not os.path.exists(self.root_data_dir):
            print("[ERROR] Data folder not found.")
            self.profiles = []
            return
            
        self.profiles = [d for d in os.listdir(self.root_data_dir) if os.path.isdir(os.path.join(self.root_data_dir, d))]

        # [Memory Optimization] Pre-clean
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[Teacher B] ViTPose (Keypoints) Loading...")
        try:
            self.pose_model = VitPoseTrt(engine_path=os.path.join(self.root_dir, "assets/models/tracking/vitpose_huge.engine"))
        except Exception as e:
            print(f"[ERROR] ViTPose Init Failed: {e}")
            sys.exit(1)

        print("[Teacher A] SAM 2 (Video Segmentation) Loading...")
        try:
            self.sam_wrapper = Sam2VideoWrapper(model_root=os.path.join(self.root_dir, "assets/models/segment_anything"))
        except Exception as e:
            print(f"[ERROR] SAM 2 Init Failed: {e}")
            sys.exit(1)

    def _force_clear_memory(self):
        if self.sam_wrapper:
            self.sam_wrapper.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _create_temp_proxy(self, frame, temp_path):
        """1프레임짜리 프리뷰용 영상 생성"""
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, 30, (w, h))
        out.write(frame)
        out.release()

    def _create_strided_video(self, source_path, target_path, interval=5):
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened(): return False, 0, 0, 0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(target_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        count = 0
        written_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % interval == 0:
                out.write(frame)
                written_count += 1
            count += 1
        cap.release()
        out.release()
        return True, w, h, written_count

    def process_previews(self):
        print("\n==================================================")
        print("   [MODE] Preview Analysis (RAM Optimized)")
        print("==================================================")
        
        for profile in self.profiles:
            profile_dir = os.path.join(self.root_data_dir, profile)
            video_paths = sorted(glob.glob(os.path.join(profile_dir, "train_video_*.mp4")))
            preview_dir = os.path.join(profile_dir, "previews")
            os.makedirs(preview_dir, exist_ok=True)
            temp_proxy_path = os.path.join(profile_dir, "temp_proxy_preview.mp4")
            
            print(f"   Target Profile: {profile} ({len(video_paths)} videos)")
            
            for i, video_path in enumerate(video_paths):
                vid_name = os.path.basename(video_path)
                preview_path = os.path.join(preview_dir, vid_name + ".jpg")
                if os.path.exists(preview_path): continue
                
                print(f"   [{i+1}/{len(video_paths)}] Analyzing: {vid_name}...", end=" ", flush=True)
                self._generate_preview_optimized(video_path, preview_path, temp_proxy_path)
                print("Done.")
                self._force_clear_memory()
            
            if os.path.exists(temp_proxy_path): os.remove(temp_proxy_path)

        print("\n[DONE] Preview generation complete.")

    def _generate_preview_optimized(self, video_path, output_path, temp_proxy_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret: return

        try:
            self._create_temp_proxy(frame, temp_proxy_path)
            self.sam_wrapper.init_state(temp_proxy_path)
            keypoints = self.pose_model.inference(frame)
            if keypoints is None: return

            valid_kpts = [kp[:2] for kp in keypoints if kp[2] > 0.4]
            if len(valid_kpts) < 3: return

            points = np.array(valid_kpts, dtype=np.float32)
            labels = np.ones(len(points), dtype=np.int32)
            mask_logits = self.sam_wrapper.add_prompt(frame_idx=0, points=points, labels=labels)
            mask_gpu = (mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
            if mask_gpu.ndim > 2: mask_gpu = np.squeeze(mask_gpu)
            
            colored_mask = np.zeros_like(frame)
            colored_mask[mask_gpu > 0] = [255, 255, 0] # Cyan
            overlay = cv2.addWeighted(frame, 1.0, colored_mask, 0.15, 0)
            contours, _ = cv2.findContours(mask_gpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
            cv2.imwrite(output_path, overlay)
        except Exception as e:
            print(f"[ERROR] {e}")

    def process_full_run(self):
        print("\n==================================================")
        print("   [MODE] Full Labeling (Propagation)")
        print("==================================================")
        
        total_profiles = len(self.profiles)
        for i, profile in enumerate(self.profiles):
            self._process_single_profile_full(profile, i, total_profiles)
            self._force_clear_memory()

    def _process_single_profile_full(self, profile, profile_idx, total_profiles):
        profile_dir = os.path.join(self.root_data_dir, profile)
        video_paths = sorted(glob.glob(os.path.join(profile_dir, "train_video_*.mp4")))
        
        out_imgs = os.path.join(profile_dir, "images")
        out_masks = os.path.join(profile_dir, "masks")
        out_labels = os.path.join(profile_dir, "labels")
        for d in [out_imgs, out_masks, out_labels]: os.makedirs(d, exist_ok=True)

        processed_log_path = os.path.join(profile_dir, "processed_videos.json")
        processed_videos = []
        if os.path.exists(processed_log_path):
            with open(processed_log_path, "r") as f:
                processed_videos = json.load(f)
        
        global_idx = self._get_next_index(out_imgs)
        newly_processed = []
        
        FRAME_INTERVAL = 1
        temp_video_path = os.path.join(profile_dir, "temp_processing_strided.mp4")
        TARGET_POSE_W, TARGET_POSE_H = 640, 352
        
        # [Added] Stop Flag Path
        stop_flag = os.path.join(self.root_data_dir, "stop_training.flag")

        for v_idx, video_path in enumerate(video_paths):
            # [Check Stop] Check before starting new video
            if os.path.exists(stop_flag):
                print(f"\n[STOP] Labeling interrupted by user (Video {v_idx+1}/{len(video_paths)})")
                break

            vid_name = os.path.basename(video_path)
            
            if vid_name in processed_videos:
                print(f"   [SKIP] Already processed: {vid_name}")
                continue

            print(f"   [VIDEO] Processing: {vid_name} (Interval: {FRAME_INTERVAL})")
            
            current_progress = int(((profile_idx * len(video_paths) + v_idx) / (total_profiles * len(video_paths))) * 100)
            print(f"[PROGRESS] {current_progress}")
            
            try:
                ok, w, h, frames = self._create_strided_video(video_path, temp_video_path, FRAME_INTERVAL)
                if not ok or frames == 0:
                    print("      [ERROR] Video read failed.")
                    continue
                
                self.sam_wrapper.init_state(temp_video_path)
                
                cap = cv2.VideoCapture(temp_video_path)
                ret, first_frame = cap.read()
                if not ret:
                    cap.release(); continue
                
                frame_small = cv2.resize(first_frame, (TARGET_POSE_W, TARGET_POSE_H))
                keypoints_small = self.pose_model.inference(frame_small)
                if keypoints_small is None:
                    cap.release(); continue

                scale_x = w / TARGET_POSE_W
                scale_y = h / TARGET_POSE_H
                
                keypoints = keypoints_small.copy()
                keypoints[:, 0] *= scale_x
                keypoints[:, 1] *= scale_y

                valid_kpts = [kp[:2] for kp in keypoints if kp[2] > 0.4]
                if len(valid_kpts) < 3:
                    cap.release(); continue

                points = np.array(valid_kpts, dtype=np.float32)
                labels = np.ones(len(points), dtype=np.int32)
                
                self.sam_wrapper.add_prompt(frame_idx=0, points=points, labels=labels)

                print(f"      [SAM] Propagating ({frames} frames)...")
                video_masks = {}
                for frame_idx, obj_ids, mask_logits in self.sam_wrapper.propagate():
                    m_tensor = mask_logits[0]
                    mask = (m_tensor > 0.0).cpu().numpy().astype(np.uint8)
                    if mask.ndim > 2: mask = np.squeeze(mask)
                    video_masks[frame_idx] = mask

                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                curr_f_idx = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_small = cv2.resize(frame, (TARGET_POSE_W, TARGET_POSE_H))
                    kpts_small = self.pose_model.inference(frame_small)
                    mask = video_masks.get(curr_f_idx, None)
                    
                    if kpts_small is not None and mask is not None:
                        kpts = kpts_small.copy()
                        kpts[:, 0] *= scale_x
                        kpts[:, 1] *= scale_y
                        
                        if mask.ndim > 2: mask = np.squeeze(mask)
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
                newly_processed.append(vid_name)
                
            except Exception as e:
                print(f"      [ERROR] {e}")
            finally:
                self._force_clear_memory()
                if os.path.exists(temp_video_path):
                    try: os.remove(temp_video_path)
                    except: pass
        
        if newly_processed:
            processed_videos.extend(newly_processed)
            with open(processed_log_path, "w") as f:
                json.dump(processed_videos, f, indent=4)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs='?', default="personal_data")
    parser.add_argument("--mode", choices=['preview', 'full'], default='full')
    args = parser.parse_args()

    labeler = AutoLabeler(args.session)
    
    if args.mode == 'preview':
        labeler.process_previews()
    else:
        labeler.process_full_run()