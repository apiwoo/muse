# Project MUSE - filter_bad_data.py
# (C) 2025 MUSE Corp.
# Purpose: Removes frames where body keypoints fall outside the segmentation mask.
# Target: Prevent training pollution.
# Updated: Filter based on TORSO & ARMS only (Ignore Head & Legs clipping).

import os
import sys
import json
import cv2
import numpy as np
import glob
from tqdm import tqdm

def filter_data(profile_name, root_session="personal_data"):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root_dir, "recorded_data", root_session, profile_name)
    
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    label_dir = os.path.join(data_dir, "labels")
    
    if not os.path.exists(label_dir):
        print(f"[ERROR] Labels not found for profile: {profile_name}")
        return

    label_files = glob.glob(os.path.join(label_dir, "*.json"))
    total_files = len(label_files)
    deleted_count = 0
    
    print(f"[FILTER] Scanning {total_files} frames in '{profile_name}'...")

    # [Update] Keypoint Indices Check
    # 머리(0~4)와 다리(13~16)는 검사하지 않습니다.
    # 오직 상체와 팔만 마스크 안에 들어와 있으면 유효한 데이터로 간주합니다.
    # -----------------------------------------------------------
    # 0~4: Face (Ignored)
    # 5,6: Shoulders (Check)
    # 7,8: Elbows (Check)
    # 9,10: Wrists (Check)
    # 11,12: Hips (Check)
    # 13~16: Legs (Ignored)
    # -----------------------------------------------------------
    CHECK_INDICES = list(range(5, 13)) 

    for label_path in tqdm(label_files):
        basename = os.path.splitext(os.path.basename(label_path))[0]
        mask_path = os.path.join(mask_dir, f"{basename}.png")
        img_path = os.path.join(img_dir, f"{basename}.jpg")
        
        if not os.path.exists(mask_path):
            continue

        # Load Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        
        h, w = mask.shape
        
        # Load Label
        with open(label_path, "r") as f:
            data = json.load(f)
            keypoints = data.get("keypoints", []) # List of [x, y, conf]

        is_bad = False
        
        for idx in CHECK_INDICES:
            if idx >= len(keypoints): break
            
            kp = keypoints[idx]
            x, y, conf = int(kp[0]), int(kp[1]), kp[2]
            
            # Check only confident keypoints
            if conf > 0.4:
                # Boundary check
                if x < 0 or x >= w or y < 0 or y >= h:
                    continue 
                
                # Mask Value Check (0 = Background)
                if mask[y, x] == 0:
                    is_bad = True
                    break
        
        if is_bad:
            try:
                os.remove(label_path)
                if os.path.exists(img_path): os.remove(img_path)
                if os.path.exists(mask_path): os.remove(mask_path)
                deleted_count += 1
            except Exception as e:
                print(f"[ERR] Failed to delete {basename}: {e}")

    ratio = (deleted_count / total_files) * 100 if total_files > 0 else 0
    print(f"[FILTER] Result: Deleted {deleted_count}/{total_files} ({ratio:.1f}%) bad frames (Torso/Arms Only).")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_bad_data.py <profile_name>")
    else:
        filter_data(sys.argv[1])