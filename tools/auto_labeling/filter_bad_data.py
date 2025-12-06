# Project MUSE - filter_bad_data.py
# (C) 2025 MUSE Corp.
# Purpose: Removes frames where body keypoints fall outside the segmentation mask.
# Target: Prevent training pollution.

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

    # Keypoint Indices to check (Body only: 5~16)
    # 0~4: Face (Nose, Eyes, Ears) - Ignored (can be outside due to hair/angles)
    # 5~16: Shoulders, Elbows, Wrists, Hips, Knees, Ankles
    CHECK_INDICES = list(range(5, 17)) 

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
                    continue # Out of bounds implies outside mask, but usually handled by crop logic.
                             # Here we focus on points INSIDE frame but OUTSIDE mask.
                
                # Mask Value Check (0 = Background)
                # We use a small tolerance? No, strict check as requested.
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
    print(f"[FILTER] Result: Deleted {deleted_count}/{total_files} ({ratio:.1f}%) bad frames.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_bad_data.py <profile_name>")
    else:
        filter_data(sys.argv[1])