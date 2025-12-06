# Project MUSE - body_tracker.py
# Updated for Dual Engine Support
# (C) 2025 MUSE Corp. All rights reserved.

import os
import glob
import time
import numpy as np
import cv2

try:
    from ai.distillation.student.inference_trt import DualInferenceTRT
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

from ai.distillation.student.inference import StudentInference

class BodyTracker:
    def __init__(self, profiles=None):
        """
        [BodyTracker V9.0 Dual]
        - Scans for 'student_seg_*.engine' and 'student_pose_*.engine' pairs.
        - Loads DualInferenceTRT
        """
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.model_dir = os.path.join(self.root_dir, "assets", "models", "personal")
        
        self.models = {}
        self.active_model = None
        self.active_profile = None
        
        print("[BRAIN] [BodyTracker] Dual-Engine Scan & Preload...")
        
        # 1. Scan Seg Engines
        seg_files = glob.glob(os.path.join(self.model_dir, "student_seg_*.engine"))
        
        if not seg_files:
            print("   [WARNING] No dual engine files found. Fallback mode?")
            # Legacy fallback omitted for clarity in this strict update
        else:
            for seg_path in seg_files:
                # Expect filename: student_seg_{profile}.engine
                # Construct pose path: student_pose_{profile}.engine
                basename = os.path.basename(seg_path)
                p_name = basename.replace("student_seg_", "").replace(".engine", "")
                
                pose_path = os.path.join(self.model_dir, f"student_pose_{p_name}.engine")
                
                if os.path.exists(pose_path):
                    print(f"   -> Loading Pair [{p_name}]...", end=" ")
                    try:
                        model = DualInferenceTRT(seg_path, pose_path)
                        if model.is_ready:
                            self.models[p_name] = model
                            print("OK")
                        else:
                            print("Failed (Not Ready)")
                    except Exception as e:
                        print(f"Error: {e}")
                else:
                    print(f"   [SKIP] Missing pose engine for {p_name}")

        # Set initial
        if 'default' in self.models:
            self.set_profile('default')
        elif len(self.models) > 0:
            first_key = list(self.models.keys())[0]
            self.set_profile(first_key)
            
        self.latest_mask = None

    def set_profile(self, profile_name):
        if profile_name in self.models:
            self.active_profile = profile_name
            self.active_model = self.models[profile_name]
            print(f"[BRAIN] [BodyTracker] Switched to: {profile_name}")
            return True
        return False

    def process(self, frame):
        if self.active_model is None or frame is None: return None
        mask, kpts = self.active_model.infer(frame)
        self.latest_mask = mask
        return kpts

    def get_mask(self):
        return self.latest_mask

    def draw_debug(self, frame, keypoints):
        if keypoints is None: return frame
        CONF_THRESH = 0.4
        for i in range(17):
            x, y, conf = keypoints[i]
            if conf > CONF_THRESH:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        return frame