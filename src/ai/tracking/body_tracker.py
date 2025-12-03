# Project MUSE - body_tracker.py
# Multi-Profile Auto-Scanner & Full Debug Draw
# (C) 2025 MUSE Corp. All rights reserved.

import os
import glob
import time
import numpy as np
import cv2

try:
    from ai.distillation.student.inference_trt import StudentInferenceTRT
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

from ai.distillation.student.inference import StudentInference

class BodyTracker:
    def __init__(self, profiles=None):
        """
        [BodyTracker V8.0]
        - Scans 'assets/models/personal/*.engine'
        - Preloads all found profiles
        """
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.model_dir = os.path.join(self.root_dir, "assets", "models", "personal")
        
        self.models = {}
        self.active_model = None
        self.active_profile = None
        
        print("🧠 [BodyTracker] 스캔 및 모델 프리로딩 시작...")
        
        # 1. Scan Engines
        engine_files = glob.glob(os.path.join(self.model_dir, "student_*.engine"))
        
        if not engine_files:
            print("   ⚠️ .engine 파일이 없습니다. PyTorch(CPU) 모드로 대체합니다.")
            self.models['default'] = StudentInference()
        else:
            for ef in engine_files:
                # filename: student_front.engine -> profile: front
                basename = os.path.basename(ef)
                p_name = basename.replace("student_", "").replace(".engine", "")
                
                print(f"   -> Loading [{p_name}]...", end=" ")
                model = StudentInferenceTRT(ef)
                if model.is_ready:
                    self.models[p_name] = model
                    print("OK")
                else:
                    print("Failed")

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
            print(f"🧠 [BodyTracker] Switched to: {profile_name}")
            return True
        else:
            # Fallback to default if exists
            if 'default' in self.models:
                self.active_profile = 'default'
                self.active_model = self.models['default']
                print(f"⚠️ [BodyTracker] '{profile_name}' not found. Using default.")
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
        """
        [Visual Check] 뼈대 그리기 (Full Logic Restored)
        """
        if keypoints is None:
            return frame

        CONF_THRESH = 0.4

        # 1. 점 찍기 (Joints)
        for i in range(17):
            x, y, conf = keypoints[i]
            h, w = frame.shape[:2]
            if x < 0 or x >= w or y < 0 or y >= h: continue

            if conf > CONF_THRESH:
                color = (255, 100, 0) if i % 2 == 1 else (0, 100, 255)
                if i <= 4: color = (0, 255, 255) # Face
                radius = 4 if i <= 4 else 6
                cv2.circle(frame, (int(x), int(y)), radius, color, -1)
                cv2.circle(frame, (int(x), int(y)), radius+1, (255, 255, 255), 1)

        # 2. 선 연결 (Skeleton)
        # COCO 17 Keypoints Format
        skeleton = [
            (5, 7), (7, 9),       # Left Arm
            (6, 8), (8, 10),      # Right Arm
            (11, 13), (13, 15),   # Left Leg
            (12, 14), (14, 16),   # Right Leg
            (5, 6),               # Shoulders
            (11, 12),             # Hips
            (5, 11), (6, 12),     # Torso
            (0, 1), (0, 2),       # Face (Nose to Eyes)
            (1, 3), (2, 4)        # Face (Eyes to Ears)
        ]

        for p1, p2 in skeleton:
            if p1 < len(keypoints) and p2 < len(keypoints):
                x1, y1, c1 = keypoints[p1]
                x2, y2, c2 = keypoints[p2]
                if c1 > CONF_THRESH and c2 > CONF_THRESH:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        return frame