# Project MUSE - inference.py
# Student Inference (SegFormer) - PyTorch Fallback
# (C) 2025 MUSE Corp. All rights reserved.

import torch
import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from ai.distillation.student.model_arch import MuseStudentModel

class StudentInference:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎓 [Student] PyTorch Inference Init (SegFormer)...")

        self.model = MuseStudentModel(num_keypoints=17, pretrained=False).to(self.device)
        self.model.eval()

        if model_path is None:
            # Default fallback
            model_path = "assets/models/personal/student_default.pth"

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.is_ready = True
        else:
            print(f"   ⚠️ Model not found: {model_path}")
            self.is_ready = False

        self.input_size = (960, 544)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def infer(self, frame_bgr):
        if not self.is_ready or frame_bgr is None: return None, None

        h_orig, w_orig = frame_bgr.shape[:2]
        img_resized = cv2.resize(frame_bgr, self.input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        img_tensor = (img_tensor - self.mean) / self.std

        with torch.no_grad():
            pred_seg, pred_pose = self.model(img_tensor)
            
            # SegFormer는 Output이 1/4 Scale 일 수 있으나 Model에서 Interpolate함
            mask_prob = torch.sigmoid(pred_seg)
            mask_tensor = (mask_prob > 0.5).float().squeeze().cpu().numpy()
            mask_uint8 = (mask_tensor * 255).astype(np.uint8)
            mask_final = cv2.resize(mask_uint8, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
            heatmaps = pred_pose.squeeze().cpu().numpy()
            keypoints = self._parse_heatmaps(heatmaps, (w_orig, h_orig))

        return mask_final, keypoints

    def _parse_heatmaps(self, heatmaps, original_size):
        kpts = []
        w_orig, h_orig = original_size
        _, h_map, w_map = heatmaps.shape
        scale_x = w_orig / w_map
        scale_y = h_orig / h_map

        for i in range(17):
            hm = heatmaps[i]
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hm)
            if max_val > 0.1:
                x = max_loc[0] * scale_x
                y = max_loc[1] * scale_y
                kpts.append([x, y, max_val])
            else:
                kpts.append([0, 0, 0.0])
        return np.array(kpts)