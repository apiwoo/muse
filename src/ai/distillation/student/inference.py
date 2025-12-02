# Project MUSE - inference.py
# The "Student" Inference Engine: Lightweight & Real-time (PyTorch)
# (C) 2025 MUSE Corp. All rights reserved.

import torch
import cv2
import numpy as np
import os
import sys

# 프로젝트 루트 경로 확보
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from ai.distillation.student.model_arch import MuseStudentModel

class StudentInference:
    def __init__(self, model_path=None):
        """
        [MUSE Student Inference - PyTorch Fallback]
        - High-Fidelity Mode: 960x544
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎓 [Student] 추론 엔진 초기화 (Device: {self.device})")

        # 1. 모델 아키텍처 준비 (ResNet-34 U-Net)
        self.model = MuseStudentModel(num_keypoints=17).to(self.device)
        self.model.eval()

        # 2. 가중치 로드
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            model_path = os.path.join(base_dir, "assets", "models", "personal", "student_model_final.pth")

        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"   ✅ 모델 로드 성공: {os.path.basename(model_path)}")
                self.is_ready = True
            except Exception as e:
                print(f"   ❌ 모델 로드 실패: {e}")
                self.is_ready = False
        else:
            print(f"   ⚠️ 모델 파일이 없습니다: {model_path}")
            self.is_ready = False

        # 추론용 상수 (High-Fidelity)
        self.input_w = 960
        self.input_h = 544
        self.input_size = (self.input_w, self.input_h)
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def infer(self, frame_bgr):
        """
        :param frame_bgr: (H, W, 3) OpenCV Image
        :return: (mask_binary, keypoints)
        """
        if not self.is_ready or frame_bgr is None:
            return None, None

        h_orig, w_orig = frame_bgr.shape[:2]

        # 1. Preprocess (Resize & Normalize)
        img_resized = cv2.resize(frame_bgr, self.input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # To Tensor
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        img_tensor = (img_tensor - self.mean) / self.std

        # 2. Inference
        with torch.no_grad():
            pred_seg, pred_pose = self.model(img_tensor)
            
            # --- Output 1: Segmentation ---
            # Sigmoid & Threshold
            mask_prob = torch.sigmoid(pred_seg)
            mask_tensor = (mask_prob > 0.5).float().squeeze().cpu().numpy()
            
            # Resize back to original
            mask_uint8 = (mask_tensor * 255).astype(np.uint8)
            mask_final = cv2.resize(mask_uint8, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
            # --- Output 2: Pose ---
            # Heatmaps are now same resolution as input (1/1 scale)
            heatmaps = pred_pose.squeeze().cpu().numpy() # (17, 544, 960)
            keypoints = self._parse_heatmaps(heatmaps, (w_orig, h_orig))

        return mask_final, keypoints

    def _parse_heatmaps(self, heatmaps, original_size):
        """히트맵에서 최대값 좌표(x, y)를 추출하고 원본 크기로 복원"""
        kpts = []
        w_orig, h_orig = original_size
        _, h_map, w_map = heatmaps.shape # (17, 544, 960)
        
        scale_x = w_orig / w_map
        scale_y = h_orig / h_map

        for i in range(17):
            hm = heatmaps[i]
            # Find max loc
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hm)
            
            if max_val > 0.1: # Threshold
                x = max_loc[0] * scale_x
                y = max_loc[1] * scale_y
                kpts.append([x, y, max_val])
            else:
                kpts.append([0, 0, 0.0]) # Not found
                
        return np.array(kpts)