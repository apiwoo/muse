# Project MUSE - test_segmentation_comparison.py
# (C) 2025 MUSE Corp. All rights reserved.
# Purpose: Compare Real-time Matting/Segmentation Models
# Target: Boundary Quality (Alpha Blending) Check

import cv2
import numpy as np
import time
import os
import sys
import torch
from torchvision import transforms
from PIL import Image

# 3rd Party Libs
try:
    import mediapipe as mp
    import onnxruntime as ort
except ImportError:
    print("[ERROR] Required libs: mediapipe, onnxruntime")
    sys.exit(1)

# Paths
current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_file))
modnet_path = os.path.join(root_dir, "assets", "models", "segmentation", "modnet.onnx")

class SegmentationTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Device: {self.device}")
        
        self.models = {}
        self.current_model_idx = 0
        self.model_names = ["MediaPipe Selfie", "MODNet (ONNX)", "DeepLabV3+ (MobileNet)"]
        
        # View Mode: 0=Composite, 1=Alpha Mask, 2=Edge Heatmap
        self.view_mode = 0 
        self.view_names = ["Composite (Green)", "Alpha Mask (BW)", "Edge Heatmap (Red)"]

        # Init Models
        self._init_mediapipe()
        self._init_modnet()
        self._init_deeplab()

    def _init_mediapipe(self):
        print("⏳ [1/3] Loading MediaPipe...")
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.mp_model = self.mp_selfie.SelfieSegmentation(model_selection=1) # 1=Landscape(High accuracy)
        self.models[0] = self.infer_mediapipe

    def _init_modnet(self):
        print(f"⏳ [2/3] Loading MODNet ({modnet_path})...")
        if os.path.exists(modnet_path):
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.modnet_session = ort.InferenceSession(modnet_path, providers=providers)
                self.modnet_input_name = self.modnet_session.get_inputs()[0].name
                self.models[1] = self.infer_modnet
            except Exception as e:
                print(f"   ❌ MODNet Load Failed: {e}")
        else:
            print("   ⚠️ MODNet file missing. Run 'tools/download_models.py'")

    def _init_deeplab(self):
        print("⏳ [3/3] Loading DeepLabV3+ (MobileNetV3)...")
        try:
            # Load PyTorch Hub Model (Lightweight)
            self.deeplab = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
            self.deeplab.to(self.device)
            self.deeplab.eval()
            self.deeplab_preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.models[2] = self.infer_deeplab
        except Exception as e:
            print(f"   ❌ DeepLab Load Failed: {e}")

    # ================= INFERENCE LOGIC =================

    def infer_mediapipe(self, frame):
        # MediaPipe expects RGB, returns 0~1 float mask
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_model.process(frame_rgb)
        mask = results.segmentation_mask # (H, W) float
        return mask

    def infer_modnet(self, frame):
        # MODNet expects (1, 3, 512, 512), range [-1, 1] usually or normalized
        # This ONNX specific version usually likes 512x512
        h, w = frame.shape[:2]
        target_size = 512
        
        img = cv2.resize(frame, (target_size, target_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] then subtract mean/std if needed, 
        # but standardized MODNet ONNX often takes [0,1] or [-0.5, 0.5]
        # Let's try standard range correction for this ONNX: (img - 127.5) / 127.5
        img = (img.astype(np.float32) - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1)) # CHW
        img = np.expand_dims(img, axis=0)  # BCHW
        
        pred = self.modnet_session.run(None, {self.modnet_input_name: img})[0]
        # Pred shape: (1, 1, 512, 512)
        
        mask = pred[0, 0]
        mask = cv2.resize(mask, (w, h))
        return mask

    def infer_deeplab(self, frame):
        # DeepLab expects Tensor (B, C, H, W)
        h, w = frame.shape[:2]
        
        # Resize for speed (optional, but 1080p is slow for DeepLab)
        infer_h, infer_w = 520, 520
        frame_resized = cv2.resize(frame, (infer_w, infer_h))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.deeplab_preprocess(frame_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.deeplab(input_tensor)['out'][0]
        
        output_predictions = output.argmax(0).byte().cpu().numpy()
        # Class 15 is Person in COCO/Pascal
        mask = (output_predictions == 15).astype(np.float32)
        
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    # ================= VISUALIZATION =================

    def visualize(self, frame, mask):
        h, w = frame.shape[:2]
        
        # Ensure mask is float 0..1
        if mask.max() > 1.0: mask = mask / 255.0
        mask = np.clip(mask, 0, 1)

        # 1. Composite (Green Screen)
        if self.view_mode == 0:
            bg_color = np.array([0, 255, 0], dtype=np.uint8) # Green
            bg_img = np.full((h, w, 3), bg_color, dtype=np.uint8)
            
            mask_3ch = np.dstack((mask, mask, mask))
            
            # Alpha Blending: src * alpha + bg * (1 - alpha)
            foreground = frame.astype(np.float32) * mask_3ch
            background = bg_img.astype(np.float32) * (1.0 - mask_3ch)
            combined = (foreground + background).astype(np.uint8)
            return combined

        # 2. Alpha Mask (B/W)
        elif self.view_mode == 1:
            mask_u8 = (mask * 255).astype(np.uint8)
            return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

        # 3. Edge Heatmap (Highlight Semi-transparent areas)
        elif self.view_mode == 2:
            # 경계면: 0.0과 1.0이 아닌 중간값(0.1 ~ 0.9)을 가진 픽셀들
            # Matting 모델(MODNet)은 이게 넓게 나오고, Segmentation(DeepLab)은 거의 없음.
            
            # Edge Intensity: 0.5에서 가장 붉게, 0이나 1에 가까울수록 검게
            # Math: 1.0 - |2 * (mask - 0.5)|  -> 0.5일때 1.0, 0 or 1일때 0.0
            edge_val = 1.0 - np.abs(2.0 * (mask - 0.5))
            edge_val = np.clip(edge_val, 0, 1) ** 2 # Contrast
            
            edge_u8 = (edge_val * 255).astype(np.uint8)
            
            # Red Heatmap
            heatmap = np.zeros_like(frame)
            heatmap[:, :, 2] = edge_u8 # Red Channel
            
            # 원본 위에 오버레이
            return cv2.addWeighted(frame, 0.5, heatmap, 2.0, 0)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n==============================================")
        print("   MUSE Segmentation Comparison Tool")
        print("==============================================")
        print("   [1] MediaPipe (Fastest, Smoothing)")
        print("   [2] MODNet (Matting Specialized)")
        print("   [3] DeepLabV3+ (Semantic Segmentation)")
        print("   --------------------------------------")
        print("   [M] Toggle View Mode (Composite -> Mask -> Edge)")
        print("   [Q] Quit")
        print("==============================================\n")

        prev_time = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            start_t = time.time()
            
            # Inference
            if self.current_model_idx in self.models:
                mask = self.models[self.current_model_idx](frame)
            else:
                mask = np.zeros(frame.shape[:2], dtype=np.float32) # Fallback

            # Inference Time
            infer_ms = (time.time() - start_t) * 1000
            
            # Visualize
            display = self.visualize(frame, mask)
            
            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            # HUD
            model_name = self.model_names[self.current_model_idx]
            view_name = self.view_names[self.view_mode]
            
            # Text UI
            cv2.rectangle(display, (0, 0), (600, 80), (0, 0, 0), -1)
            cv2.putText(display, f"Model [1-3]: {model_name}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"View [M]: {view_name}", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"FPS: {fps:.1f} | Infer: {infer_ms:.1f}ms", (400, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("MUSE Segmentation Test", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('1'): self.current_model_idx = 0
            elif key == ord('2'): self.current_model_idx = 1
            elif key == ord('3'): self.current_model_idx = 2
            elif key == ord('m'): self.view_mode = (self.view_mode + 1) % 3

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester = SegmentationTester()
    tester.run()