# Project MUSE - vitpose_trt.py
# Target: RTX 3060/4090 Mode A (High Performance)
# (C) 2025 MUSE Corp. All rights reserved.

import tensorrt as trt
import cupy as cp
import numpy as np
import cv2
import os
import sys

# Windows DLL Path Fix
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from utils.cuda_helper import setup_cuda_environment
    setup_cuda_environment()
except ImportError:
    pass

class VitPoseTrt:
    def __init__(self, engine_path="assets/models/tracking/vitpose_huge.engine"):
        """
        [High-End] ViTPose TensorRT Inference Engine
        - Backend: TensorRT 10.x + CuPy (Zero-Copy)
        - Model: ViT-Huge (COCO 17 Keypoints)
        """
        print(f"🚀 [ViTPose] TensorRT 엔진 로딩 중: {os.path.basename(engine_path)}")
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 1. Load Engine
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"❌ 엔진 파일이 없습니다: {engine_path}")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if not self.engine:
            raise RuntimeError("❌ 엔진 역직렬화 실패")

        self.context = self.engine.create_execution_context()
        
        # 2. Allocate Buffers (using CuPy for GPU Memory)
        # Input: (1, 3, 256, 192) -> NCHW
        self.input_shape = (1, 3, 256, 192)
        self.input_dtype = np.float32
        self.input_size = np.prod(self.input_shape) * 4 # bytes
        
        # Output: (1, 17, 64, 48) -> Heatmaps (1/4 scale)
        self.output_shape = (1, 17, 64, 48)
        self.output_size = np.prod(self.output_shape) * 4
        
        # GPU Buffers
        self.d_input = cp.zeros(self.input_shape, dtype=cp.float32)
        self.d_output = cp.zeros(self.output_shape, dtype=cp.float32)
        
        # Binding (TensorRT 10.x style)
        # 0: input, 1: output
        self.bindings = [int(self.d_input.data.ptr), int(self.d_output.data.ptr)]
        
        # Preprocessing Constants (ImageNet Mean/Std)
        self.mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32).reshape(1, 3, 1, 1)
        self.std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32).reshape(1, 3, 1, 1)
        
        print("✅ [ViTPose] 엔진 초기화 완료 (Ready)")

    def inference(self, frame_bgr):
        """
        :param frame_bgr: (H, W, 3) BGR Image (CPU numpy or GPU cupy)
        :return: keypoints (17, 3) -> [x, y, conf] (Original Scale)
        """
        if frame_bgr is None:
            return None

        h_orig, w_orig = frame_bgr.shape[:2]

        # 1. Preprocess (Resize & Normalize)
        # Resize to (192, 256) -> (W, H)
        # Note: cv2.resize takes (W, H)
        img_resized = cv2.resize(frame_bgr, (192, 256))
        
        # To GPU & RGB Conversion
        img_gpu = cp.asarray(img_resized)
        img_gpu = img_gpu[..., ::-1] # BGR -> RGB
        
        # (H, W, C) -> (B, C, H, W) & Normalize
        img_gpu = img_gpu.transpose(2, 0, 1).astype(cp.float32) / 255.0
        img_gpu = img_gpu.reshape(1, 3, 256, 192)
        img_gpu = (img_gpu - self.mean) / self.std
        
        # Copy to input buffer (Flatten for safety if needed, but shape matches)
        cp.copyto(self.d_input, img_gpu)

        # 2. Inference
        self.context.execute_v2(bindings=self.bindings)

        # 3. Post-process (Heatmap -> Keypoints)
        # Heatmaps: (1, 17, 64, 48)
        heatmaps = self.d_output # GPU array
        
        # Find max location in each heatmap (17 channels)
        # Flatten H,W dimensions to find argmax easily
        # (1, 17, 3072)
        heatmaps_flat = heatmaps.reshape(1, 17, -1)
        
        # Max Values (Confidence)
        max_vals = cp.amax(heatmaps_flat, axis=2) # (1, 17)
        max_vals = max_vals.reshape(17, 1)
        
        # Max Indices
        max_inds = cp.argmax(heatmaps_flat, axis=2) # (1, 17)
        
        # Convert indices to (x, y)
        # Width of heatmap is 48
        w_heat = 48
        y_heat = max_inds // w_heat
        x_heat = max_inds % w_heat
        
        # Stack [x, y, conf]
        # Coordinates are in heatmap scale (64x48).
        # We need to scale them back to original image size.
        # Scale factors: Input (256x192) -> Heatmap (64x48) is 1/4
        # So Heatmap -> Input is x4
        # Input -> Original is (w_orig/192, h_orig/256)
        
        scale_x = (w_orig / 192.0) * 4.0
        scale_y = (h_orig / 256.0) * 4.0
        
        kpts = cp.stack([x_heat, y_heat], axis=1).astype(cp.float32)
        kpts[:, 0] *= scale_x
        kpts[:, 1] *= scale_y
        
        # Combine with confidence
        # Result: (17, 3)
        result_kpts = cp.hstack([kpts, max_vals])
        
        # Return as CPU numpy for compatibility with other modules
        return cp.asnumpy(result_kpts)