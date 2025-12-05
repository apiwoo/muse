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
        - V2.0 Update: Letterbox Preprocessing (비율 왜곡 방지)
        """
        print(f"[ViTPose] TensorRT 엔진 로딩 중: {os.path.basename(engine_path)}")
        
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
        self.input_h, self.input_w = 256, 192
        self.input_shape = (1, 3, self.input_h, self.input_w)
        
        # Output: (1, 17, 64, 48) -> Heatmaps (1/4 scale)
        self.output_shape = (1, 17, 64, 48)
        
        # GPU Buffers
        self.d_input = cp.zeros(self.input_shape, dtype=cp.float32)
        self.d_output = cp.zeros(self.output_shape, dtype=cp.float32)
        
        # Binding
        self.bindings = [int(self.d_input.data.ptr), int(self.d_output.data.ptr)]
        
        # Preprocessing Constants (ImageNet Mean/Std)
        self.mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32).reshape(1, 3, 1, 1)
        self.std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32).reshape(1, 3, 1, 1)
        
        print("✅ [ViTPose] 엔진 초기화 완료 (Ready)")

    def inference(self, frame_bgr):
        """
        :param frame_bgr: (H, W, 3) BGR Image (CPU numpy)
        :return: keypoints (17, 3) -> [x, y, conf] (Original Scale)
        """
        if frame_bgr is None:
            return None

        h_orig, w_orig = frame_bgr.shape[:2]

        # [Step 1] Letterbox Resize (비율 유지)
        # 1. 스케일 계산 (가로/세로 중 더 많이 줄여야 하는 쪽 기준)
        scale = min(self.input_w / w_orig, self.input_h / h_orig)
        
        # 2. 리사이즈된 크기
        nw = int(w_orig * scale)
        nh = int(h_orig * scale)
        
        # 3. 리사이즈 수행
        img_resized = cv2.resize(frame_bgr, (nw, nh))
        
        # 4. 패딩 (회색 배경)
        # 캔버스 생성 (256, 192)
        img_canvas = np.full((self.input_h, self.input_w, 3), 127.5, dtype=np.uint8)
        
        # 중앙 정렬을 위한 오프셋 계산
        pad_w = (self.input_w - nw) // 2
        pad_h = (self.input_h - nh) // 2
        
        # 이미지 붙여넣기
        img_canvas[pad_h:pad_h+nh, pad_w:pad_w+nw] = img_resized

        # [Step 2] To GPU & Normalize
        img_gpu = cp.asarray(img_canvas)
        img_gpu = img_gpu[..., ::-1] # BGR -> RGB
        
        # (H, W, C) -> (B, C, H, W)
        img_gpu = img_gpu.transpose(2, 0, 1).astype(cp.float32) / 255.0
        img_gpu = img_gpu.reshape(1, 3, self.input_h, self.input_w)
        img_gpu = (img_gpu - self.mean) / self.std
        
        cp.copyto(self.d_input, img_gpu)

        # [Step 3] Inference
        self.context.execute_v2(bindings=self.bindings)

        # [Step 4] Post-process (Heatmap -> Keypoints)
        heatmaps = self.d_output # GPU array
        
        # Flatten for argmax
        heatmaps_flat = heatmaps.reshape(1, 17, -1)
        
        # Confidence
        max_vals = cp.amax(heatmaps_flat, axis=2).reshape(1, 17, 1)
        
        # Coordinates (in Heatmap 64x48 scale)
        max_inds = cp.argmax(heatmaps_flat, axis=2)
        
        w_heat = 48 # Heatmap width
        y_heat = max_inds // w_heat
        x_heat = max_inds % w_heat
        
        # Stack [x, y]
        kpts = cp.stack([x_heat, y_heat], axis=-1).astype(cp.float32)
        
        # [Critical] 좌표 복원 (Heatmap -> Input -> Original)
        # 1. Heatmap(64x48) -> Input(256x192) : 4배 확대
        kpts *= 4.0
        
        # 2. Remove Padding (Letterbox 역연산)
        kpts[..., 0] -= pad_w
        kpts[..., 1] -= pad_h
        
        # 3. Scale Back (Input -> Original)
        kpts /= scale
        
        # Combine [x, y, conf]
        result_kpts = cp.concatenate([kpts, max_vals], axis=-1)
        
        return cp.asnumpy(result_kpts[0])