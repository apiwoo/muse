# Project MUSE - vitpose_trt.py
# Target: RTX 3060/4090 Mode A (High Performance)
# (C) 2025 MUSE Corp. All rights reserved.

import tensorrt as trt
import cupy as cp
import numpy as np
import cv2
import os
import sys
import gc

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
        - V2.1 Update: Safe Loader (Memory Leak Fix)
        """
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"[ERROR] Engine file not found: {engine_path}\n-> Please run 'tools/trt_converter.py' to rebuild it.")

        # [Safety Check] File Size Validation
        file_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"[ViTPose] TensorRT Engine Loading: {os.path.basename(engine_path)} ({file_size_mb:.1f} MB)")
        
        # 엔진 파일이 비정상적으로 크거나 작으면 경고 (보통 500MB~2GB 사이)
        if file_size_mb < 10 or file_size_mb > 8192:
            print(f"[WARNING] Engine file size seems abnormal. If it hangs, delete {engine_path} and rebuild.")

        self.logger = trt.Logger(trt.Logger.WARNING)
        
        try:
            # 1. Load Engine to Memory
            # 파일을 한 번에 읽어서 변수에 담습니다.
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            
            # 2. Deserialize (CPU RAM -> GPU VRAM)
            # 런타임을 with 문으로 사용하여 리소스 자동 해제 유도
            with trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            # [CRITICAL] Delete raw binary from RAM immediately
            # GPU로 로딩이 끝났으므로, CPU 램에 있는 수 GB짜리 바이너리 데이터는 즉시 삭제합니다.
            del engine_data
            gc.collect() # 강제 가비지 컬렉션 실행
            # print("   [MEM] Raw engine data cleared from RAM.")
            
        except Exception as e:
            print(f"[CRITICAL] Failed to load TensorRT engine: {e}")
            print("-> The engine file might be corrupted or incompatible with current driver.")
            print("-> SOLUTION: Delete the .engine file and run 'tools/trt_converter.py'.")
            raise e

        if not self.engine:
            raise RuntimeError("[ERROR] Engine deserialization returned None. (Version Mismatch?)")

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
        
        print("[OK] [ViTPose] Engine Ready & Memory Cleaned")

    def inference(self, frame_bgr):
        """
        :param frame_bgr: (H, W, 3) BGR Image (CPU numpy)
        :return: keypoints (17, 3) -> [x, y, conf] (Original Scale)
        """
        if frame_bgr is None:
            return None

        h_orig, w_orig = frame_bgr.shape[:2]

        # [Step 1] Letterbox Resize (Maintain Aspect Ratio)
        # 1. Calculate scale
        scale = min(self.input_w / w_orig, self.input_h / h_orig)
        
        # 2. Resized dims
        nw = int(w_orig * scale)
        nh = int(h_orig * scale)
        
        # 3. Resize
        img_resized = cv2.resize(frame_bgr, (nw, nh))
        
        # 4. Padding (Gray background)
        # Create canvas (256, 192)
        img_canvas = np.full((self.input_h, self.input_w, 3), 127.5, dtype=np.uint8)
        
        # Calculate offset
        pad_w = (self.input_w - nw) // 2
        pad_h = (self.input_h - nh) // 2
        
        # Paste image
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
        
        # [Critical] Restore Coordinates (Heatmap -> Input -> Original)
        # 1. Heatmap(64x48) -> Input(256x192) : 4x scale
        kpts *= 4.0
        
        # 2. Remove Padding (Letterbox Inverse)
        kpts[..., 0] -= pad_w
        kpts[..., 1] -= pad_h
        
        # 3. Scale Back (Input -> Original)
        kpts /= scale
        
        # Combine [x, y, conf]
        result_kpts = cp.concatenate([kpts, max_vals], axis=-1)
        
        return cp.asnumpy(result_kpts[0])