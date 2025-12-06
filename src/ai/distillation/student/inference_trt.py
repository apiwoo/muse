# Project MUSE - inference_trt.py
# Dual TensorRT Inference (Seg & Pose Split)
# (C) 2025 MUSE Corp. All rights reserved.

import tensorrt as trt
import cupy as cp
import cv2
import os
import numpy as np

class DualInferenceTRT:
    """
    [Dual Engine Runtime]
    Simultaneously runs two TensorRT engines (Segmentation + Pose)
    Input is shared (uploaded once), Outputs are separate.
    """
    def __init__(self, seg_engine_path, pose_engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.is_ready = False
        
        self.input_h, self.input_w = 544, 960
        self.input_shape = (1, 3, self.input_h, self.input_w)
        
        # Preprocessing constants
        self.mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32).reshape(1,3,1,1)
        self.std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32).reshape(1,3,1,1)

        # 1. Load Seg Engine
        self.engine_seg = self._load_engine(seg_engine_path)
        self.ctx_seg = self.engine_seg.create_execution_context() if self.engine_seg else None
        
        # 2. Load Pose Engine
        self.engine_pose = self._load_engine(pose_engine_path)
        self.ctx_pose = self.engine_pose.create_execution_context() if self.engine_pose else None
        
        if self.ctx_seg and self.ctx_pose:
            self.is_ready = True
            self._allocate_buffers()
        else:
            print("[ERROR] Failed to load one or both engines.")

    def _load_engine(self, path):
        if not os.path.exists(path): return None
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        # Shared Input Buffer
        self.d_input = cp.zeros(self.input_shape, dtype=cp.float32)
        
        # Output Buffers
        # Seg: (1, 1, H, W)
        self.d_seg = cp.zeros((1, 1, self.input_h, self.input_w), dtype=cp.float32)
        # Pose: (1, 17, H, W)
        self.d_pose = cp.zeros((1, 17, self.input_h, self.input_w), dtype=cp.float32)
        
        # Bindings
        # Assuming model export order: input (0) -> output (1)
        self.bindings_seg = [int(self.d_input.data.ptr), int(self.d_seg.data.ptr)]
        self.bindings_pose = [int(self.d_input.data.ptr), int(self.d_pose.data.ptr)]

    def infer(self, frame_bgr):
        if not self.is_ready or frame_bgr is None: return None, None
        
        h_orig, w_orig = frame_bgr.shape[:2]
        
        # 1. Preprocess (One time)
        img_resized = cv2.resize(frame_bgr, (self.input_w, self.input_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        img_gpu = cp.asarray(img_rgb).transpose(2,0,1).astype(cp.float32) / 255.0
        img_gpu = img_gpu.reshape(1, 3, self.input_h, self.input_w)
        img_gpu = (img_gpu - self.mean) / self.std
        
        cp.copyto(self.d_input, img_gpu)
        
        # 2. Execute (Sequential execution on GPU is very fast)
        # Could use CUDA streams for parallel execution, but sequential is fine for this size
        self.ctx_seg.execute_v2(bindings=self.bindings_seg)
        self.ctx_pose.execute_v2(bindings=self.bindings_pose)
        
        # 3. Postprocess Seg
        mask_gpu = 1.0 / (1.0 + cp.exp(-self.d_seg)) # Sigmoid
        mask_gpu = (mask_gpu > 0.5).astype(cp.float32).squeeze()
        mask_cpu = cp.asnumpy(mask_gpu * 255).astype(np.uint8)
        mask_final = cv2.resize(mask_cpu, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        # 4. Postprocess Pose
        heatmaps = cp.asnumpy(self.d_pose.squeeze())
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
            
            # Threshold 0.05
            if max_val > 0.05:
                x = max_loc[0] * scale_x
                y = max_loc[1] * scale_y
                kpts.append([x, y, max_val])
            else:
                kpts.append([0, 0, 0.0])
        return np.array(kpts)