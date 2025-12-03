# Project MUSE - inference_trt.py
# TensorRT Inference for SegFormer
# (C) 2025 MUSE Corp. All rights reserved.

import tensorrt as trt
import cupy as cp
import cv2
import os
import numpy as np

class StudentInferenceTRT:
    def __init__(self, engine_path):
        """
        [MUSE SegFormer TRT]
        - Input: 960x544
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        if not os.path.exists(engine_path):
            self.is_ready = False
            return

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        if not self.engine:
            self.is_ready = False
            return
            
        self.context = self.engine.create_execution_context()
        self.is_ready = True
        
        # Buffers
        self.input_h, self.input_w = 544, 960
        self.input_shape = (1, 3, self.input_h, self.input_w)
        
        # Output shapes (SegFormer)
        # seg: (1, 1, 544, 960), pose: (1, 17, 544, 960)
        
        self.d_input = cp.zeros(self.input_shape, dtype=cp.float32)
        self.d_seg = cp.zeros((1, 1, self.input_h, self.input_w), dtype=cp.float32)
        self.d_pose = cp.zeros((1, 17, self.input_h, self.input_w), dtype=cp.float32)
        
        self.bindings = [int(self.d_input.data.ptr), int(self.d_seg.data.ptr), int(self.d_pose.data.ptr)]
        
        self.mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32).reshape(1,3,1,1)
        self.std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32).reshape(1,3,1,1)

    def infer(self, frame_bgr):
        if not self.is_ready or frame_bgr is None: return None, None
        
        h_orig, w_orig = frame_bgr.shape[:2]
        
        # Preprocess
        img_resized = cv2.resize(frame_bgr, (self.input_w, self.input_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        img_gpu = cp.asarray(img_rgb).transpose(2,0,1).astype(cp.float32) / 255.0
        img_gpu = img_gpu.reshape(1, 3, self.input_h, self.input_w)
        img_gpu = (img_gpu - self.mean) / self.std
        
        cp.copyto(self.d_input, img_gpu)
        
        # Execute
        self.context.execute_v2(bindings=self.bindings)
        
        # Postprocess
        # Seg
        mask_gpu = 1.0 / (1.0 + cp.exp(-self.d_seg)) # Sigmoid
        mask_gpu = (mask_gpu > 0.5).astype(cp.float32).squeeze()
        mask_cpu = cp.asnumpy(mask_gpu * 255).astype(np.uint8)
        mask_final = cv2.resize(mask_cpu, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        # Pose
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
            if max_val > 0.1:
                x = max_loc[0] * scale_x
                y = max_loc[1] * scale_y
                kpts.append([x, y, max_val])
            else:
                kpts.append([0, 0, 0.0])
        return np.array(kpts)