# Project MUSE - inference_trt.py
# Dual TensorRT Inference (Seg & Pose Split)
# Optimized: Full GPU Pipeline (Zero-Copy) using Pure CuPy
# Updated: Hybrid Resolution (544p Seg / 352p Pose)
# Fixed: Pose Output Buffer Sizing (Full Res)
# (C) 2025 MUSE Corp. All rights reserved.

import tensorrt as trt
import cupy as cp
import os
import numpy as np

# [Note] Removing cv2 dependency for core inference to ensure GPU purity.
# If needed for debugging, import it locally.

class DualInferenceTRT:
    """
    [Dual Engine Runtime]
    Simultaneously runs two TensorRT engines (Segmentation + Pose)
    Input is processed twice (544p for Seg, 352p for Pose).
    """
    def __init__(self, seg_engine_path, pose_engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.is_ready = False
        
        # [Config] Resolutions
        self.seg_h, self.seg_w = 544, 960
        self.pose_h, self.pose_w = 352, 640
        
        self.seg_shape = (1, 3, self.seg_h, self.seg_w)
        self.pose_shape = (1, 3, self.pose_h, self.pose_w)
        
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
        # Input Buffers (Separate)
        self.d_input_seg = cp.zeros(self.seg_shape, dtype=cp.float32)
        self.d_input_pose = cp.zeros(self.pose_shape, dtype=cp.float32)
        
        # Output Buffers
        # Seg: (1, 1, 544, 960)
        self.d_seg = cp.zeros((1, 1, self.seg_h, self.seg_w), dtype=cp.float32)
        
        # [CRITICAL FIX] Pose Output Buffer Size
        # Model architecture uses interpolate(scale=4), so output matches input resolution (352x640).
        # Previous code assumed 1/4 scale (88x160) which caused cudaErrorIllegalAddress.
        self.d_pose = cp.zeros((1, 17, self.pose_h, self.pose_w), dtype=cp.float32)
        
        # Bindings
        # Assuming model export order: input (0) -> output (1)
        self.bindings_seg = [int(self.d_input_seg.data.ptr), int(self.d_seg.data.ptr)]
        self.bindings_pose = [int(self.d_input_pose.data.ptr), int(self.d_pose.data.ptr)]

    def _resize_gpu_nearest(self, img, new_h, new_w):
        """
        Pure CuPy Nearest Neighbor Resizing.
        Fastest method, sufficient for model input.
        img: (H, W, C)
        """
        h, w = img.shape[:2]
        
        # Generate grid
        # We want to sample from the original image at these coordinates
        # Using integer division for nearest neighbor behavior
        
        # y_indices: (new_h, 1)
        y_indices = (cp.arange(new_h, dtype=cp.int32) * h // new_h).reshape(-1, 1)
        # x_indices: (1, new_w)
        x_indices = (cp.arange(new_w, dtype=cp.int32) * w // new_w).reshape(1, -1)
        
        # Broadcast to (new_h, new_w, C)
        resized = img[y_indices, x_indices, :]
        return resized

    def infer(self, frame_input):
        """
        :param frame_input: Could be CPU numpy or GPU cupy array (BGR)
        :return: mask_gpu (1080p), keypoints_cpu
        """
        if not self.is_ready or frame_input is None: return None, None
        
        # 1. Input Handling (Zero-Copy if possible)
        if hasattr(frame_input, 'device'):
            img_gpu_src = frame_input
            h_orig, w_orig = frame_input.shape[:2]
        else:
            # Fallback: Upload CPU -> GPU
            img_gpu_src = cp.asarray(frame_input)
            h_orig, w_orig = frame_input.shape[:2]

        # ----------------------------------------------------
        # PATH A: Segmentation (544p)
        # ----------------------------------------------------
        img_seg = self._resize_gpu_nearest(img_gpu_src, self.seg_h, self.seg_w)
        
        # BGR -> RGB & Normalize
        img_pre_seg = img_seg[..., ::-1].transpose(2,0,1).astype(cp.float32) / 255.0
        img_pre_seg = img_pre_seg.reshape(1, 3, self.seg_h, self.seg_w)
        img_pre_seg = (img_pre_seg - self.mean) / self.std
        
        cp.copyto(self.d_input_seg, img_pre_seg)
        self.ctx_seg.execute_v2(bindings=self.bindings_seg)
        
        # Postprocess Seg (Sigmoid)
        mask_prob = 1.0 / (1.0 + cp.exp(-self.d_seg)) 
        mask_small = (mask_prob > 0.5).astype(cp.float32).squeeze() 
        
        # Upscale Mask to 1080p
        mask_small_3d = mask_small[..., None]
        mask_final_gpu = self._resize_gpu_nearest(mask_small_3d, h_orig, w_orig)
        mask_final_gpu = mask_final_gpu.squeeze()

        # ----------------------------------------------------
        # PATH B: Pose (352p)
        # ----------------------------------------------------
        img_pose = self._resize_gpu_nearest(img_gpu_src, self.pose_h, self.pose_w)
        
        img_pre_pose = img_pose[..., ::-1].transpose(2,0,1).astype(cp.float32) / 255.0
        img_pre_pose = img_pre_pose.reshape(1, 3, self.pose_h, self.pose_w)
        img_pre_pose = (img_pre_pose - self.mean) / self.std
        
        cp.copyto(self.d_input_pose, img_pre_pose)
        self.ctx_pose.execute_v2(bindings=self.bindings_pose)
        
        # Postprocess Pose (CPU for Parsing)
        heatmaps = cp.asnumpy(self.d_pose.squeeze())
        keypoints = self._parse_heatmaps(heatmaps, (w_orig, h_orig))
        
        return mask_final_gpu, keypoints

    def _parse_heatmaps(self, heatmaps, original_size):
        kpts = []
        w_orig, h_orig = original_size
        _, h_map, w_map = heatmaps.shape 
        
        scale_x = w_orig / w_map
        scale_y = h_orig / h_map
        
        import cv2 
        
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