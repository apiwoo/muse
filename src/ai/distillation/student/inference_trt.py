# Project MUSE - inference_trt.py
# Dual TensorRT Inference (Parallel Execution)
# Optimized: CUDA Streams for Concurrent Seg & Pose
# Compatibility: TensorRT 10.x (execute_async_v3)
# (C) 2025 MUSE Corp. All rights reserved.

import tensorrt as trt
import cupy as cp
import os
import numpy as np
import cv2

class DualInferenceTRT:
    """
    [Dual Engine Runtime - Parallel Edition]
    Runs Segmentation and Pose estimation simultaneously using CUDA Streams.
    Expected Speedup: AI Latency 30ms -> ~18-20ms
    """
    def __init__(self, seg_engine_path, pose_engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.is_ready = False
        
        # [Config] Resolutions
        # Seg: 544p (Detailed Boundary)
        # Pose: 352p (Structural Core)
        self.seg_h, self.seg_w = 544, 960
        self.pose_h, self.pose_w = 352, 640
        
        self.seg_shape = (1, 3, self.seg_h, self.seg_w)
        self.pose_shape = (1, 3, self.pose_h, self.pose_w)
        
        # Preprocessing constants (GPU)
        self.mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32).reshape(1,3,1,1)
        self.std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32).reshape(1,3,1,1)

        # 1. Load Engines
        self.engine_seg = self._load_engine(seg_engine_path)
        self.engine_pose = self._load_engine(pose_engine_path)
        
        self.ctx_seg = self.engine_seg.create_execution_context() if self.engine_seg else None
        self.ctx_pose = self.engine_pose.create_execution_context() if self.engine_pose else None
        
        if self.ctx_seg and self.ctx_pose:
            self.is_ready = True
            self._allocate_buffers()
            # [Optimization] Create separate streams for parallelism
            self.stream_seg = cp.cuda.Stream(non_blocking=True)
            self.stream_pose = cp.cuda.Stream(non_blocking=True)
            print("[TRT] Dual Streams Initialized (Parallel Mode + TRT 10 Fixed)")
        else:
            print("[ERROR] Failed to load one or both engines.")

    def _load_engine(self, path):
        if not os.path.exists(path): return None
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        # Input Buffers
        self.d_input_seg = cp.zeros(self.seg_shape, dtype=cp.float32)
        self.d_input_pose = cp.zeros(self.pose_shape, dtype=cp.float32)
        
        # Output Buffers
        # Seg Output: (1, 1, 544, 960) -> Logits
        self.d_seg = cp.zeros((1, 1, self.seg_h, self.seg_w), dtype=cp.float32)
        # Pose Output: (1, 17, 352, 640) -> Heatmaps (Full scale due to bilinear upsampling in model)
        self.d_pose = cp.zeros((1, 17, self.pose_h, self.pose_w), dtype=cp.float32)
        
        # [TRT 10 Fix] Set Tensor Addresses explicitly
        # Segmentation Context
        name_in_seg = self.engine_seg.get_tensor_name(0)
        name_out_seg = self.engine_seg.get_tensor_name(1)
        self.ctx_seg.set_tensor_address(name_in_seg, int(self.d_input_seg.data.ptr))
        self.ctx_seg.set_tensor_address(name_out_seg, int(self.d_seg.data.ptr))

        # Pose Context
        name_in_pose = self.engine_pose.get_tensor_name(0)
        name_out_pose = self.engine_pose.get_tensor_name(1)
        self.ctx_pose.set_tensor_address(name_in_pose, int(self.d_input_pose.data.ptr))
        self.ctx_pose.set_tensor_address(name_out_pose, int(self.d_pose.data.ptr))

    def _resize_gpu_nearest(self, img, new_h, new_w):
        """Pure CuPy Nearest Neighbor Resizing"""
        h, w = img.shape[:2]
        # Generate grid indices
        y_indices = (cp.arange(new_h, dtype=cp.int32) * h // new_h).reshape(-1, 1)
        x_indices = (cp.arange(new_w, dtype=cp.int32) * w // new_w).reshape(1, -1)
        # Broadcast sampling
        resized = img[y_indices, x_indices, :]
        return resized

    def infer(self, frame_input):
        """
        Executes Seg and Pose in parallel streams using execute_async_v3.
        """
        if not self.is_ready or frame_input is None: return None, None
        
        # Input Handling (Zero-Copy if possible)
        if hasattr(frame_input, 'device'):
            img_gpu_src = frame_input
            h_orig, w_orig = frame_input.shape[:2]
        else:
            img_gpu_src = cp.asarray(frame_input)
            h_orig, w_orig = frame_input.shape[:2]

        # ----------------------------------------------------
        # STREAM A: Segmentation (SegFormer)
        # ----------------------------------------------------
        with self.stream_seg:
            # 1. Preprocess (Resize -> RGB -> Norm)
            img_seg = self._resize_gpu_nearest(img_gpu_src, self.seg_h, self.seg_w)
            img_pre_seg = img_seg[..., ::-1].transpose(2,0,1).astype(cp.float32) / 255.0
            img_pre_seg = (img_pre_seg.reshape(1, 3, self.seg_h, self.seg_w) - self.mean) / self.std
            
            # 2. Copy to Engine Buffer (Async)
            cp.copyto(self.d_input_seg, img_pre_seg)
            
            # 3. Execute Engine (Async v3)
            # Address is already set in _allocate_buffers
            self.ctx_seg.execute_async_v3(stream_handle=self.stream_seg.ptr)
            
            # 4. Postprocess Part 1 (Sigmoid & Threshold)
            mask_prob = 1.0 / (1.0 + cp.exp(-self.d_seg))
            mask_small = (mask_prob > 0.5).astype(cp.float32).squeeze()
            
            # 5. Restore Resolution
            mask_small_3d = mask_small[..., None]
            mask_final_gpu = self._resize_gpu_nearest(mask_small_3d, h_orig, w_orig).squeeze()

        # ----------------------------------------------------
        # STREAM B: Pose Estimation (SegFormer Pose Head)
        # ----------------------------------------------------
        with self.stream_pose:
            # 1. Preprocess
            img_pose = self._resize_gpu_nearest(img_gpu_src, self.pose_h, self.pose_w)
            img_pre_pose = img_pose[..., ::-1].transpose(2,0,1).astype(cp.float32) / 255.0
            img_pre_pose = (img_pre_pose.reshape(1, 3, self.pose_h, self.pose_w) - self.mean) / self.std
            
            # 2. Copy & Execute (Async v3)
            cp.copyto(self.d_input_pose, img_pre_pose)
            self.ctx_pose.execute_async_v3(stream_handle=self.stream_pose.ptr)

        # ----------------------------------------------------
        # SYNCHRONIZE
        # ----------------------------------------------------
        # Wait for both streams to complete
        self.stream_seg.synchronize()
        self.stream_pose.synchronize()
        
        # ----------------------------------------------------
        # POST-PROCESS POSE (CPU)
        # ----------------------------------------------------
        # Implicit sync happens via asnumpy
        heatmaps = cp.asnumpy(self.d_pose.squeeze())
        keypoints = self._parse_heatmaps(heatmaps, (w_orig, h_orig))
        
        return mask_final_gpu, keypoints

    def _parse_heatmaps(self, heatmaps, original_size):
        """
        Extract coordinates from heatmaps.
        heatmaps: (17, H_map, W_map)
        """
        kpts = []
        w_orig, h_orig = original_size
        _, h_map, w_map = heatmaps.shape 
        
        scale_x = w_orig / w_map
        scale_y = h_orig / h_map
        
        for i in range(17):
            hm = heatmaps[i]
            # Find global maximum
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hm)
            
            if max_val > 0.05: # Confidence Threshold
                x = max_loc[0] * scale_x
                y = max_loc[1] * scale_y
                kpts.append([x, y, max_val])
            else:
                kpts.append([0, 0, 0.0])
                
        return np.array(kpts)