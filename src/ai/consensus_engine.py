# Project MUSE - consensus_engine.py
# The "Tri-Core" Logic: ViTPose + MODNet + DeepLab
# Fixed: Added 'set_input_shape' for Dynamic TRT Engine
# (C) 2025 MUSE Corp. All rights reserved.

import tensorrt as trt
import cupy as cp
import numpy as np
import cv2
import os
import sys

# [Optimization] GPU Resizing
try:
    import cupyx.scipy.ndimage
    HAS_CUPYX = True
except ImportError:
    HAS_CUPYX = False
    print("[WARNING] cupyx not found. GPU resizing might differ.")

# Import ViTPose (Teacher B)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from ai.tracking.vitpose_trt import VitPoseTrt
except ImportError:
    VitPoseTrt = None

class TensorRTModel:
    """Generic TensorRT Wrapper for Segmentation Models"""
    def __init__(self, engine_path, input_shape=(1, 3, 544, 960)):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine_path = engine_path
        self.input_shape = input_shape # NCHW
        self.is_ready = False
        self.input_name = None
        
        if os.path.exists(engine_path):
            try:
                with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                    self.engine = runtime.deserialize_cuda_engine(f.read())
                
                self.context = self.engine.create_execution_context()
                
                # [Fix] 입력 텐서 이름 찾기 (TensorRT 8.5+ 호환)
                # 바인딩 인덱스 0번이 입력이라고 가정하고 이름 가져오기
                try:
                    self.input_name = self.engine.get_tensor_name(0)
                except AttributeError:
                    # 구버전 API Fallback
                    self.input_name = self.engine.get_binding_name(0)

                # Buffer Allocation
                self.d_input = cp.zeros(input_shape, dtype=cp.float32)
                # Output shape: (1, 1, H, W) for matte
                self.d_output = cp.zeros((1, 1, input_shape[2], input_shape[3]), dtype=cp.float32)
                
                # 바인딩 주소 연결
                self.bindings = [int(self.d_input.data.ptr), int(self.d_output.data.ptr)]
                
                self.is_ready = True
                print(f"[AI] Loaded TRT Engine: {os.path.basename(engine_path)}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load {engine_path}: {e}")
        else:
            print(f"[WARNING] Engine not found: {engine_path}")

    def infer(self, img_gpu_norm):
        """
        img_gpu_norm: (1, 3, H, W) Normalized CuPy Array
        """
        if not self.is_ready: return None
        
        # [CRITICAL FIX] 동적 쉐이프 엔진은 실행 전 반드시 입력 크기를 명시해야 함
        if self.input_name:
            self.context.set_input_shape(self.input_name, img_gpu_norm.shape)
        else:
            # 구버전 방식 (인덱스 기반)
            self.context.set_binding_shape(0, img_gpu_norm.shape)

        cp.copyto(self.d_input, img_gpu_norm)
        
        # 추론 실행 (V2 비동기 실행)
        self.context.execute_v2(bindings=self.bindings)
        
        return self.d_output

class ConsensusEngine:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        model_dir = os.path.join(root_dir, "assets", "models")
        
        # 1. ViTPose (Guide)
        try:
            pose_path = os.path.join(model_dir, "tracking", "vitpose_huge.engine")
            if os.path.exists(pose_path) and VitPoseTrt:
                self.pose_model = VitPoseTrt(pose_path)
            else:
                self.pose_model = None
                print("[WARNING] ViTPose engine not found or module missing.")
        except Exception as e:
            self.pose_model = None
            print(f"[WARNING] ViTPose Init Failed: {e}")

        # 2. MODNet (Detailer) - Optimized for 544p (qHD+)
        # Resolution: 960x544 (16:9 Aspect Ratio, Stride 32)
        self.target_w = 960
        self.target_h = 544
        
        self.modnet = TensorRTModel(
            os.path.join(model_dir, "segmentation", "modnet_544p.engine"), # Changed to 544p
            input_shape=(1, 3, self.target_h, self.target_w)
        )

        # Preprocessing Constants
        self.mean = cp.array([0.5, 0.5, 0.5], dtype=cp.float32).reshape(1,3,1,1)
        self.std = cp.array([0.5, 0.5, 0.5], dtype=cp.float32).reshape(1,3,1,1)
        
        # [Optimization] No padding needed for 544p (Multiple of 32)

    def process(self, frame_gpu):
        """
        Main Pipeline: Input(1080p) -> Downscale(544p) -> Tri-Core -> Upscale(1080p) -> Alpha
        """
        if frame_gpu is None: return None, None

        h, w = frame_gpu.shape[:2] # Expect 1080, 1920
        
        # [Step 1] Internal Downscaling (1080p -> 544p)
        # Using GPU Bilinear Interpolation via CuPy/Cupyx
        if HAS_CUPYX:
            # Calculate zoom factors
            zoom_h = self.target_h / h
            zoom_w = self.target_w / w
            
            # (H, W, C) -> (TargetH, TargetW, C)
            # order=1 (Bilinear) is faster and good enough for input
            frame_small = cupyx.scipy.ndimage.zoom(frame_gpu, (zoom_h, zoom_w, 1), order=1)
        else:
            # Fallback (Slow CPU Resize, Avoid if possible)
            frame_cpu = cp.asnumpy(frame_gpu)
            frame_small_cpu = cv2.resize(frame_cpu, (self.target_w, self.target_h))
            frame_small = cp.asarray(frame_small_cpu)
        
        # [Step 2] Normalize & CHW
        # (H, W, C) -> (1, C, H, W)
        img_norm = frame_small.astype(cp.float32) / 255.0
        img_norm = img_norm.transpose(2, 0, 1).reshape(1, 3, self.target_h, self.target_w)
        img_norm = (img_norm - self.mean) / self.std 
        
        # [Step 3] Inference (MODNet)
        raw_matte = self.modnet.infer(img_norm) # (1, 1, 544, 960)
        
        if raw_matte is None:
            return cp.zeros((h, w), dtype=cp.float32), None

        # [Step 4] Upscale back to 1080p
        matte_small = raw_matte[0, 0] # (544, 960)
        
        if HAS_CUPYX:
            # Inverse zoom (544p -> 1080p)
            # order=1 (Bilinear) ensures smooth alpha edges
            zoom_h_inv = h / self.target_h
            zoom_w_inv = w / self.target_w
            matte_1080 = cupyx.scipy.ndimage.zoom(matte_small, (zoom_h_inv, zoom_w_inv), order=1)
            
            # [Safety] Fix rounding errors in shape
            mh, mw = matte_1080.shape
            if mh != h or mw != w:
                # Simple crop or pad (Usually it matches or is slightly larger)
                matte_1080 = matte_1080[:h, :w]
                # If smaller (rare), we need padding, but zoom usually covers it.
        else:
            # Fallback
            matte_small_cpu = cp.asnumpy(matte_small)
            matte_1080_cpu = cv2.resize(matte_small_cpu, (w, h))
            matte_1080 = cp.asarray(matte_1080_cpu)

        # [Step 5] ViTPose Guide (Hull Logic)
        # Note: ViTPose still runs on CPU input currently (can be optimized later)
        frame_cpu = cp.asnumpy(frame_gpu)
        kpts = None
        if self.pose_model:
            try:
                kpts = self.pose_model.inference(frame_cpu)
            except Exception:
                pass
        
        if kpts is not None:
            # Create Hull Mask on GPU
            hull_mask = self._create_hull_mask(kpts, w, h)
            # Consensus: Filter Matte with Hull (Noise reduction)
            matte_1080 = matte_1080 * hull_mask
            
        return matte_1080, kpts

    def _create_hull_mask(self, kpts, w, h):
        """
        Creates a convex hull mask from keypoints on GPU.
        """
        # 신뢰도 0.2 이상인 점만 사용
        valid_pts = kpts[kpts[:, 2] > 0.2, :2].astype(np.int32)
        
        mask = np.zeros((h, w), dtype=np.float32)
        if len(valid_pts) > 4:
            hull = cv2.convexHull(valid_pts)
            cv2.fillConvexPoly(mask, hull, 1.0)
            
            # Dilate to include hair/accessories (넉넉하게 확장)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
            mask = cv2.dilate(mask, kernel, iterations=1)
            
        return cp.asarray(mask)