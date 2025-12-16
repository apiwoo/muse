# Project MUSE - consensus_engine.py
# V8.5 Hybrid: Fully Parallelized Architecture (Threaded Default Mode)
# Optimized: Runs MODNet & ViTPose in parallel threads for Default Mode
# Compatibility: TensorRT 10.x (execute_async_v3)
# (C) 2025 MUSE Corp. All rights reserved.

import tensorrt as trt
import cupy as cp
import numpy as np
import cv2
import os
import sys
from concurrent.futures import ThreadPoolExecutor

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

# Import Student (Personalized)
try:
    from ai.distillation.student.inference_trt import DualInferenceTRT
except ImportError:
    DualInferenceTRT = None

class TensorRTModel:
    """Generic TensorRT Wrapper for Segmentation Models (MODNet)"""
    def __init__(self, engine_path, input_shape=(1, 3, 544, 960), lazy=False):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine_path = engine_path
        self.input_shape = input_shape # NCHW
        self.is_ready = False
        self.input_name = None
        
        # Lazy Loading: Init variables but don't load engine yet
        self.context = None
        self.d_input = None
        self.d_output = None
        
        if not lazy:
            self.load()

    def load(self):
        if self.is_ready: return
        
        if os.path.exists(self.engine_path):
            try:
                print(f"[AI] Loading TensorRT Engine: {os.path.basename(self.engine_path)}...")
                with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                    self.engine = runtime.deserialize_cuda_engine(f.read())
                
                self.context = self.engine.create_execution_context()
                
                # Input/Output Names
                self.input_name = self.engine.get_tensor_name(0)
                output_name = self.engine.get_tensor_name(1)

                # Allocate Buffers
                self.d_input = cp.zeros(self.input_shape, dtype=cp.float32)
                self.d_output = cp.zeros((1, 1, self.input_shape[2], self.input_shape[3]), dtype=cp.float32)
                
                # [TRT 10 Fix] Set addresses
                self.context.set_tensor_address(self.input_name, int(self.d_input.data.ptr))
                self.context.set_tensor_address(output_name, int(self.d_output.data.ptr))
                
                self.is_ready = True
                print(f"[AI] [OK] Engine Loaded: {os.path.basename(self.engine_path)}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load {self.engine_path}: {e}")
        else:
            print(f"[WARNING] Engine not found: {self.engine_path}")

    def infer(self, img_gpu_norm):
        if not self.is_ready: return None
        
        if self.input_name:
            self.context.set_input_shape(self.input_name, img_gpu_norm.shape)

        cp.copyto(self.d_input, img_gpu_norm)
        
        # [TRT 10 Fix] execute_async_v3 with default stream (0)
        self.context.execute_async_v3(stream_handle=0)
        
        return self.d_output

class ConsensusEngine:
    def __init__(self, root_dir):
        """
        [Hybrid Mode V8.5]
        - Default: Parallelized MODNet + ViTPose (using ThreadPool)
        - Personal: DualInferenceTRT (using CUDA Streams)
        """
        self.root_dir = root_dir
        self.model_dir = os.path.join(root_dir, "assets", "models")
        self.personal_dir = os.path.join(self.model_dir, "personal")
        
        print("[AI] Initializing Hybrid Consensus Engine (Parallel Ready)...")

        # 1. Init Default Models Placeholders (Do NOT load yet)
        self.pose_model = None
        self.modnet = TensorRTModel(
            os.path.join(self.model_dir, "segmentation", "modnet_544p.engine"),
            input_shape=(1, 3, 544, 960),
            lazy=True # Wait until needed
        )

        # 2. Personal Models State
        self.student_models = {} # Cache loaded students
        self.current_student = None
        self.use_personal = False
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Preprocessing Constants
        self.mean = cp.array([0.5, 0.5, 0.5], dtype=cp.float32).reshape(1,3,1,1)
        self.std = cp.array([0.5, 0.5, 0.5], dtype=cp.float32).reshape(1,3,1,1)

    def _ensure_default_models_loaded(self):
        """Load default models only if they aren't loaded yet."""
        if self.pose_model is None:
            print("[AI] [Fallback] Loading ViTPose (Default Strategy)...")
            try:
                # [Modified] Default to Base engine
                pose_path = os.path.join(self.model_dir, "tracking", "vitpose_base.engine")
                
                # Check Base -> Fallback Huge
                if not os.path.exists(pose_path):
                    pose_path = os.path.join(self.model_dir, "tracking", "vitpose_huge.engine")
                
                if os.path.exists(pose_path) and VitPoseTrt:
                    self.pose_model = VitPoseTrt(pose_path)
                else:
                    print("[WARNING] ViTPose engine not found (Checked Base/Huge).")
            except Exception as e:
                print(f"[WARNING] ViTPose Init Failed: {e}")
        
        if not self.modnet.is_ready:
            print("[AI] [Fallback] Loading MODNet (Default Strategy)...")
            self.modnet.load()

    def set_profile(self, profile_name):
        """
        Switch AI Strategy based on Profile.
        Checks for 'student_seg_{profile}.engine' and 'student_pose_{profile}.engine'.
        """
        seg_path = os.path.join(self.personal_dir, f"student_seg_{profile_name}.engine")
        pose_path = os.path.join(self.personal_dir, f"student_pose_{profile_name}.engine")
        
        # Check if personal model exists
        if os.path.exists(seg_path) and os.path.exists(pose_path) and DualInferenceTRT:
            # Load if not in cache
            if profile_name not in self.student_models:
                print(f"[AI] Loading Student Engine into VRAM...")
                try:
                    self.student_models[profile_name] = DualInferenceTRT(seg_path, pose_path)
                except Exception as e:
                    print(f"[ERROR] Student Load Failed: {e}")
            
            # Activate
            student = self.student_models.get(profile_name)
            if student and student.is_ready:
                self.current_student = student
                self.use_personal = True
                print(f"[AI] >>> Strategy Switched: PERSONALIZED MODEL ({profile_name})")
                return

        # Fallback to Default
        self._ensure_default_models_loaded() # Load heavy models NOW
        self.use_personal = False
        self.current_student = None
        print(f"[AI] >>> Strategy Switched: DEFAULT MODEL (MODNet + ViTPose)")

    def _run_modnet(self, frame_gpu):
        """Worker function for MODNet"""
        h, w = frame_gpu.shape[:2]
        target_h, target_w = 544, 960
        zoom_h = target_h / h
        zoom_w = target_w / w

        # GPU Resize (MODNet Input)
        if HAS_CUPYX:
            frame_small = cupyx.scipy.ndimage.zoom(frame_gpu, (zoom_h, zoom_w, 1), order=1)
        else:
            frame_cpu = cp.asnumpy(frame_gpu)
            frame_small_cpu = cv2.resize(frame_cpu, (target_w, target_h))
            frame_small = cp.asarray(frame_small_cpu)
        
        # Normalize
        img_norm = frame_small.astype(cp.float32) / 255.0
        img_norm = img_norm.transpose(2, 0, 1).reshape(1, 3, target_h, target_w)
        img_norm = (img_norm - self.mean) / self.std 
        
        return self.modnet.infer(img_norm)

    def _run_vitpose(self, frame_gpu):
        """Worker function for ViTPose (Handles GPU->CPU copy internally)"""
        if self.pose_model is None: return None
        try:
            # Transfer GPU -> CPU happens inside this thread.
            if hasattr(frame_gpu, 'get'):
                frame_cpu = frame_gpu.get()
            else:
                frame_cpu = frame_gpu
            return self.pose_model.inference(frame_cpu)
        except Exception: 
            return None

    def process(self, frame_gpu):
        """
        Main Process Loop.
        Returns: (matte_1080_gpu, kpts_cpu)
        """
        if frame_gpu is None: return None, None
        
        # ==========================================================
        # STRATEGY A: Personal Model (Already Parallelized via Streams)
        # ==========================================================
        if self.use_personal and self.current_student:
            mask_gpu, kpts = self.current_student.infer(frame_gpu)
            
            if mask_gpu is None:
                h, w = frame_gpu.shape[:2]
                mask_gpu = cp.zeros((h, w), dtype=cp.float32)
                
            return mask_gpu, kpts

        # ==========================================================
        # STRATEGY B: Default (Parallelized via Threads)
        # ==========================================================
        # Ensure default models are ready
        if not self.modnet.is_ready:
            self._ensure_default_models_loaded()

        h, w = frame_gpu.shape[:2]
        
        # [PARALLEL EXECUTION]
        future_seg = self.executor.submit(self._run_modnet, frame_gpu)
        future_pose = self.executor.submit(self._run_vitpose, frame_gpu)
        
        raw_matte = future_seg.result()
        kpts = future_pose.result()
        
        if raw_matte is None:
            return cp.zeros((h, w), dtype=cp.float32), kpts

        # MODNet Output: (1, 1, 544, 960)
        matte_small = raw_matte[0, 0]
        
        # Upscale Matte to 1080p
        target_h, target_w = 544, 960
        if HAS_CUPYX:
            zoom_h_inv = h / target_h
            zoom_w_inv = w / target_w
            matte_1080 = cupyx.scipy.ndimage.zoom(matte_small, (zoom_h_inv, zoom_w_inv), order=1)
            
            mh, mw = matte_1080.shape
            if mh != h or mw != w:
                matte_1080 = matte_1080[:h, :w]
        else:
            matte_small_cpu = cp.asnumpy(matte_small)
            matte_1080_cpu = cv2.resize(matte_small_cpu, (w, h))
            matte_1080 = cp.asarray(matte_1080_cpu)

        return matte_1080, kpts