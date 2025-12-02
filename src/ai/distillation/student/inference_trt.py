# Project MUSE - inference_trt.py
# The "Student" Inference Engine: TensorRT Optimized (High Performance)
# (C) 2025 MUSE Corp. All rights reserved.

import tensorrt as trt
import cv2
import numpy as np
import os
import sys

# High-Performance GPU Memory Handling
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("⚠️ [StudentTRT] CuPy not found. Using slower CPU->GPU transfer.")

class StudentInferenceTRT:
    def __init__(self, engine_path=None):
        """
        [MUSE Student Inference - TensorRT Edition]
        - 역할: .engine 파일을 로드하여 4K 60fps 방어 가능한 초고속 추론 수행
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.is_ready = False
        
        # 기본 경로 설정
        if engine_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            engine_path = os.path.join(base_dir, "assets", "models", "personal", "student_model.engine")
            
        if not os.path.exists(engine_path):
            print(f"⚠️ [StudentTRT] 엔진 파일 없음: {engine_path}")
            print("   -> 'tools/convert_student_to_trt.py'를 실행하여 최적화하세요.")
            return

        print(f"🚀 [StudentTRT] TensorRT 엔진 로딩 중... ({os.path.basename(engine_path)})")
        
        try:
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            if self.engine:
                self.context = self.engine.create_execution_context()
                self._allocate_buffers()
                self.is_ready = True
                print("✅ [StudentTRT] 엔진 초기화 완료")
            else:
                print("❌ [StudentTRT] 엔진 역직렬화 실패")
                
        except Exception as e:
            print(f"❌ [StudentTRT] 초기화 오류: {e}")

        # Preprocessing Constants
        self.input_h, self.input_w = 512, 512
        self.mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32).reshape(1, 3, 1, 1) if HAS_CUPY else np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32).reshape(1, 3, 1, 1) if HAS_CUPY else np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def _allocate_buffers(self):
        """GPU 메모리 버퍼 할당"""
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            
            # shape 처리 (batch size 1 고정)
            vol = 1
            for s in shape:
                vol *= s if s > 0 else 1
            
            # CuPy로 GPU 메모리 할당
            if HAS_CUPY:
                # trt.nptype(dtype) 사용 불가 시 수동 매핑
                np_dtype = np.float32 
                size = vol * np.dtype(np_dtype).itemsize
                
                # CuPy Array 생성
                gpu_mem = cp.zeros(vol, dtype=np_dtype)
                self.bindings.append(int(gpu_mem.data.ptr))
                
                binding_info = {
                    'name': name,
                    'mem': gpu_mem,
                    'shape': shape # [Modification] Shape 정보 저장
                }
            else:
                # Fallback implementation omitted for brevity, assuming CuPy exists as per check_gpu
                raise RuntimeError("CuPy is required for TensorRT inference in MUSE.")

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(binding_info)
            else:
                self.outputs.append(binding_info)
                
            self.context.set_input_shape(name, shape)

    def infer(self, frame_bgr):
        """
        :param frame_bgr: (H, W, 3) Image
        :return: (mask, keypoints)
        """
        if not self.is_ready or frame_bgr is None:
            return None, None

        h_orig, w_orig = frame_bgr.shape[:2]

        # 1. Preprocess (Resize & Normalize)
        # Using OpenCV for resize (CPU) -> Upload to GPU
        img_resized = cv2.resize(frame_bgr, (self.input_w, self.input_h))
        
        if HAS_CUPY:
            # CPU -> GPU Upload & Normalize
            img_gpu = cp.asarray(img_resized) # (512, 512, 3)
            img_gpu = img_gpu[..., ::-1] # BGR -> RGB
            img_gpu = img_gpu.transpose(2, 0, 1).astype(cp.float32) / 255.0 # (3, 512, 512)
            img_gpu = (img_gpu - self.mean.reshape(3, 1, 1)) / self.std.reshape(3, 1, 1)
            img_gpu = img_gpu.ravel() # Flatten
            
            # Copy to Input Buffer
            cp.copyto(self.inputs[0]['mem'], img_gpu)
            
            # 2. Inference
            self.context.execute_v2(bindings=self.bindings)
            
            # 3. Postprocess
            # Output 0: seg_logits (1, 1, 512, 512)
            # Output 1: pose_heatmaps (1, 17, 128, 128)
            
            # 이름으로 구분하여 가져오기
            seg_mem = None
            pose_mem = None
            
            seg_shape = (1, 512, 512) # Default shape
            pose_shape = (17, 128, 128) # Default shape

            for out in self.outputs:
                if 'seg' in out['name']: 
                    seg_mem = out['mem']
                    seg_shape = out['shape']
                elif 'pose' in out['name']: 
                    pose_mem = out['mem']
                    pose_shape = out['shape']
            
            if seg_mem is None: 
                seg_mem = self.outputs[0]['mem']
                seg_shape = self.outputs[0]['shape']
            if pose_mem is None: 
                pose_mem = self.outputs[1]['mem']
                pose_shape = self.outputs[1]['shape']

            # --- Segmentation ---
            # [Modification] 하드코딩 제거: 실제 텐서 크기 기반으로 reshape
            # TensorRT Shape (Batch, Channel, Height, Width) or similar
            # Assuming standard layout, we use the stored shape.
            # remove batch dimension if present (e.g. (1, 1, 512, 512) -> (1, 512, 512))
            
            # Simple fallback reshape logic if tuple has batch dim
            valid_seg_shape = []
            for s in seg_shape:
                if s > 1 or (s == 1 and len(valid_seg_shape) < 3): 
                   valid_seg_shape.append(s)
            
            # Ensure it matches expected logic (CHW)
            if len(valid_seg_shape) == 3:
                seg_logits = seg_mem.reshape(valid_seg_shape)
            else:
                seg_logits = seg_mem.reshape(1, 512, 512) # Fallback

            # Sigmoid using CuPy
            mask_prob = 1.0 / (1.0 + cp.exp(-seg_logits))
            mask_bin = (mask_prob > 0.5).astype(cp.uint8) * 255
            mask_cpu = cp.asnumpy(mask_bin[0])
            
            # Resize back to original
            mask_final = cv2.resize(mask_cpu, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

            # --- Pose ---
            # Remove Batch dim
            valid_pose_shape = []
            for s in pose_shape:
                if s > 1 or (s == 1 and len(valid_pose_shape) < 3):
                    valid_pose_shape.append(s)

            if len(valid_pose_shape) == 3:
                 heatmaps = pose_mem.reshape(valid_pose_shape)
            else:
                 heatmaps = pose_mem.reshape(17, 128, 128) # Fallback

            # Heatmaps are still on GPU, need to find max location
            # For simplicity, download heatmaps to CPU for parsing (Parsing is cheap)
            heatmaps_cpu = cp.asnumpy(heatmaps)
            keypoints = self._parse_heatmaps(heatmaps_cpu, (w_orig, h_orig))

            return mask_final, keypoints
            
        return None, None

    def _parse_heatmaps(self, heatmaps, original_size):
        """히트맵 파싱 (CPU)"""
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