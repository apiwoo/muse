# Project MUSE - beauty_engine.py
# Optimized V10.0: Native CUDA Remap & Fused Kernels
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import time
from ai.tracking.facemesh import FaceMesh

# [GPU Acceleration Setup]
try:
    import cupy as cp
    import cupyx.scipy.ndimage
    HAS_CUDA = True
    print("✅ [BeautyEngine] GPU Acceleration Enabled (CuPy + Native CUDA)")
except ImportError:
    HAS_CUDA = False
    print("⚠️ [BeautyEngine] CuPy not found. Fallback to CPU Mode.")

# ==============================================================================
# [CUDA Kernel 1] Fused Warp Vector Generator (FP32)
# 벡터장 생성 커널
# ==============================================================================
WARP_KERNEL_CODE = r'''
extern "C" __global__
void warp_kernel(
    float* dx, float* dy,       // Output buffers (H, W) - float32
    const float* params,        // [cx, cy, r, strength, vx, vy, mode] * num_points
    int num_points,             // Number of warp points
    int width, int height       // Grid dimensions
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // Accumulator
    float acc_dx = 0.0f;
    float acc_dy = 0.0f;

    for (int i = 0; i < num_points; ++i) {
        int base = i * 7;
        float cx = params[base + 0];
        float cy = params[base + 1];
        float r = params[base + 2];
        float s = params[base + 3];
        float vx = params[base + 4];
        float vy = params[base + 5];
        int mode = (int)params[base + 6];

        float diff_x = x - cx;
        float diff_y = y - cy;
        float dist_sq = diff_x * diff_x + diff_y * diff_y;
        
        if (dist_sq < (r * r)) {
            float dist = sqrtf(dist_sq);
            float factor = (1.0f - dist / r);
            factor = factor * factor * s;

            if (mode == 0) { // Expand (확대)
                acc_dx -= diff_x * factor;
                acc_dy -= diff_y * factor;
            } 
            else if (mode == 1) { // Shrink with Vector (방향성 이동/수축)
                // vx, vy 방향으로 픽셀을 당겨옴
                acc_dx -= vx * factor * r * 0.5f;
                acc_dy -= vy * factor * r * 0.5f;
            }
            else if (mode == 2) { // Shrink to Center (중심 수축)
                acc_dx += diff_x * factor;
                acc_dy += diff_y * factor;
            }
        }
    }

    dx[idx] += acc_dx;
    dy[idx] += acc_dy;
}
'''

# ==============================================================================
# [CUDA Kernel 2] Native Remap & Upscaling
# ==============================================================================
REMAP_KERNEL_CODE = r'''
extern "C" __global__
void remap_kernel(
    const unsigned char* src,  // Source Image (H x W x 3)
    unsigned char* dst,        // Dest Image (H x W x 3)
    const float* dx_small,     // Warp Vector X (h x w)
    const float* dy_small,     // Warp Vector Y (h x w)
    int width, int height,     // Full Resolution
    int small_width, int small_height, // Small Resolution
    int scale                  // Upscale Factor (e.g., 4)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int sx = x / scale;
    int sy = y / scale;

    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;

    int s_idx = sy * small_width + sx;

    float shift_x = dx_small[s_idx] * (float)scale;
    float shift_y = dy_small[s_idx] * (float)scale;

    float src_x = (float)x + shift_x;
    float src_y = (float)y + shift_y;

    int x1 = (int)floorf(src_x);
    int y1 = (int)floorf(src_y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float wx2 = src_x - (float)x1;
    float wx1 = 1.0f - wx2;
    float wy2 = src_y - (float)y1;
    float wy1 = 1.0f - wy2;

    x1 = max(0, min(x1, width - 1));
    y1 = max(0, min(y1, height - 1));
    x2 = max(0, min(x2, width - 1));
    y2 = max(0, min(y2, height - 1));

    int stride = width * 3;
    int out_idx = y * stride + x * 3;

    for (int c = 0; c < 3; ++c) {
        float val = 
            (float)src[y1 * stride + x1 * 3 + c] * wx1 * wy1 +
            (float)src[y1 * stride + x2 * 3 + c] * wx2 * wy1 +
            (float)src[y2 * stride + x1 * 3 + c] * wx1 * wy2 +
            (float)src[y2 * stride + x2 * 3 + c] * wx2 * wy2;
        
        dst[out_idx + c] = (unsigned char)val;
    }
}
'''

class BeautyEngine:
    def __init__(self):
        """
        [Mode A] Real-time Beauty Engine (GPU Edition V10.0)
        - V10.0: Full CUDA Remapping
        - V4.0: Body Quartet Reshape (Shoulder, Ribcage, Waist, Hip)
        """
        print("💄 [BeautyEngine] GPU 엔진 V10.0 (Quartet Update)")
        
        self.map_scale = 0.25 
        
        self.cache_w = 0
        self.cache_h = 0
        self.gpu_initialized = False
        
        self.gpu_dx = None
        self.gpu_dy = None
        
        self.prev_gpu_dx = None
        self.prev_gpu_dy = None
        self.prev_face_center = None

        self.warp_params = [] 
        
        if HAS_CUDA:
            self.warp_kernel = cp.RawKernel(WARP_KERNEL_CODE, 'warp_kernel')
            self.remap_kernel = cp.RawKernel(REMAP_KERNEL_CODE, 'remap_kernel')

    def process(self, frame, faces, body_landmarks=None, params=None):
        if frame is None: return frame
        if params is None: params = {}

        is_gpu_input = HAS_CUDA and hasattr(frame, 'device')
        
        if is_gpu_input:
            h, w = frame.shape[:2]
            frame_gpu = frame 
        else:
            h, w = frame.shape[:2]
            if HAS_CUDA:
                frame_gpu = cp.asarray(frame)
            else:
                return frame 

        if self.cache_w != w or self.cache_h != h:
            self.cache_w, self.cache_h = w, h
            self.gpu_initialized = False
            self.prev_gpu_dx = None 
            self.prev_gpu_dy = None
            self.prev_face_center = None
            print(f"⚡ [BeautyEngine] Grid Cache Rebuilt: {w}x{h}")

        sw, sh = int(w * self.map_scale), int(h * self.map_scale)
        
        if not self.gpu_initialized:
            self.gpu_dx = cp.zeros((sh, sw), dtype=cp.float32)
            self.gpu_dy = cp.zeros((sh, sw), dtype=cp.float32)
            self.gpu_initialized = True

        self.warp_params.clear() 
        has_deformation = False
        
        # --- Body Reshaping (Quartet) ---
        if body_landmarks is not None:
            if hasattr(body_landmarks, 'get'): 
                body_cpu = body_landmarks.get()
            else:
                body_cpu = body_landmarks
            
            scaled_body = body_cpu[:, :2] * self.map_scale
            
            # 1. Shoulder Narrow (어깨)
            shoulder_strength = params.get('shoulder_narrow', 0)
            if shoulder_strength > 0:
                self._collect_shoulder_params(scaled_body, shoulder_strength)
                has_deformation = True

            # 2. Ribcage Slim (흉통) - New
            ribcage_strength = params.get('ribcage_slim', 0)
            if ribcage_strength > 0:
                self._collect_ribcage_params(scaled_body, ribcage_strength)
                has_deformation = True

            # 3. Waist Slim (허리)
            waist_strength = params.get('waist_slim', 0)
            if waist_strength > 0:
                self._collect_waist_params(scaled_body, waist_strength)
                has_deformation = True

            # 4. Hip Widen (골반)
            hip_strength = params.get('hip_widen', 0)
            if hip_strength > 0:
                self._collect_hip_params(scaled_body, hip_strength)
                has_deformation = True

        # --- Face Reshaping ---
        target_alpha = 0.8
        
        if faces:
            face_v = params.get('face_v', 0)
            eye_scale = params.get('eye_scale', 0)
            head_scale = params.get('head_scale', 0) 

            current_face_center = np.mean(faces[0].landmarks, axis=0)
            if self.prev_face_center is not None:
                velocity = np.linalg.norm(current_face_center - self.prev_face_center)
                if velocity > 3.0: target_alpha = 0.0
                elif velocity > 1.0: target_alpha = 0.3
                else: target_alpha = 0.85
            self.prev_face_center = current_face_center

            for face in faces:
                lm_small = face.landmarks * self.map_scale
                
                if face_v > 0:
                    self._collect_face_contour_params(lm_small, face_v)
                if eye_scale > 0:
                    self._collect_eyes_params(lm_small, eye_scale)
                if head_scale != 0: 
                    self._collect_head_params(lm_small, head_scale)

            if face_v > 0 or eye_scale > 0 or head_scale != 0:
                has_deformation = True
        else:
            self.prev_face_center = None

        self.gpu_dx.fill(0)
        self.gpu_dy.fill(0)

        if has_deformation and self.warp_params:
            params_arr = np.array(self.warp_params, dtype=np.float32)
            params_gpu = cp.asarray(params_arr)
            
            num_points = len(self.warp_params)
            
            block_dim = (16, 16)
            grid_dim = ((sw + block_dim[0] - 1) // block_dim[0], 
                        (sh + block_dim[1] - 1) // block_dim[1])
            
            self.warp_kernel(
                grid_dim, block_dim, 
                (self.gpu_dx, self.gpu_dy, params_gpu, num_points, sw, sh)
            )

        if has_deformation or (self.prev_gpu_dx is not None):
            self._apply_temporal_smoothing_fast(target_alpha)
            
            cupyx.scipy.ndimage.gaussian_filter(self.gpu_dx, sigma=5, output=self.gpu_dx)
            cupyx.scipy.ndimage.gaussian_filter(self.gpu_dy, sigma=5, output=self.gpu_dy)
            
            result_gpu = cp.empty_like(frame_gpu)
            
            block_dim = (32, 32)
            grid_dim = ((w + block_dim[0] - 1) // block_dim[0], 
                        (h + block_dim[1] - 1) // block_dim[1])
            
            scale = int(1.0 / self.map_scale) 
            
            self.remap_kernel(
                grid_dim, block_dim,
                (frame_gpu, result_gpu, self.gpu_dx, self.gpu_dy, 
                 w, h, sw, sh, scale)
            )
            
            if is_gpu_input:
                return result_gpu
            else:
                return result_gpu.get()
        else:
            return frame_gpu if is_gpu_input else frame

    # ==========================================================
    # [Body Reshaping Quartet]
    # ==========================================================
    def _collect_shoulder_params(self, keypoints, strength):
        """1. 어깨 줄이기 (안쪽으로 당김)"""
        l_sh, r_sh = keypoints[5], keypoints[6]
        
        shoulder_width = np.linalg.norm(l_sh - r_sh)
        if shoulder_width < 3: return
        center_shoulder = (l_sh + r_sh) / 2
        
        radius = int(shoulder_width * 0.6)
        s = strength * 0.3
        
        vec_l = center_shoulder - l_sh
        vec_r = center_shoulder - r_sh
        
        self._add_param(l_sh, radius, s, mode=1, vector=vec_l)
        self._add_param(r_sh, radius, s, mode=1, vector=vec_r)

    def _collect_ribcage_params(self, keypoints, strength):
        """2. 흉통 줄이기 (갈비뼈 부근 안쪽으로 당김)"""
        l_sh, r_sh = keypoints[5], keypoints[6]
        l_hip, r_hip = keypoints[11], keypoints[12]
        
        # 흉통 위치 추정 (어깨와 골반 사이 상단 35% 지점)
        l_rib = l_sh * 0.65 + l_hip * 0.35
        r_rib = r_sh * 0.65 + r_hip * 0.35
        center_rib = (l_rib + r_rib) / 2
        
        rib_width = np.linalg.norm(l_rib - r_rib)
        if rib_width < 3: return
        
        radius = int(rib_width * 0.7)
        s = strength * 0.4
        
        vec_l = center_rib - l_rib
        vec_r = center_rib - r_rib
        
        self._add_param(l_rib, radius, s, mode=1, vector=vec_l)
        self._add_param(r_rib, radius, s, mode=1, vector=vec_r)

    def _collect_waist_params(self, keypoints, strength):
        """3. 허리 줄이기 (안쪽으로 당김)"""
        l_sh, r_sh = keypoints[5], keypoints[6]
        l_hip, r_hip = keypoints[11], keypoints[12]
        
        # 허리 위치 (상단 60% 지점)
        l_waist = l_sh * 0.4 + l_hip * 0.6
        r_waist = r_sh * 0.4 + r_hip * 0.6
        center_waist = (l_waist + r_waist) / 2
        
        body_width = np.linalg.norm(l_waist - r_waist)
        if body_width < 3: return 
        
        radius = int(body_width * 0.6)
        s = strength * 0.4
        
        vec_l = center_waist - l_waist
        vec_r = center_waist - r_waist
        
        self._add_param(l_waist, radius, s, mode=1, vector=vec_l)
        self._add_param(r_waist, radius, s, mode=1, vector=vec_r)

    def _collect_hip_params(self, keypoints, strength):
        """4. 골반 늘리기 (바깥쪽으로 밀어냄)"""
        l_hip, r_hip = keypoints[11], keypoints[12]
        
        hip_width = np.linalg.norm(l_hip - r_hip)
        if hip_width < 3: return
        center_hip = (l_hip + r_hip) / 2
        
        radius = int(hip_width * 0.7)
        s = strength * 0.3
        
        # 바깥 방향 벡터
        vec_l_out = l_hip - center_hip 
        vec_r_out = r_hip - center_hip 
        
        self._add_param(l_hip, radius, s, mode=1, vector=vec_l_out)
        self._add_param(r_hip, radius, s, mode=1, vector=vec_r_out)

    # ==========================================================
    # [Face Reshaping]
    # ==========================================================
    def _collect_head_params(self, lm, strength):
        chin = lm[152]
        forehead = lm[10]
        head_height = np.linalg.norm(chin - forehead)
        
        up_vec = forehead - chin
        up_vec /= (np.linalg.norm(up_vec) + 1e-6)
        center = np.mean(lm, axis=0) + up_vec * (head_height * 0.5)
        
        radius = int(head_height * 1.6)
        
        dist_to_top = center[1]
        safe_factor = 1.0
        if dist_to_top < radius:
            safe_factor = max(0.2, dist_to_top / radius)
        
        if strength > 0:
            s = strength * 0.5 * safe_factor
            self._add_param(center, radius, s, mode=2)
        else:
            s = abs(strength) * 0.5 * safe_factor
            self._add_param(center, radius, s, mode=0)

    def _collect_eyes_params(self, lm, strength):
        indices_l = FaceMesh.FACE_INDICES['EYE_L']
        indices_r = FaceMesh.FACE_INDICES['EYE_R']
        pts_l = lm[indices_l]
        center_l = np.mean(pts_l, axis=0)
        width_l = np.linalg.norm(pts_l[0] - pts_l[8])
        radius_l = int(width_l * 1.5)
        pts_r = lm[indices_r]
        center_r = np.mean(pts_r, axis=0)
        width_r = np.linalg.norm(pts_r[0] - pts_r[8])
        radius_r = int(width_r * 1.5)
        self._add_param(center_l, radius_l, strength, mode=0)
        self._add_param(center_r, radius_r, strength, mode=0)

    def _collect_face_contour_params(self, lm, strength):
        target_pt = lm[FaceMesh.FACE_INDICES['NOSE_TIP'][0]]
        for idx in FaceMesh.FACE_INDICES['JAW_L']:
            pt = lm[idx]
            radius = int(np.linalg.norm(pt - target_pt) * 0.6)
            vector = target_pt - pt
            self._add_param(pt, radius, strength * 0.3, mode=1, vector=vector)
        for idx in FaceMesh.FACE_INDICES['JAW_R']:
            pt = lm[idx]
            radius = int(np.linalg.norm(pt - target_pt) * 0.6)
            vector = target_pt - pt
            self._add_param(pt, radius, strength * 0.3, mode=1, vector=vector)

    def _add_param(self, center, radius, strength, mode=0, vector=None):
        if radius <= 0: return
        vx, vy = 0.0, 0.0
        if vector is not None:
            v_len = np.linalg.norm(vector) + 1e-6
            vx, vy = vector[0]/v_len, vector[1]/v_len
            
        self.warp_params.append([
            center[0], center[1], float(radius), float(strength),
            vx, vy, float(mode)
        ])

    def _apply_temporal_smoothing_fast(self, alpha):
        if self.prev_gpu_dx is None:
            self.prev_gpu_dx = self.gpu_dx.copy()
            self.prev_gpu_dy = self.gpu_dy.copy()
            return

        if alpha == 0.0:
            self.prev_gpu_dx[:] = self.gpu_dx
            self.prev_gpu_dy[:] = self.gpu_dy
            return

        beta = 1.0 - alpha
        self.gpu_dx *= beta
        self.gpu_dx += self.prev_gpu_dx * alpha
        self.gpu_dy *= beta
        self.gpu_dy += self.prev_gpu_dy * alpha
        
        self.prev_gpu_dx[:] = self.gpu_dx
        self.prev_gpu_dy[:] = self.gpu_dy