# Project MUSE - beauty_engine.py
# Optimized V12.0: Self-Healing Background (Dynamic Update)
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import os
import glob
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
# ==============================================================================
WARP_KERNEL_CODE = r'''
extern "C" __global__
void warp_kernel(
    float* dx, float* dy,       
    const float* params,        
    int num_points,             
    int width, int height       
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
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

            if (mode == 0) { 
                acc_dx -= diff_x * factor;
                acc_dy -= diff_y * factor;
            } 
            else if (mode == 1) { 
                acc_dx -= vx * factor * r * 0.5f;
                acc_dy -= vy * factor * r * 0.5f;
            }
            else if (mode == 2) { 
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
# [CUDA Kernel 2] Background Updater (Self-Healing)
# 현재 프레임의 배경(Mask=0)을 배경 버퍼에 학습시킴
# ==============================================================================
UPDATE_BG_KERNEL_CODE = r'''
extern "C" __global__
void update_bg_kernel(
    const unsigned char* current_frame, // Live Input (BGR)
    const unsigned char* mask,          // Segmentation Mask (0=BG, 255=Person)
    unsigned char* bg_buffer,           // Persistent Background Buffer (Update Target)
    int width, int height,
    float learning_rate                 // 0.0 ~ 1.0 (How fast to adapt)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int stride = width;
    int pixel_idx = idx * 3;

    // Check Mask (0 is Background)
    // We update only if it is SURELY background.
    if (mask[idx] < 10) { 
        for (int c = 0; c < 3; ++c) {
            float old_bg = (float)bg_buffer[pixel_idx + c];
            float new_bg = (float)current_frame[pixel_idx + c];
            
            // Linear Interpolation (Moving Average)
            // bg = bg * (1 - lr) + new * lr
            float updated = old_bg * (1.0f - learning_rate) + new_bg * learning_rate;
            
            bg_buffer[pixel_idx + c] = (unsigned char)updated;
        }
    }
}
'''

# ==============================================================================
# [CUDA Kernel 3] Remap & Composite (Compositor)
# ==============================================================================
COMPOSITE_KERNEL_CODE = r'''
extern "C" __global__
void composite_kernel(
    const unsigned char* src,  
    const unsigned char* mask, 
    const unsigned char* bg,   
    unsigned char* dst,        
    const float* dx_small,     
    const float* dy_small,     
    int width, int height,     
    int small_width, int small_height, 
    int scale,                 
    int use_bg                 
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
    int m_stride = width;
    int out_idx = y * stride + x * 3;

    float m11 = (float)mask[y1 * m_stride + x1];
    float m12 = (float)mask[y1 * m_stride + x2];
    float m21 = (float)mask[y2 * m_stride + x1];
    float m22 = (float)mask[y2 * m_stride + x2];

    float alpha = (m11 * wx1 * wy1 + m12 * wx2 * wy1 + m21 * wx1 * wy2 + m22 * wx2 * wy2) / 255.0f;
    
    for (int c = 0; c < 3; ++c) {
        float p_val = 
            (float)src[y1 * stride + x1 * 3 + c] * wx1 * wy1 +
            (float)src[y1 * stride + x2 * 3 + c] * wx2 * wy1 +
            (float)src[y2 * stride + x1 * 3 + c] * wx1 * wy2 +
            (float)src[y2 * stride + x2 * 3 + c] * wx2 * wy2;

        float bg_val = 0.0f;
        if (use_bg == 1) {
            bg_val = (float)bg[y * stride + x * 3 + c];
            dst[out_idx + c] = (unsigned char)(p_val * alpha + bg_val * (1.0f - alpha));
        } else {
            dst[out_idx + c] = (unsigned char)p_val;
        }
    }
}
'''

class BeautyEngine:
    def __init__(self):
        """
        [Mode A] Real-time Beauty Engine (GPU Edition V12.0)
        - V12.0: Self-Healing Background (Dynamic Update)
        - V11.0: Zero Distortion
        - V10.2: Smart Smooth Logic
        """
        print("💄 [BeautyEngine] GPU 엔진 V12.0 (Self-Healing Background)")
        
        self.map_scale = 0.25 
        
        self.cache_w = 0
        self.cache_h = 0
        self.gpu_initialized = False
        
        self.gpu_dx = None
        self.gpu_dy = None
        self.prev_gpu_dx = None
        self.prev_gpu_dy = None
        
        self.bg_gpu = None
        self.has_bg = False
        
        # [V12.0] Background Learning Rate
        # 값이 클수록 조명 변화에 빨리 적응하지만, 마스크 오차 시 잔상(Ghosting)이 생길 수 있음.
        # 0.05 정도면 약 20프레임(0.6초) 만에 조명 변화를 따라감.
        self.bg_learning_rate = 0.05
        
        # [V10.1] 움직임 속도 추적
        self.prev_face_center = None
        self.prev_body_center = None 
        self.current_alpha = 0.85

        self.warp_params = [] 
        
        if HAS_CUDA:
            self.warp_kernel = cp.RawKernel(WARP_KERNEL_CODE, 'warp_kernel')
            self.composite_kernel = cp.RawKernel(COMPOSITE_KERNEL_CODE, 'composite_kernel')
            self.update_bg_kernel = cp.RawKernel(UPDATE_BG_KERNEL_CODE, 'update_bg_kernel')
            
            # 초기 배경 로드 (시작점)
            self._auto_load_background()

    def _auto_load_background(self):
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(root_dir, "recorded_data")
            sessions = sorted(glob.glob(os.path.join(data_dir, "*")), reverse=True)
            
            for session in sessions:
                bg_path = os.path.join(session, "background.jpg")
                if os.path.exists(bg_path):
                    print(f"🖼️ [BeautyEngine] 초기 배경 로드: {os.path.basename(session)}")
                    bg_img = cv2.imread(bg_path)
                    if bg_img is not None:
                        self.bg_cpu_raw = bg_img
                        self.has_bg = True
                        return
            
            print("⚠️ [BeautyEngine] 저장된 배경이 없습니다. 실시간 학습으로 시작합니다.")
            self.has_bg = True # 파일이 없어도 빈 캔버스에서 시작 가능 (검은색 -> 실시간 채움)
            self.bg_cpu_raw = None # 나중에 0으로 초기화됨
            
        except Exception as e:
            print(f"❌ [BeautyEngine] 배경 로드 중 오류: {e}")
            self.has_bg = False

    def process(self, frame, faces, body_landmarks=None, params=None, mask=None):
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

        # 그리드/버퍼 초기화
        if self.cache_w != w or self.cache_h != h:
            self.cache_w, self.cache_h = w, h
            self.gpu_initialized = False
            self.prev_gpu_dx = None 
            self.prev_gpu_dy = None
            self.prev_face_center = None
            self.prev_body_center = None
            
            # [V12.0] 배경 버퍼 초기화
            if self.has_bg:
                if self.bg_cpu_raw is not None:
                    # 파일이 있으면 그걸로 시작
                    bg_resized = cv2.resize(self.bg_cpu_raw, (w, h))
                    self.bg_gpu = cp.asarray(bg_resized)
                else:
                    # 파일이 없으면 현재 프레임으로 시작 (혹은 검은색)
                    print("⚡ [BeautyEngine] 배경 파일 없음 -> 현재 프레임으로 초기화")
                    self.bg_gpu = cp.array(frame_gpu) # Copy current frame as init BG
            else:
                self.bg_gpu = cp.zeros_like(frame_gpu)
                
            print(f"⚡ [BeautyEngine] Grid Cache Rebuilt: {w}x{h}")

        sw, sh = int(w * self.map_scale), int(h * self.map_scale)
        
        if not self.gpu_initialized:
            self.gpu_dx = cp.zeros((sh, sw), dtype=cp.float32)
            self.gpu_dy = cp.zeros((sh, sw), dtype=cp.float32)
            self.gpu_initialized = True

        self.warp_params.clear() 
        has_deformation = False
        
        # [V12.0] 마스크 준비 (GPU)
        mask_gpu = None
        use_bg = 0
        if self.has_bg and mask is not None:
            if hasattr(mask, 'device'): 
                mask_gpu = mask
            else:
                mask_gpu = cp.asarray(mask)
            use_bg = 1
        else:
            mask_gpu = cp.zeros((h, w), dtype=cp.uint8)

        # =================================================================
        # [V12.0 Core] Self-Healing Background Update
        # 워핑하기 전에, 현재 프레임의 배경 부분을 학습합니다.
        # =================================================================
        if use_bg == 1:
            block_dim = (32, 32)
            grid_dim = ((w + block_dim[0] - 1) // block_dim[0], 
                        (h + block_dim[1] - 1) // block_dim[1])
            
            self.update_bg_kernel(
                grid_dim, block_dim,
                (frame_gpu, mask_gpu, self.bg_gpu, w, h, self.bg_learning_rate)
            )

        # =================================================================
        # Smart Smooth Logic (V10.2)
        # =================================================================
        max_velocity = 0.0
        
        # Face Velocity
        current_face_center = None
        if faces:
            current_face_center = np.mean(faces[0].landmarks, axis=0)
            if self.prev_face_center is not None:
                face_vel = np.linalg.norm(current_face_center - self.prev_face_center)
                max_velocity = max(max_velocity, face_vel)
        
        # Body Velocity
        current_body_center = None
        body_cpu = None
        if body_landmarks is not None:
            if hasattr(body_landmarks, 'get'): body_cpu = body_landmarks.get()
            else: body_cpu = body_landmarks
            valid_points = []
            for idx in [5, 6, 11, 12]:
                if idx < len(body_cpu) and body_cpu[idx][2] > 0.4:
                    valid_points.append(body_cpu[idx][:2])
            if valid_points:
                current_body_center = np.mean(valid_points, axis=0)
                if self.prev_body_center is not None:
                    body_vel = np.linalg.norm(current_body_center - self.prev_body_center)
                    max_velocity = max(max_velocity, body_vel)

        self.prev_face_center = current_face_center
        self.prev_body_center = current_body_center
        
        min_vel, max_vel = 0.5, 6.0
        max_alpha, min_alpha = 0.96, 0.15
        target_alpha = max_alpha
        if max_velocity <= min_vel: target_alpha = max_alpha
        elif max_velocity >= max_vel: target_alpha = min_alpha
        else:
            ratio = (max_velocity - min_vel) / (max_vel - min_vel)
            target_alpha = max_alpha - ratio * (max_alpha - min_alpha)

        self.current_alpha = self.current_alpha * 0.8 + target_alpha * 0.2
            
        # =================================================================
        
        # Body Reshaping
        if body_cpu is not None:
            scaled_body = body_cpu[:, :2] * self.map_scale
            for key in ['shoulder_narrow', 'ribcage_slim', 'waist_slim', 'hip_widen']:
                val = params.get(key, 0)
                if val > 0:
                    if key == 'shoulder_narrow': self._collect_shoulder_params(scaled_body, val)
                    elif key == 'ribcage_slim': self._collect_ribcage_params(scaled_body, val)
                    elif key == 'waist_slim': self._collect_waist_params(scaled_body, val)
                    elif key == 'hip_widen': self._collect_hip_params(scaled_body, val)
                    has_deformation = True

        # Face Reshaping
        if faces:
            face_v = params.get('face_v', 0)
            eye_scale = params.get('eye_scale', 0)
            head_scale = params.get('head_scale', 0) 
            for face in faces:
                lm_small = face.landmarks * self.map_scale
                if face_v > 0: self._collect_face_contour_params(lm_small, face_v)
                if eye_scale > 0: self._collect_eyes_params(lm_small, eye_scale)
                if head_scale != 0: self._collect_head_params(lm_small, head_scale)

            if face_v > 0 or eye_scale > 0 or head_scale != 0:
                has_deformation = True

        self.gpu_dx.fill(0)
        self.gpu_dy.fill(0)

        # Warp Vector
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

        # Smoothing & Compositing
        if has_deformation or (self.prev_gpu_dx is not None):
            self._apply_temporal_smoothing_fast(self.current_alpha)
            
            cupyx.scipy.ndimage.gaussian_filter(self.gpu_dx, sigma=5, output=self.gpu_dx)
            cupyx.scipy.ndimage.gaussian_filter(self.gpu_dy, sigma=5, output=self.gpu_dy)
            
            result_gpu = cp.empty_like(frame_gpu)
            
            block_dim = (32, 32)
            grid_dim = ((w + block_dim[0] - 1) // block_dim[0], 
                        (h + block_dim[1] - 1) // block_dim[1])
            
            scale = int(1.0 / self.map_scale) 
            
            self.composite_kernel(
                grid_dim, block_dim,
                (frame_gpu, mask_gpu, self.bg_gpu, result_gpu, self.gpu_dx, self.gpu_dy, 
                 w, h, sw, sh, scale, use_bg)
            )
            
            if is_gpu_input:
                return result_gpu
            else:
                return result_gpu.get()
        else:
            return frame_gpu if is_gpu_input else frame

    # ==========================================================
    # [Reshaping Helpers] (동일함)
    # ==========================================================
    def _collect_shoulder_params(self, keypoints, strength):
        l_sh, r_sh = keypoints[5], keypoints[6]
        width = np.linalg.norm(l_sh - r_sh)
        if width < 3: return
        center = (l_sh + r_sh) / 2
        radius = int(width * 0.6)
        s = strength * 0.3
        self._add_param(l_sh, radius, s, mode=1, vector=(center-l_sh))
        self._add_param(r_sh, radius, s, mode=1, vector=(center-r_sh))

    def _collect_ribcage_params(self, keypoints, strength):
        l_sh, r_sh = keypoints[5], keypoints[6]
        l_hip, r_hip = keypoints[11], keypoints[12]
        l_rib = l_sh * 0.65 + l_hip * 0.35
        r_rib = r_sh * 0.65 + r_hip * 0.35
        center = (l_rib + r_rib) / 2
        width = np.linalg.norm(l_rib - r_rib)
        if width < 3: return
        radius = int(width * 0.7)
        s = strength * 0.4
        self._add_param(l_rib, radius, s, mode=1, vector=(center-l_rib))
        self._add_param(r_rib, radius, s, mode=1, vector=(center-r_rib))

    def _collect_waist_params(self, keypoints, strength):
        l_sh, r_sh = keypoints[5], keypoints[6]
        l_hip, r_hip = keypoints[11], keypoints[12]
        l_waist = l_sh * 0.4 + l_hip * 0.6
        r_waist = r_sh * 0.4 + r_hip * 0.6
        center = (l_waist + r_waist) / 2
        width = np.linalg.norm(l_waist - r_waist)
        if width < 3: return 
        radius = int(width * 0.6)
        s = strength * 0.4
        self._add_param(l_waist, radius, s, mode=1, vector=(center-l_waist))
        self._add_param(r_waist, radius, s, mode=1, vector=(center-r_waist))

    def _collect_hip_params(self, keypoints, strength):
        l_hip, r_hip = keypoints[11], keypoints[12]
        width = np.linalg.norm(l_hip - r_hip)
        if width < 3: return
        center = (l_hip + r_hip) / 2
        radius = int(width * 0.7)
        s = strength * 0.3
        self._add_param(l_hip, radius, s, mode=1, vector=(l_hip-center))
        self._add_param(r_hip, radius, s, mode=1, vector=(r_hip-center))

    def _collect_head_params(self, lm, strength):
        chin = lm[152]
        forehead = lm[10]
        height = np.linalg.norm(chin - forehead)
        up_vec = forehead - chin
        up_vec /= (np.linalg.norm(up_vec) + 1e-6)
        center = np.mean(lm, axis=0) + up_vec * (height * 0.5)
        radius = int(height * 1.6)
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