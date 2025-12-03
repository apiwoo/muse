# Project MUSE - beauty_engine.py
# V15.0: Grid Warping + Smart Composite (Clean Plate) - Full Implementation
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import os
import glob
from ai.tracking.facemesh import FaceMesh

try:
    import cupy as cp
    import cupyx.scipy.ndimage
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("⚠️ [BeautyEngine] CuPy not found. Fallback to CPU Mode.")

# [Kernel 1] Grid Generation (TPS Logic)
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
            if (mode == 0) { // Expand/Shrink (Radial)
                acc_dx -= diff_x * factor;
                acc_dy -= diff_y * factor;
            } else if (mode == 1) { // Shift (Vector)
                acc_dx -= vx * factor * r * 0.5f;
                acc_dy -= vy * factor * r * 0.5f;
            } else if (mode == 2) { // Shrink (Inverse Radial)
                acc_dx += diff_x * factor;
                acc_dy += diff_y * factor;
            }
        }
    }
    dx[idx] += acc_dx;
    dy[idx] += acc_dy;
}
'''

# [Kernel 2] Smart Composite (The Clean Plate Logic)
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

    // 1. Get Displacement (Bilinear Interpolation from small grid)
    int sx = x / scale;
    int sy = y / scale;
    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;
    int s_idx = sy * small_width + sx;

    float shift_x = dx_small[s_idx] * (float)scale;
    float shift_y = dy_small[s_idx] * (float)scale;

    // Source Coordinates (u, v)
    float u = (float)x + shift_x;
    float v = (float)y + shift_y;

    int u_i = (int)u;
    int v_i = (int)v;

    // Boundary Check
    if (u_i < 0 || u_i >= width || v_i < 0 || v_i >= height) {
        if (use_bg) {
            int idx = (y * width + x) * 3;
            dst[idx+0] = bg[idx+0];
            dst[idx+1] = bg[idx+1];
            dst[idx+2] = bg[idx+2];
        } else {
            int idx = (y * width + x) * 3;
            dst[idx] = 0; dst[idx+1] = 0; dst[idx+2] = 0;
        }
        return;
    }

    // 2. Check Mask at Source (u, v)
    int mask_idx = v_i * width + u_i;
    unsigned char m_val = mask[mask_idx];

    int dst_idx = (y * width + x) * 3;
    int src_idx = (v_i * width + u_i) * 3;
    int bg_idx = (y * width + x) * 3; // BG uses original coords

    if (m_val > 10) { 
        // Person -> Warp
        dst[dst_idx+0] = src[src_idx+0];
        dst[dst_idx+1] = src[src_idx+1];
        dst[dst_idx+2] = src[src_idx+2];
    } else {
        // Background -> Static Clean Plate
        if (use_bg) {
            dst[dst_idx+0] = bg[bg_idx+0];
            dst[dst_idx+1] = bg[bg_idx+1];
            dst[dst_idx+2] = bg[bg_idx+2];
        } else {
            dst[dst_idx+0] = src[src_idx+0]; // Fallback
            dst[dst_idx+1] = src[src_idx+1];
            dst[dst_idx+2] = src[src_idx+2];
        }
    }
}
'''

class BeautyEngine:
    def __init__(self, profiles=[]):
        print("💄 [BeautyEngine] V15.0 Smart Composite Ready")
        self.map_scale = 0.25 
        self.cache_w = 0
        self.cache_h = 0
        self.gpu_initialized = False
        
        self.gpu_dx = None
        self.gpu_dy = None
        self.prev_gpu_dx = None
        self.prev_gpu_dy = None
        
        self.bg_buffers = {}
        self.active_profile = 'default'
        self.bg_gpu = None
        self.has_bg = False
        
        self.warp_params = [] 
        self.current_alpha = 0.85
        self.prev_face_center = None
        
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.root_dir, "recorded_data", "personal_data")
        
        if HAS_CUDA:
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.warp_kernel = cp.RawKernel(WARP_KERNEL_CODE, 'warp_kernel')
            self.composite_kernel = cp.RawKernel(COMPOSITE_KERNEL_CODE, 'composite_kernel')
            self._load_all_backgrounds(profiles)

    def _load_all_backgrounds(self, profiles):
        if not HAS_CUDA: return
        for p in profiles:
            bg_path = os.path.join(self.data_dir, p, "background.jpg")
            if os.path.exists(bg_path):
                img = cv2.imread(bg_path)
                if img is not None:
                    self.bg_buffers[p] = {'cpu': img, 'gpu': None}
            else:
                self.bg_buffers[p] = {'cpu': None, 'gpu': None}

    def set_profile(self, profile_name):
        if profile_name in self.bg_buffers:
            self.active_profile = profile_name
            if self.bg_buffers[profile_name]['gpu'] is not None:
                self.bg_gpu = self.bg_buffers[profile_name]['gpu']
                self.has_bg = True
            else:
                self.has_bg = False
        else:
            self.active_profile = profile_name
            self.bg_buffers[profile_name] = {'cpu': None, 'gpu': None}
            self.has_bg = False

    def reset_background(self, frame):
        if not HAS_CUDA or frame is None: return
        with self.stream:
            new_bg_gpu = cp.array(frame) if not hasattr(frame, 'device') else cp.copy(frame)
            self.bg_gpu = new_bg_gpu
            if self.active_profile not in self.bg_buffers:
                self.bg_buffers[self.active_profile] = {'cpu': None, 'gpu': None}
            self.bg_buffers[self.active_profile]['gpu'] = new_bg_gpu
            self.has_bg = True
        
        self.stream.synchronize()
        frame_cpu = cp.asnumpy(new_bg_gpu)
        save_path = os.path.join(self.data_dir, self.active_profile, "background.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame_cpu)

    def process(self, frame, faces, body_landmarks=None, params=None, mask=None):
        if frame is None or not HAS_CUDA: return frame
        if params is None: params = {}

        is_gpu_input = hasattr(frame, 'device')
        with self.stream:
            if is_gpu_input:
                frame_gpu = frame 
                h, w = frame.shape[:2]
            else:
                h, w = frame.shape[:2]
                frame_gpu = cp.asarray(frame)

            if self.cache_w != w or self.cache_h != h:
                self.cache_w, self.cache_h = w, h
                self.gpu_initialized = False
                self._init_bg_buffers(w, h, frame_gpu)

            sw, sh = int(w * self.map_scale), int(h * self.map_scale)
            if not self.gpu_initialized:
                self.gpu_dx = cp.zeros((sh, sw), dtype=cp.float32)
                self.gpu_dy = cp.zeros((sh, sw), dtype=cp.float32)
                self.prev_gpu_dx = cp.zeros((sh, sw), dtype=cp.float32)
                self.prev_gpu_dy = cp.zeros((sh, sw), dtype=cp.float32)
                self.gpu_initialized = True

            if self.bg_gpu is None:
                self.bg_gpu = cp.copy(frame_gpu)
                self.has_bg = True

            # Mask Handling
            if self.has_bg and mask is not None:
                if hasattr(mask, 'device'): mask_gpu = mask
                else: mask_gpu = cp.asarray(mask)
                use_bg = 1
            else:
                mask_gpu = cp.zeros((h, w), dtype=cp.uint8)
                use_bg = 0

            # Calculate Warp Params
            self.warp_params.clear()
            has_deformation = False
            
            # 1. Body Reshaping Logic
            body_cpu = body_landmarks.get() if hasattr(body_landmarks, 'get') else body_landmarks
            if body_cpu is not None:
                scaled_body = body_cpu[:, :2] * self.map_scale
                
                if params.get('shoulder_narrow', 0) > 0:
                    self._collect_shoulder_params(scaled_body, params['shoulder_narrow'])
                    has_deformation = True
                
                if params.get('ribcage_slim', 0) > 0:
                    self._collect_ribcage_params(scaled_body, params['ribcage_slim'])
                    has_deformation = True
                    
                if params.get('waist_slim', 0) > 0:
                    self._collect_waist_params(scaled_body, params['waist_slim'])
                    has_deformation = True
                    
                if params.get('hip_widen', 0) > 0:
                    self._collect_hip_params(scaled_body, params['hip_widen'])
                    has_deformation = True
            
            # 2. Face Reshaping Logic
            if faces:
                face_v = params.get('face_v', 0)
                eye_scale = params.get('eye_scale', 0)
                head_scale = params.get('head_scale', 0)
                
                for face in faces:
                    lm_small = face.landmarks * self.map_scale
                    if face_v > 0: 
                        self._collect_face_contour_params(lm_small, face_v)
                        has_deformation = True
                    if eye_scale > 0: 
                        self._collect_eyes_params(lm_small, eye_scale)
                        has_deformation = True
                    if head_scale != 0: 
                        self._collect_head_params(lm_small, head_scale)
                        has_deformation = True

            self.gpu_dx.fill(0)
            self.gpu_dy.fill(0)

            if self.warp_params:
                params_arr = np.array(self.warp_params, dtype=np.float32)
                params_gpu = cp.asarray(params_arr)
                
                block_dim = (16, 16)
                grid_dim = ((sw + block_dim[0] - 1) // block_dim[0], (sh + block_dim[1] - 1) // block_dim[1])
                self.warp_kernel(grid_dim, block_dim, (self.gpu_dx, self.gpu_dy, params_gpu, len(self.warp_params), sw, sh))
                
            # Smoothing Logic
            if has_deformation or (self.prev_gpu_dx is not None):
                self._apply_temporal_smoothing_fast(self.current_alpha)
                
                cupyx.scipy.ndimage.gaussian_filter(self.gpu_dx, sigma=5, output=self.gpu_dx)
                cupyx.scipy.ndimage.gaussian_filter(self.gpu_dy, sigma=5, output=self.gpu_dy)

                # Composite
                result_gpu = cp.empty_like(frame_gpu)
                block_dim = (32, 32)
                grid_dim = ((w + block_dim[0] - 1) // block_dim[0], (h + block_dim[1] - 1) // block_dim[1])
                scale = int(1.0 / self.map_scale)
                
                self.composite_kernel(
                    grid_dim, block_dim,
                    (frame_gpu, mask_gpu, self.bg_gpu, result_gpu, self.gpu_dx, self.gpu_dy, 
                     w, h, sw, sh, scale, use_bg)
                )

                if is_gpu_input: return result_gpu
                else: return result_gpu.get()
            
            # No changes
            return frame_gpu if is_gpu_input else frame

    def _init_bg_buffers(self, w, h, tmpl):
        for p, data in self.bg_buffers.items():
            if data['gpu'] is None:
                if data['cpu'] is not None:
                    rz = cv2.resize(data['cpu'], (w, h))
                    data['gpu'] = cp.asarray(rz)
                else:
                    data['gpu'] = cp.zeros_like(tmpl)

    # --- Reshaping Helpers (Full Implementation) ---

    def _collect_shoulder_params(self, kpts, s):
        if len(kpts) < 7: return
        l, r = kpts[5], kpts[6]
        c = (l+r)/2
        w = np.linalg.norm(l-r)
        if w < 1: return
        self._add_param(l, w*0.6, s*0.3, 1, c-l)
        self._add_param(r, w*0.6, s*0.3, 1, c-r)

    def _collect_ribcage_params(self, kpts, s):
        if len(kpts) < 13: return
        l_sh, r_sh = kpts[5], kpts[6]
        l_hip, r_hip = kpts[11], kpts[12]
        l_rib = l_sh * 0.65 + l_hip * 0.35
        r_rib = r_sh * 0.65 + r_hip * 0.35
        c = (l_rib + r_rib) / 2
        w = np.linalg.norm(l_rib - r_rib)
        if w < 1: return
        self._add_param(l_rib, w*0.7, s*0.4, 1, c-l_rib)
        self._add_param(r_rib, w*0.7, s*0.4, 1, c-r_rib)

    def _collect_waist_params(self, kpts, s):
        if len(kpts) < 13: return
        l_sh, r_sh = kpts[5], kpts[6]
        l_hip, r_hip = kpts[11], kpts[12]
        l_waist = l_sh * 0.4 + l_hip * 0.6
        r_waist = r_sh * 0.4 + r_hip * 0.6
        c = (l_waist + r_waist) / 2
        w = np.linalg.norm(l_waist - r_waist)
        if w < 1: return
        self._add_param(l_waist, w*0.6, s*0.4, 1, c-l_waist)
        self._add_param(r_waist, w*0.6, s*0.4, 1, c-r_waist)

    def _collect_hip_params(self, kpts, s):
        if len(kpts) < 13: return
        l_hip, r_hip = kpts[11], kpts[12]
        c = (l_hip + r_hip) / 2
        w = np.linalg.norm(l_hip - r_hip)
        if w < 1: return
        # Widen: Vector away from center
        self._add_param(l_hip, w*0.7, s*0.3, 1, l_hip-c)
        self._add_param(r_hip, w*0.7, s*0.3, 1, r_hip-c)

    def _collect_head_params(self, lm, s):
        chin = lm[152]
        forehead = lm[10]
        height = np.linalg.norm(chin - forehead)
        if height < 1: return
        up_vec = forehead - chin
        up_vec /= (np.linalg.norm(up_vec) + 1e-6)
        center = np.mean(lm, axis=0) + up_vec * (height * 0.5)
        radius = int(height * 1.6)
        
        # Scaling mode: 2=Shrink, 0=Expand
        if s > 0:
            self._add_param(center, radius, s * 0.5, 2)
        else:
            self._add_param(center, radius, abs(s) * 0.5, 0)

    def _collect_eyes_params(self, lm, s):
        # Indices for Left/Right Eye (MediaPipe 478)
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
        
        # Mode 0 = Expand (Big Eyes)
        self._add_param(center_l, radius_l, s, 0)
        self._add_param(center_r, radius_r, s, 0)

    def _collect_face_contour_params(self, lm, s):
        target_pt = lm[FaceMesh.FACE_INDICES['NOSE_TIP'][0]]
        for idx in FaceMesh.FACE_INDICES['JAW_L']:
            pt = lm[idx]
            radius = int(np.linalg.norm(pt - target_pt) * 0.6)
            vector = target_pt - pt
            self._add_param(pt, radius, s * 0.3, 1, vector)
        for idx in FaceMesh.FACE_INDICES['JAW_R']:
            pt = lm[idx]
            radius = int(np.linalg.norm(pt - target_pt) * 0.6)
            vector = target_pt - pt
            self._add_param(pt, radius, s * 0.3, 1, vector)

    def _add_param(self, c, r, s, m=0, v=None):
        vx, vy = 0.0, 0.0
        if v is not None:
            norm = np.linalg.norm(v) + 1e-6
            vx, vy = v[0]/norm, v[1]/norm
        self.warp_params.append([c[0], c[1], float(r), float(s), vx, vy, float(m)])

    def _apply_temporal_smoothing_fast(self, alpha):
        if self.prev_gpu_dx is None:
            self.prev_gpu_dx = self.gpu_dx.copy()
            self.prev_gpu_dy = self.gpu_dy.copy()
            return

        beta = 1.0 - alpha
        self.gpu_dx *= beta
        self.gpu_dx += self.prev_gpu_dx * alpha
        self.gpu_dy *= beta
        self.gpu_dy += self.prev_gpu_dy * alpha
        
        self.prev_gpu_dx[:] = self.gpu_dx
        self.prev_gpu_dy[:] = self.gpu_dy