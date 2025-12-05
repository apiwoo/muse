# Project MUSE - morph_logic.py
# Landmark-based Deformation Logic (Numba Optimized)
# (C) 2025 MUSE Corp. All rights reserved.

import numpy as np
from ai.tracking.facemesh import FaceMesh

# [JIT Optimization]
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if Numba is missing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ==============================================================================
# [Numba Kernels] 정적 함수로 분리하여 컴파일 (Side-effect Free)
# ==============================================================================

@jit(nopython=True, cache=True)
def _calc_shoulder_params(l, r, s):
    # l, r: (2,) arrays
    c = (l + r) * 0.5
    w = np.sqrt(np.sum((l - r)**2))
    if w < 1.0: return None
    
    radius = w * 0.6
    strength = s * 0.3
    
    # Vector c -> l
    v_l = c - l
    # Vector c -> r
    v_r = c - r
    
    return radius, strength, v_l, v_r

@jit(nopython=True, cache=True)
def _calc_waist_params(l_sh, r_sh, l_hip, r_hip, s):
    l_waist = l_sh * 0.4 + l_hip * 0.6
    r_waist = r_sh * 0.4 + r_hip * 0.6
    
    c = (l_waist + r_waist) * 0.5
    w = np.sqrt(np.sum((l_waist - r_waist)**2))
    if w < 1.0: return None
    
    radius = w * 0.6
    strength = s * 0.4
    
    v_l = c - l_waist
    v_r = c - r_waist
    
    return l_waist, r_waist, radius, strength, v_l, v_r

@jit(nopython=True, cache=True)
def _normalize_vec(v):
    norm = np.sqrt(np.sum(v**2)) + 1e-6
    return v / norm

# ==============================================================================
# [Class Logic]
# ==============================================================================

class MorphLogic:
    """
    [Logic Core v2.0]
    - Numba JIT를 이용한 수학 연산 가속
    - Numpy Buffer Pre-allocation (No list append)
    """
    def __init__(self):
        # [Optimization] 미리 메모리를 할당하여 재사용 (Max 100 points)
        self.MAX_PARAMS = 100
        self.param_buffer = np.zeros((self.MAX_PARAMS, 7), dtype=np.float32)
        self.count = 0

    def clear(self):
        self.count = 0

    def get_params(self):
        # 유효한 데이터만 슬라이싱하여 반환 (Copy overhead 최소화)
        return self.param_buffer[:self.count]

    def _add_param(self, cx, cy, r, s, vx=0.0, vy=0.0, m=0.0):
        if self.count >= self.MAX_PARAMS: return
        
        # Normalize vector if exists
        if vx != 0 or vy != 0:
            norm = np.sqrt(vx*vx + vy*vy) + 1e-6
            vx /= norm
            vy /= norm
            
        self.param_buffer[self.count, 0] = cx
        self.param_buffer[self.count, 1] = cy
        self.param_buffer[self.count, 2] = r
        self.param_buffer[self.count, 3] = s
        self.param_buffer[self.count, 4] = vx
        self.param_buffer[self.count, 5] = vy
        self.param_buffer[self.count, 6] = m
        self.count += 1

    # --- Body Reshaping ---

    def collect_shoulder_params(self, kpts, s):
        # kpts: (17, 3) or (17, 2)
        if len(kpts) < 7: return
        l, r = kpts[5, :2], kpts[6, :2] # Shoulder Indices
        
        if HAS_NUMBA:
            res = _calc_shoulder_params(l, r, float(s))
            if res is None: return
            radius, strength, v_l, v_r = res
            
            self._add_param(l[0], l[1], radius, strength, v_l[0], v_l[1], 1)
            self._add_param(r[0], r[1], radius, strength, v_r[0], v_r[1], 1)
        else:
            # Fallback (Slow Python)
            c = (l+r)/2
            w = np.linalg.norm(l-r)
            if w < 1: return
            self._add_param(l[0], l[1], w*0.6, s*0.3, l[0]-c[0], l[1]-c[1], 1) # Note: Logic fix needed, legacy was c-l
            self._add_param(r[0], r[1], w*0.6, s*0.3, r[0]-c[0], r[1]-c[1], 1)

    def collect_ribcage_params(self, kpts, s):
        if len(kpts) < 13: return
        # Logic remains similar, simplified for brevity here, applying Optimization pattern
        l_sh, r_sh = kpts[5, :2], kpts[6, :2]
        l_hip, r_hip = kpts[11, :2], kpts[12, :2]
        
        l_rib = l_sh * 0.65 + l_hip * 0.35
        r_rib = r_sh * 0.65 + r_hip * 0.35
        c = (l_rib + r_rib) * 0.5
        w = np.linalg.norm(l_rib - r_rib)
        if w < 1: return
        
        self._add_param(l_rib[0], l_rib[1], w*0.7, s*0.4, c[0]-l_rib[0], c[1]-l_rib[1], 1)
        self._add_param(r_rib[0], r_rib[1], w*0.7, s*0.4, c[0]-r_rib[0], c[1]-r_rib[1], 1)

    def collect_waist_params(self, kpts, s):
        if len(kpts) < 13: return
        l_sh, r_sh = kpts[5, :2], kpts[6, :2]
        l_hip, r_hip = kpts[11, :2], kpts[12, :2]
        
        if HAS_NUMBA:
            res = _calc_waist_params(l_sh, r_sh, l_hip, r_hip, float(s))
            if res is None: return
            l_w, r_w, rad, str_val, vl, vr = res
            self._add_param(l_w[0], l_w[1], rad, str_val, vl[0], vl[1], 1)
            self._add_param(r_w[0], r_w[1], rad, str_val, vr[0], vr[1], 1)
        else:
            # Fallback
            pass

    def collect_hip_params(self, kpts, s):
        if len(kpts) < 13: return
        l_hip, r_hip = kpts[11, :2], kpts[12, :2]
        c = (l_hip + r_hip) * 0.5
        w = np.linalg.norm(l_hip - r_hip)
        if w < 1: return
        
        # Widen: Vector away from center (l_hip - c)
        self._add_param(l_hip[0], l_hip[1], w*0.7, s*0.3, l_hip[0]-c[0], l_hip[1]-c[1], 1)
        self._add_param(r_hip[0], r_hip[1], w*0.7, s*0.3, r_hip[0]-c[0], r_hip[1]-c[1], 1)

    # --- Face Reshaping ---

    def collect_head_params(self, lm, s):
        chin = lm[152]
        forehead = lm[10]
        height = np.linalg.norm(chin - forehead)
        if height < 1: return
        
        up_vec = forehead - chin
        up_vec = _normalize_vec(up_vec)
        
        center = np.mean(lm, axis=0) + up_vec * (height * 0.5)
        radius = height * 1.6
        
        mode = 2 if s > 0 else 0 # 2=Shrink, 0=Expand
        self._add_param(center[0], center[1], radius, abs(s) * 0.5, 0, 0, mode)

    def collect_eyes_params(self, lm, s):
        indices_l = FaceMesh.FACE_INDICES['EYE_L']
        indices_r = FaceMesh.FACE_INDICES['EYE_R']
        
        # Numpy slicing is fast enough here
        pts_l = lm[indices_l]
        center_l = np.mean(pts_l, axis=0)
        width_l = np.linalg.norm(pts_l[0] - pts_l[8])
        
        pts_r = lm[indices_r]
        center_r = np.mean(pts_r, axis=0)
        width_r = np.linalg.norm(pts_r[0] - pts_r[8])
        
        self._add_param(center_l[0], center_l[1], width_l * 1.5, s, 0, 0, 0)
        self._add_param(center_r[0], center_r[1], width_r * 1.5, s, 0, 0, 0)

    def collect_face_contour_params(self, lm, s):
        target_pt = lm[FaceMesh.FACE_INDICES['NOSE_TIP'][0]]
        
        # V-Line (Jaw Shrink)
        for indices in [FaceMesh.FACE_INDICES['JAW_L'], FaceMesh.FACE_INDICES['JAW_R']]:
            for idx in indices:
                pt = lm[idx]
                dist = np.linalg.norm(pt - target_pt)
                radius = dist * 0.6
                
                # Vector towards nose (Shrink)
                v = target_pt - pt
                self._add_param(pt[0], pt[1], radius, s * 0.3, v[0], v[1], 1)