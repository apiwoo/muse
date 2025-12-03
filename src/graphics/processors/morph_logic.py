# Project MUSE - morph_logic.py
# Landmark-based Deformation Logic
# (C) 2025 MUSE Corp. All rights reserved.

import numpy as np
from ai.tracking.facemesh import FaceMesh

class MorphLogic:
    """
    [Logic Core]
    랜드마크 좌표를 기반으로 TPS(Thin Plate Spline) 변형 파라미터를 계산합니다.
    """
    def __init__(self):
        self.warp_params = []

    def clear(self):
        self.warp_params.clear()

    def get_params(self):
        return self.warp_params

    def _add_param(self, c, r, s, m=0, v=None):
        vx, vy = 0.0, 0.0
        if v is not None:
            norm = np.linalg.norm(v) + 1e-6
            vx, vy = v[0]/norm, v[1]/norm
        self.warp_params.append([c[0], c[1], float(r), float(s), vx, vy, float(m)])

    # --- Body Reshaping ---

    def collect_shoulder_params(self, kpts, s):
        if len(kpts) < 7: return
        l, r = kpts[5], kpts[6]
        c = (l+r)/2
        w = np.linalg.norm(l-r)
        if w < 1: return
        self._add_param(l, w*0.6, s*0.3, 1, c-l)
        self._add_param(r, w*0.6, s*0.3, 1, c-r)

    def collect_ribcage_params(self, kpts, s):
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

    def collect_waist_params(self, kpts, s):
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

    def collect_hip_params(self, kpts, s):
        if len(kpts) < 13: return
        l_hip, r_hip = kpts[11], kpts[12]
        c = (l_hip + r_hip) / 2
        w = np.linalg.norm(l_hip - r_hip)
        if w < 1: return
        # Widen: Vector away from center
        self._add_param(l_hip, w*0.7, s*0.3, 1, l_hip-c)
        self._add_param(r_hip, w*0.7, s*0.3, 1, r_hip-c)

    # --- Face Reshaping ---

    def collect_head_params(self, lm, s):
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

    def collect_eyes_params(self, lm, s):
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

    def collect_face_contour_params(self, lm, s):
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