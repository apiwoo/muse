# Project MUSE - beauty_engine.py
# Optimized V3.1: Low-Res Vector Field + Global Smoothing
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
from ai.tracking.facemesh import FaceMesh

class BeautyEngine:
    def __init__(self):
        """
        [Mode A] Real-time Beauty Engine
        - V3.1 Optimization:
          1. Low-Res Vector Field: 벡터 연산을 1/4 해상도에서 수행 (속도 16배↑)
          2. Global Smoothing: 벡터 맵 전체에 블러를 주어 턱선 울퉁불퉁함 제거
          3. Grid Caching: 기본 그리드 재사용
        """
        print("💄 [BeautyEngine] 초고속 스무딩 엔진 초기화 (V3.1 - 270p Field)")
        
        # [Optimization Config]
        # 벡터 필드 해상도 비율 (0.25 = 1/4 크기)
        # 1080p -> 270p에서 연산하므로 매우 빠름
        self.map_scale = 0.25 
        
        # Grid Cache
        self.cache_w = 0
        self.cache_h = 0
        self.base_map_x = None
        self.base_map_y = None

    def process(self, frame, faces, body_landmarks=None, params=None):
        if frame is None: return frame
        if params is None: params = {}

        h, w = frame.shape[:2]
        
        # [Step 0] Base Grid Caching (Original Size)
        # 최종 remap은 원본 해상도에서 해야 하므로 원본 크기 그리드는 필요함
        if self.cache_w != w or self.cache_h != h:
            self.cache_w, self.cache_h = w, h
            # map_x, map_y는 float32여야 함
            grid_y, grid_x = np.indices((h, w), dtype=np.float32)
            self.base_map_x = grid_x
            self.base_map_y = grid_y
            print(f"⚡ [BeautyEngine] Base Grid Cache Rebuilt: {w}x{h}")

        # [Step 1] Low-Res Vector Field 생성
        # 벡터 연산용 작은 맵 (예: 1920x1080 -> 480x270)
        sw, sh = int(w * self.map_scale), int(h * self.map_scale)
        small_dx = np.zeros((sh, sw), dtype=np.float32)
        small_dy = np.zeros((sh, sw), dtype=np.float32)
        
        has_deformation = False

        # [Step 2] Accumulate Vectors (on Small Map)
        
        # 1. Body Reshaping
        waist_strength = params.get('waist_slim', 0)
        if body_landmarks is not None and waist_strength > 0:
            # 좌표도 스케일에 맞춰 줄여서 전달해야 함
            scaled_body = body_landmarks.copy()
            scaled_body[:, :2] *= self.map_scale
            
            self._accumulate_waist(small_dx, small_dy, scaled_body, waist_strength)
            has_deformation = True

        # 2. Face Reshaping
        if faces:
            face_v = params.get('face_v', 0)
            eye_scale = params.get('eye_scale', 0)

            if face_v > 0 or eye_scale > 0:
                for face in faces:
                    # 좌표 스케일 다운
                    lm_small = face.landmarks * self.map_scale
                    
                    if face_v > 0:
                        self._accumulate_face_contour(small_dx, small_dy, lm_small, face_v)
                    if eye_scale > 0:
                        self._accumulate_eyes(small_dx, small_dy, lm_small, eye_scale)
                has_deformation = True

        # [Step 3] Upscale & Apply
        if has_deformation:
            # [Quality Key] Global Smoothing
            # 벡터 필드 전체를 블러링하여 뾰족한 부분(울퉁불퉁함)을 부드럽게 폄
            # 저해상도에서의 5px 블러는 원본에서 20px 블러 효과와 비슷함
            small_dx = cv2.GaussianBlur(small_dx, (5, 5), 0)
            small_dy = cv2.GaussianBlur(small_dy, (5, 5), 0)
            
            # 1. 원본 크기로 확대 (Linear가 가장 빠르고 부드러움)
            total_dx = cv2.resize(small_dx, (w, h), interpolation=cv2.INTER_LINEAR)
            total_dy = cv2.resize(small_dy, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 2. 이동량 보정 (좌표계가 커졌으므로 이동 거리도 비율만큼 커져야 함)
            # 1/0.25 = 4배
            scale_factor = 1.0 / self.map_scale
            total_dx *= scale_factor
            total_dy *= scale_factor
            
            # 3. 최종 맵 생성 (기본 그리드 + 변형 벡터)
            map_x = self.base_map_x + total_dx
            map_y = self.base_map_y + total_dy
            
            # 4. Remap (1회 수행)
            result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            return result
        
        return frame

    # ==========================================================
    # [Vector Accumulation Logic] (Low-Res Friendly)
    # ==========================================================
    def _accumulate_waist(self, dx_map, dy_map, keypoints, strength):
        # *주의* keypoints는 이미 scaled 된 상태임
        CONF_THRESH = 0.3
        idx_l_sh, idx_r_sh = 5, 6
        idx_l_hip, idx_r_hip = 11, 12
        
        try:
            if (keypoints[idx_l_sh, 2] < CONF_THRESH or keypoints[idx_r_sh, 2] < CONF_THRESH): return
            l_sh, r_sh = keypoints[idx_l_sh, :2], keypoints[idx_r_sh, :2]
            l_hip, r_hip = keypoints[idx_l_hip, :2], keypoints[idx_r_hip, :2]
        except IndexError: return

        l_waist = l_sh * 0.4 + l_hip * 0.6
        r_waist = r_sh * 0.4 + r_hip * 0.6
        center_waist = (l_waist + r_waist) / 2
        
        body_width = np.linalg.norm(l_waist - r_waist)
        # 저해상도이므로 최소 폭 기준도 작아짐
        if body_width < 3: return 
        
        radius = int(body_width * 0.6)
        s = strength * 0.4

        vec_l = center_waist - l_waist
        self._add_warp_vector(dx_map, dy_map, l_waist, radius, s, mode='shrink', vector=vec_l)

        vec_r = center_waist - r_waist
        self._add_warp_vector(dx_map, dy_map, r_waist, radius, s, mode='shrink', vector=vec_r)

    def _accumulate_eyes(self, dx_map, dy_map, lm, strength):
        indices_l = FaceMesh.FACE_INDICES['EYE_L']
        indices_r = FaceMesh.FACE_INDICES['EYE_R']

        pts_l = lm[indices_l]
        center_l = np.mean(pts_l, axis=0).astype(int)
        width_l = np.linalg.norm(pts_l[0] - pts_l[8])
        radius_l = int(width_l * 1.5)

        pts_r = lm[indices_r]
        center_r = np.mean(pts_r, axis=0).astype(int)
        width_r = np.linalg.norm(pts_r[0] - pts_r[8])
        radius_r = int(width_r * 1.5)

        self._add_warp_vector(dx_map, dy_map, center_l, radius_l, strength, mode='expand')
        self._add_warp_vector(dx_map, dy_map, center_r, radius_r, strength, mode='expand')

    def _accumulate_face_contour(self, dx_map, dy_map, lm, strength):
        target_pt = lm[FaceMesh.FACE_INDICES['NOSE_TIP'][0]]
        
        # 턱선 포인트마다 벡터 누적
        for idx in FaceMesh.FACE_INDICES['JAW_L']:
            pt = lm[idx]
            # 반경을 좀 더 크게 잡아 부드럽게 (저해상도 기준)
            radius = int(np.linalg.norm(pt - target_pt) * 0.35) 
            vector = target_pt - pt
            self._add_warp_vector(dx_map, dy_map, pt, radius, strength * 0.3, mode='shrink', vector=vector)

        for idx in FaceMesh.FACE_INDICES['JAW_R']:
            pt = lm[idx]
            radius = int(np.linalg.norm(pt - target_pt) * 0.35)
            vector = target_pt - pt
            self._add_warp_vector(dx_map, dy_map, pt, radius, strength * 0.3, mode='shrink', vector=vector)

    def _add_warp_vector(self, dx_map, dy_map, center, radius, strength, mode='expand', vector=None):
        """
        [Core] 벡터 필드에 힘(Displacement)을 더하는 함수
        """
        cx, cy = int(center[0]), int(center[1])
        r = int(radius)
        if r <= 0: return

        h, w = dx_map.shape[:2]
        
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        
        if x1 >= x2 or y1 >= y2: return

        # 로컬 그리드
        grid_y, grid_x = np.indices((y2-y1, x2-x1), dtype=np.float32)
        
        # 원점 기준 좌표
        lcx, lcy = cx - x1, cy - y1
        local_dx = grid_x - lcx
        local_dy = grid_y - lcy
        dist_sq = local_dx**2 + local_dy**2
        
        mask = dist_sq < r*r
        if not np.any(mask): return
        
        dist = np.sqrt(dist_sq[mask])
        
        # 부드러운 감쇠 (Smooth Falloff)
        factor = (1.0 - dist / r) ** 2 * strength
        
        delta_x = np.zeros_like(local_dx)
        delta_y = np.zeros_like(local_dy)

        if mode == 'expand':
            delta_x[mask] -= local_dx[mask] * factor
            delta_y[mask] -= local_dy[mask] * factor
        elif mode == 'shrink':
            if vector is not None:
                vx, vy = vector
                v_len = np.sqrt(vx**2 + vy**2) + 1e-6
                vx, vy = vx/v_len, vy/v_len
                # 강도 계수 0.5
                delta_x[mask] -= vx * factor * r * 0.5
                delta_y[mask] -= vy * factor * r * 0.5
            else:
                delta_x[mask] += local_dx[mask] * factor
                delta_y[mask] += local_dy[mask] * factor

        # 누적 (Accumulate)
        dx_map[y1:y2, x1:x2] += delta_x
        dy_map[y1:y2, x1:x2] += delta_y