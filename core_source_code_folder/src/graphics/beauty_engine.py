# Project MUSE - beauty_engine.py
# Created for Mode A (Visual Supremacy)
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import mediapipe as mp
# [Fix] main.py의 sys.path 설정에 맞춰 'src.' 제거
from ai.tracking.facemesh import FaceMesh

class BeautyEngine:
    def __init__(self):
        """
        [Mode A] Real-time Beauty Engine
        - 역할: 얼굴/몸 랜드마크를 기반으로 이미지 왜곡(Warping) 수행
        - V1.3 Update: Background Protection (Segmentation)
        """
        print("💄 [BeautyEngine] 성형 엔진 초기화 (V1.3 - Background Protection)")
        
        # [New] 배경 보호를 위한 세그멘테이션 모델 로드
        self.mp_seg = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_seg.SelfieSegmentation(model_selection=1) # 1: Landscape mode (더 정확함)

    def process(self, frame, faces, body_landmarks=None, params=None):
        """
        메인 처리 함수
        :param frame: 입력 이미지 (BGR)
        :param faces: FaceMesh 객체 리스트
        :param body_landmarks: (17, 3) NumPy Array [x, y, conf] (from ViTPose)
        :param params: 성형 파라미터 (dict)
        """
        if frame is None:
            return frame

        if params is None:
            params = {}

        # 원본 보존 (배경 복원용)
        original_bg = frame.copy()
        
        # 워핑을 적용할 작업용 이미지
        warped_frame = frame.copy()

        # [Step 1] Body Reshaping (ViTPose COCO Format)
        if body_landmarks is not None:
            # 허리 축소
            if params.get('waist_slim', 0) > 0:
                warped_frame = self._warp_waist(warped_frame, body_landmarks, strength=params['waist_slim'])

        # [Step 2] Face Reshaping
        if faces:
            for face in faces:
                lm = self._get_landmarks(face)
                if lm is None: continue

                # V라인 (턱 깎기)
                if params.get('face_v', 0) > 0:
                    warped_frame = self._warp_face_contour(warped_frame, lm, strength=params['face_v'])

                # 왕눈이 (눈 키우기)
                if params.get('eye_scale', 0) > 0:
                    warped_frame = self._warp_eyes(warped_frame, lm, strength=params['eye_scale'])

        # [Step 3] Background Protection (배경 왜곡 방지)
        # 워핑된 결과물에서 '사람'만 오려내어, 원본 배경(original_bg) 위에 덮어씌웁니다.
        # 이렇게 하면 배경이 휘어지는 현상이 사라집니다.
        
        # 1. 마스크 추출 (MediaPipe는 RGB를 원함)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(frame_rgb)
        
        if results.segmentation_mask is not None:
            # 마스크 처리 (0~1 사이 값)
            mask = results.segmentation_mask
            
            # 경계선 부드럽게 (Anti-aliasing)
            # Bilateral Filter 등을 쓰면 더 좋지만 속도를 위해 GaussianBlur 사용
            mask = cv2.GaussianBlur(mask, (0, 0), 2) # Sigma=2
            mask = np.stack((mask,) * 3, axis=-1) # 3채널로 확장

            # [핵심] 합성: (사람_워핑 * 마스크) + (원본_배경 * (1-마스크))
            # 마스크가 1에 가까울수록(사람) 워핑된 이미지를 쓰고,
            # 마스크가 0에 가까울수록(배경) 원본 배경을 씁니다.
            
            # 주의: 워핑된 이미지(warped_frame)의 마스크 위치가 원본과 약간 다를 수 있지만(몸을 줄였으니까),
            # 줄어든 만큼은 배경이 보여야 하므로 원본 마스크를 쓰는게 맞습니다.
            # (엄밀히는 워핑된 마스크를 써야 하지만, 축소 워핑이라 원본 마스크가 더 큽니다.
            #  따라서 줄어든 틈새로 원본 배경이 보이게 됩니다 -> 자연스러운 Inpainting 효과)
            
            final_output = (warped_frame * mask + original_bg * (1.0 - mask)).astype(np.uint8)
            return final_output
        else:
            # 마스크 추출 실패 시 그냥 워핑된거 반환
            return warped_frame

    def _get_landmarks(self, face):
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            return face.landmark_2d_106.astype(int)
        return None

    # ==========================================================
    # [Body Warping Logic] - Updated for ViTPose (COCO)
    # ==========================================================
    def _warp_waist(self, img, keypoints, strength):
        """
        허리 잘록하게 만들기 (Waist Slimming)
        """
        # 신뢰도 임계값
        CONF_THRESH = 0.4

        # COCO Indices
        idx_l_sh, idx_r_sh = 5, 6
        idx_l_hip, idx_r_hip = 11, 12
        
        try:
            # 신뢰도 체크
            if (keypoints[idx_l_sh, 2] < CONF_THRESH or keypoints[idx_r_sh, 2] < CONF_THRESH or
                keypoints[idx_l_hip, 2] < CONF_THRESH or keypoints[idx_r_hip, 2] < CONF_THRESH):
                return img

            # 좌표 추출
            l_sh = keypoints[idx_l_sh, :2]
            r_sh = keypoints[idx_r_sh, :2]
            l_hip = keypoints[idx_l_hip, :2]
            r_hip = keypoints[idx_r_hip, :2]
            
        except IndexError:
            return img

        # 허리 위치 추정 (어깨와 힙 사이, 힙 쪽에 가깝게)
        l_waist = l_sh * 0.4 + l_hip * 0.6
        r_waist = r_sh * 0.4 + r_hip * 0.6
        
        # 몸통 중심선
        center_waist = (l_waist + r_waist) / 2
        
        # 워핑 반경
        body_width = np.linalg.norm(l_waist - r_waist)
        if body_width < 10: return img
        
        radius = int(body_width * 0.7) # 반경을 조금 더 넓게
        
        # 강도 조절
        warp_strength = strength * 0.5 # 0.0 ~ 0.5

        # 왼쪽 허리 당기기 (중심 쪽으로)
        vec_l = center_waist - l_waist
        img = self._apply_local_warp(img, l_waist, radius, warp_strength, mode='shrink', vector=vec_l)

        # 오른쪽 허리 당기기 (중심 쪽으로)
        vec_r = center_waist - r_waist
        img = self._apply_local_warp(img, r_waist, radius, warp_strength, mode='shrink', vector=vec_r)

        return img

    # ==========================================================
    # [Face Warping Logic] (기존 유지)
    # ==========================================================
    def _warp_eyes(self, img, lm, strength):
        indices_l = FaceMesh.FACE_INDICES['EYE_L']
        indices_r = FaceMesh.FACE_INDICES['EYE_R']

        pts_l = lm[indices_l]
        center_l = np.mean(pts_l, axis=0).astype(int)
        eye_width_l = np.linalg.norm(pts_l[np.argmax(pts_l[:,0])] - pts_l[np.argmin(pts_l[:,0])])
        radius_l = int(eye_width_l * 1.8)

        pts_r = lm[indices_r]
        center_r = np.mean(pts_r, axis=0).astype(int)
        eye_width_r = np.linalg.norm(pts_r[np.argmax(pts_r[:,0])] - pts_r[np.argmin(pts_r[:,0])])
        radius_r = int(eye_width_r * 1.8)

        img = self._apply_local_warp(img, center_l, radius_l, strength, mode='expand')
        img = self._apply_local_warp(img, center_r, radius_r, strength, mode='expand')
        return img

    def _warp_face_contour(self, img, lm, strength):
        target_pt = lm[86]
        left_jaw_indices = [14, 15, 16, 5, 6, 7] 
        for idx in left_jaw_indices:
            pt = lm[idx]
            radius = int(np.linalg.norm(pt - lm[0]) * 0.4) 
            vector = target_pt - pt
            img = self._apply_local_warp(img, pt, radius, strength * 0.3, mode='shrink', vector=vector)

        right_jaw_indices = [30, 31, 32, 21, 22, 23]
        for idx in right_jaw_indices:
            pt = lm[idx]
            radius = int(np.linalg.norm(pt - lm[0]) * 0.4)
            vector = target_pt - pt
            img = self._apply_local_warp(img, pt, radius, strength * 0.3, mode='shrink', vector=vector)

        return img

    def _apply_local_warp(self, img, center, radius, strength, mode='expand', vector=None):
        """
        [Core Algorithm] 국소 영역 워핑
        """
        cx, cy = int(center[0]), int(center[1])
        r = int(radius)
        
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(img.shape[1], cx + r), min(img.shape[0], cy + r)
        
        roi = img[y1:y2, x1:x2]
        if roi.size == 0: return img

        h, w = roi.shape[:2]
        grid_y, grid_x = np.indices((h, w), dtype=np.float32)
        
        lcx, lcy = cx - x1, cy - y1
        dx = grid_x - lcx
        dy = grid_y - lcy
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq)
        
        mask = dist < r
        factor = np.zeros_like(dist)
        with np.errstate(divide='ignore', invalid='ignore'):
             factor[mask] = (1.0 - dist[mask] / r) ** 2 * strength

        map_x = grid_x.copy()
        map_y = grid_y.copy()

        if mode == 'expand':
            map_x[mask] -= dx[mask] * factor[mask]
            map_y[mask] -= dy[mask] * factor[mask]
            
        elif mode == 'shrink':
            if vector is not None:
                vx, vy = vector
                v_len = np.sqrt(vx*vx + vy*vy)
                if v_len > 0:
                    vx, vy = vx/v_len, vy/v_len
                    map_x[mask] -= vx * factor[mask] * r * 0.5
                    map_y[mask] -= vy * factor[mask] * r * 0.5
            else:
                map_x[mask] += dx[mask] * factor[mask]
                map_y[mask] += dy[mask] * factor[mask]

        warped_roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
        
        # 자연스러운 합성을 위해 Gaussian Blur 마스크 사용
        mask_img = np.zeros((h, w), dtype=np.float32)
        mask_img[mask] = 1.0
        mask_img = cv2.GaussianBlur(mask_img, (5, 5), 0)
        
        mask_3ch = mask_img[..., np.newaxis]
        img[y1:y2, x1:x2] = (warped_roi * mask_3ch + roi * (1.0 - mask_3ch)).astype(np.uint8)
        
        return img