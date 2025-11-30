# Project MUSE - beauty_engine.py
# Created for Mode A (Visual Supremacy)
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
# [Fix] main.py의 sys.path 설정에 맞춰 'src.' 제거
from ai.tracking.facemesh import FaceMesh

class BeautyEngine:
    def __init__(self):
        """
        [Mode A] Real-time Beauty Engine
        - 역할: 얼굴 랜드마크를 기반으로 이미지 왜곡(Warping) 수행
        - 초기 버전: OpenCV CPU/NumPy 기반 (추후 Shader 포팅 예정)
        """
        print("💄 [BeautyEngine] 성형 엔진 초기화 (V1.0 - OpenCV Backend)")
        pass

    def process(self, frame, faces, params=None):
        """
        메인 처리 함수
        :param frame: 입력 이미지 (BGR)
        :param faces: FaceMesh에서 감지된 얼굴 객체 리스트
        :param params: 성형 파라미터 (예: {'eye_scale': 0.2, 'face_v': 0.1})
        """
        if frame is None or not faces:
            return frame

        # 기본 파라미터 설정 (값이 없으면 기본값 적용)
        if params is None:
            # 테스트용 기본값: 눈 25% 확대, 턱 15% 깎기
            params = {'eye_scale': 0.25, 'face_v': 0.15}

        result = frame.copy()

        for face in faces:
            # 1. 랜드마크 좌표 가져오기 (필수)
            lm = self._get_landmarks(face)
            if lm is None:
                continue

            # 2. 기능별 워핑 적용
            # 순서 중요: 턱을 먼저 깎고 눈을 키우는 게 보통 자연스러움
            
            # [기능 1] V라인 (턱 깎기)
            if params.get('face_v', 0) > 0:
                result = self._warp_face_contour(result, lm, strength=params['face_v'])

            # [기능 2] 왕눈이 (눈 키우기)
            if params.get('eye_scale', 0) > 0:
                result = self._warp_eyes(result, lm, strength=params['eye_scale'])

        return result

    def _get_landmarks(self, face):
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            return face.landmark_2d_106.astype(int)
        return None

    def _warp_eyes(self, img, lm, strength):
        """
        눈 키우기 (Local Scaling)
        - Update: 새로운 인덱스(33~42, 87~96) 적용
        """
        # FaceMesh에서 정의한 눈 인덱스 사용
        indices_l = FaceMesh.FACE_INDICES['EYE_L']
        indices_r = FaceMesh.FACE_INDICES['EYE_R']

        # 왼쪽 눈 중심 및 반지름 계산
        pts_l = lm[indices_l]
        center_l = np.mean(pts_l, axis=0).astype(int)
        # 눈 크기 추정 (좌우 폭의 1.8배)
        eye_width_l = np.linalg.norm(pts_l[np.argmax(pts_l[:,0])] - pts_l[np.argmin(pts_l[:,0])])
        radius_l = int(eye_width_l * 1.8)

        # 오른쪽 눈 중심 및 반지름 계산
        pts_r = lm[indices_r]
        center_r = np.mean(pts_r, axis=0).astype(int)
        eye_width_r = np.linalg.norm(pts_r[np.argmax(pts_r[:,0])] - pts_r[np.argmin(pts_r[:,0])])
        radius_r = int(eye_width_r * 1.8)

        # 워핑 적용
        img = self._apply_local_warp(img, center_l, radius_l, strength, mode='expand')
        img = self._apply_local_warp(img, center_r, radius_r, strength, mode='expand')
        
        return img

    def _warp_face_contour(self, img, lm, strength):
        """
        턱 깎기 (V-Line)
        - Update: 복잡한 턱 라인 인덱스(JAW_L, JAW_R) 대응
        """
        # 당기는 목표점: 코 끝(86번)
        target_pt = lm[86]

        # 1. 왼쪽 턱 깎기 (JAW_L)
        # 턱 라인 중 사각턱 부위(귀 밑 ~ 턱 중간)를 타겟팅
        # JAW_L 리스트: [1(관자), 9..16(외곽), 2..8(턱선), 0(턱끝)]
        # 이 중에서 12~16번(외곽 하단)과 4~8번(턱선)이 깎아야 할 주요 부위
        left_jaw_indices = [14, 15, 16, 5, 6, 7] 
        
        for idx in left_jaw_indices:
            pt = lm[idx]
            # 영향 범위: 턱의 크기에 비례
            radius = int(np.linalg.norm(pt - lm[0]) * 0.4) 
            # 코 끝 방향으로 당김 (Shrink)
            vector = target_pt - pt
            img = self._apply_local_warp(img, pt, radius, strength * 0.3, mode='shrink', vector=vector)

        # 2. 오른쪽 턱 깎기 (JAW_R)
        # JAW_R 리스트: [17(관자), 25..32(외곽), 18..24(턱선), 0(턱끝)]
        # 대칭되는 인덱스: 28~32(외곽 하단), 20~24(턱선)
        right_jaw_indices = [30, 31, 32, 21, 22, 23]

        for idx in right_jaw_indices:
            pt = lm[idx]
            radius = int(np.linalg.norm(pt - lm[0]) * 0.4)
            vector = target_pt - pt
            img = self._apply_local_warp(img, pt, radius, strength * 0.3, mode='shrink', vector=vector)

        return img

    def _apply_local_warp(self, img, center, radius, strength, mode='expand', vector=None):
        """
        [Core Algorithm] 국소 영역 워핑 (최적화 버전)
        - 전체 이미지를 Remap하지 않고 ROI만 잘라서 처리함
        """
        cx, cy = center
        r = int(radius)
        
        # 1. ROI 추출 (이미지 범위를 벗어나지 않게)
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(img.shape[1], cx + r), min(img.shape[0], cy + r)
        
        roi = img[y1:y2, x1:x2]
        if roi.size == 0: return img

        h, w = roi.shape[:2]
        
        # 2. 매핑 그리드 생성
        grid_y, grid_x = np.indices((h, w), dtype=np.float32)
        
        # 로컬 중심 좌표
        lcx, lcy = cx - x1, cy - y1
        
        # 3. 변위 계산
        dx = grid_x - lcx
        dy = grid_y - lcy
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq)
        
        # 마스크: 반지름 내부만 적용
        mask = dist < r
        
        # 워핑 팩터 (중심에서 멀어질수록 약해짐)
        # (1 - d/r)^2 커브 사용
        factor = np.zeros_like(dist)
        # 0으로 나누기 방지 및 마스크 적용
        with np.errstate(divide='ignore', invalid='ignore'):
             factor[mask] = (1.0 - dist[mask] / r) ** 2 * strength

        # 4. 좌표 이동 (Remap Map 생성)
        map_x = grid_x.copy()
        map_y = grid_y.copy()

        if mode == 'expand':
            # 확대: 픽셀을 중심 안쪽에서 가져옴 (Pull)
            # 현재 위치(x)에 (x - dx*factor) 위치의 색상을 칠함
            map_x[mask] -= dx[mask] * factor[mask]
            map_y[mask] -= dy[mask] * factor[mask]
            
        elif mode == 'shrink':
            # 축소/이동: 픽셀을 바깥쪽/벡터 반대에서 가져옴 (Push)
            if vector is not None:
                # 특정 벡터 방향으로 밀기
                vx, vy = vector
                # 정규화
                v_len = np.sqrt(vx*vx + vy*vy)
                if v_len > 0:
                    vx, vy = vx/v_len, vy/v_len
                    map_x[mask] -= vx * factor[mask] * r * 0.5 # 스케일 보정
                    map_y[mask] -= vy * factor[mask] * r * 0.5
            else:
                # 단순 축소
                map_x[mask] += dx[mask] * factor[mask]
                map_y[mask] += dy[mask] * factor[mask]

        # 5. Remap 적용 (Bilinear Interpolation)
        warped_roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
        
        # 6. 자연스러운 합성 (Alpha Blending)
        # 경계선이 튀지 않도록 마스크를 부드럽게 처리
        mask_img = np.zeros((h, w), dtype=np.float32)
        mask_img[mask] = 1.0
        mask_img = cv2.GaussianBlur(mask_img, (5, 5), 0) # Smooth Edge
        
        # ROI 덮어쓰기 (img = warped * alpha + original * (1-alpha))
        mask_3ch = mask_img[..., np.newaxis]
        img[y1:y2, x1:x2] = (warped_roi * mask_3ch + roi * (1.0 - mask_3ch)).astype(np.uint8)
        
        return img