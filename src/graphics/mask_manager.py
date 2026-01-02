# Project MUSE - mask_manager.py
# Skin Mask Generation Component
# Extracted from beauty_engine.py for modular architecture
# (C) 2025 MUSE Corp. All rights reserved.

"""
Mask generation module for skin processing.

This module contains:
- MaskManager: CPU-based fast mask generation using OpenCV
"""

import cv2
import numpy as np
from ai.tracking.facemesh import FaceMesh

try:
    import cupy as cp
    import cupyx.scipy.ndimage
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


class MaskManager:
    """
    [V25] Fast skin mask manager (CPU OpenCV + GPU cache).

    Features:
    - Uses cv2.fillPoly for fast polygon rasterization
    - Precise exclusion of eyes, eyebrows, lips (optional)
    - Caches GPU mask to avoid repeated transfers
    - Soft edge via Gaussian blur for natural blending
    """

    # FaceMesh indices (static reference)
    FACE_OVAL_INDICES = FaceMesh.FACE_INDICES.get("FACE_OVAL", [])
    FOREHEAD_INDICES = FaceMesh.FACE_INDICES.get("FOREHEAD", [])

    # [V50] 이마 상단 인덱스 (FACE_OVAL 배열 내 위치)
    # FACE_OVAL 총 36개 포인트 중 이마 상단 라인을 형성하는 9개 인덱스
    # 0-4: 우측 이마 (10, 338, 297, 332, 284)
    # 32-35: 좌측 이마 (54, 103, 67, 109)
    FOREHEAD_TOP_INDICES_IN_OVAL = [0, 1, 2, 3, 4, 32, 33, 34, 35]

    # Exclusion zones
    EYE_L_INDICES = FaceMesh.POLYGON_INDICES.get("EYE_L_POLY", [])
    EYE_R_INDICES = FaceMesh.POLYGON_INDICES.get("EYE_R_POLY", [])
    BROW_L_INDICES = FaceMesh.POLYGON_INDICES.get("BROW_L_POLY", [])
    BROW_R_INDICES = FaceMesh.POLYGON_INDICES.get("BROW_R_POLY", [])
    LIPS_INDICES = FaceMesh.POLYGON_INDICES.get("LIPS_OUTER_POLY", [])
    LIPS_INNER_INDICES = FaceMesh.POLYGON_INDICES.get("LIPS_INNER_POLY", [])

    def __init__(self):
        """Initialize mask manager with empty cache."""
        self.cache_w = 0
        self.cache_h = 0
        self.mask_cpu = None
        self.mask_gpu = None

    def generate_mask(self, landmarks, w, h, padding_ratio=1.15, exclude_features=False):
        """
        Generate skin mask using OpenCV (CPU, very fast < 2ms).

        Args:
            landmarks: Face landmarks array (numpy)
            w, h: Image dimensions
            padding_ratio: Padding for exclusion zones
            exclude_features: If False, no exclusion zones (YY-style)
                            If True, exclude eyes/brows/lips (legacy)
        Returns:
            Mask as CuPy array (GPU) if available, else numpy array
        """
        # Reuse buffer if size matches
        if self.mask_cpu is None or self.cache_w != w or self.cache_h != h:
            self.mask_cpu = np.zeros((h, w), dtype=np.uint8)
            self.cache_w = w
            self.cache_h = h
        else:
            self.mask_cpu.fill(0)

        mask = self.mask_cpu

        # 1. Fill face oval with forehead extension
        # [V50] 이마 영역 확장 - FaceMesh 한계 보완
        face_pts = landmarks[self.FACE_OVAL_INDICES].astype(np.float32).copy()

        # 얼굴 높이 계산 (이마 상단 ~ 턱 끝)
        forehead_pt = landmarks[10]  # 이마 중앙
        chin_pt = landmarks[152]     # 턱 끝
        face_height = chin_pt[1] - forehead_pt[1]

        if face_height > 1:
            # 이마 확장 비율 (얼굴 높이의 25%)
            forehead_extension_ratio = 0.25
            offset_y = -face_height * forehead_extension_ratio

            # 이마 상단 인덱스들에만 오프셋 적용
            for idx in self.FOREHEAD_TOP_INDICES_IN_OVAL:
                if idx < len(face_pts):
                    face_pts[idx, 1] += offset_y
                    # 화면 밖으로 나가지 않도록 클리핑
                    face_pts[idx, 1] = max(0, face_pts[idx, 1])

        face_pts = face_pts.astype(np.int32)
        cv2.fillPoly(mask, [face_pts], 255)

        # 2-4. Exclude features only if requested (legacy mode)
        # YY-Style: Bilateral filter handles edge preservation automatically
        if exclude_features:
            self._exclude_region(mask, landmarks, self.EYE_L_INDICES, padding_ratio * 1.3)
            self._exclude_region(mask, landmarks, self.EYE_R_INDICES, padding_ratio * 1.3)
            self._exclude_region(mask, landmarks, self.BROW_L_INDICES, padding_ratio * 1.1)
            self._exclude_region(mask, landmarks, self.BROW_R_INDICES, padding_ratio * 1.1)
            self._exclude_region(mask, landmarks, self.LIPS_INDICES, padding_ratio * 1.2)

        # 5. Soft edge - increased sigma for smoother blending
        mask = cv2.GaussianBlur(mask, (35, 35), 17)
        self.mask_cpu = mask

        # Transfer to GPU
        if HAS_CUDA:
            self.mask_gpu = cp.asarray(mask)
            return self.mask_gpu
        return mask

    def _exclude_region(self, mask, landmarks, indices, padding=1.0):
        """
        Exclude a polygon region from the mask with padding.

        Args:
            mask: Mask array to modify in-place
            landmarks: Full landmarks array
            indices: Indices of landmarks for this region
            padding: Scale factor for region (>1 = larger exclusion)
        """
        if len(indices) == 0:
            return

        pts = landmarks[indices].astype(np.float32)

        if padding != 1.0:
            center = np.mean(pts, axis=0)
            pts = center + (pts - center) * padding

        pts = pts.astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 0)

    def get_mask_cpu(self):
        """Get mask as numpy array."""
        return self.mask_cpu

    def get_mask_gpu(self):
        """Get mask as CuPy array (or None if not available)."""
        return self.mask_gpu

    def generate_hybrid_mask(self, landmarks, ai_skin_mask, w, h, padding_ratio=1.25):
        """
        Generate hybrid mask combining FaceMesh polygon and AI skin parsing.

        This method combines the stability of FaceMesh polygon-based masking
        with the pixel-level precision of AI skin parsing.

        Logic:
            1. Generate FaceMesh-based face region mask (bounding stability)
            2. Intersect with AI skin mask if available (pixel precision)
            3. Apply Gaussian blur for smooth edges

        Benefits:
            - AI mask: Pixel-level precision (prevents hair intrusion)
            - FaceMesh mask: Face region bounding (stability/fallback)
            - Intersection: Combines advantages of both approaches

        Args:
            landmarks: Face landmarks array (numpy)
            ai_skin_mask: AI-detected skin mask (CuPy uint8, 0-255) or None
            w, h: Image dimensions
            padding_ratio: Padding for face region (default 1.25)

        Returns:
            Hybrid mask as CuPy array (GPU) if available, else numpy array
        """
        # Step 1: Generate FaceMesh-based face region mask
        # Use slightly larger padding to ensure full coverage
        face_region = self.generate_mask(
            landmarks, w, h,
            padding_ratio=padding_ratio * 1.1,  # Slightly larger for hybrid
            exclude_features=False  # Don't exclude features (AI mask handles this)
        )

        if face_region is None:
            return ai_skin_mask if ai_skin_mask is not None else None

        # Step 2: Combine with AI mask if available
        if ai_skin_mask is not None and HAS_CUDA:
            # Ensure both are on GPU
            if not hasattr(face_region, 'device'):
                face_region = cp.asarray(face_region)

            if not hasattr(ai_skin_mask, 'device'):
                ai_skin_mask = cp.asarray(ai_skin_mask)

            # Ensure same shape
            if ai_skin_mask.shape != face_region.shape:
                # Resize AI mask to match if needed
                ai_skin_mask = cupyx.scipy.ndimage.zoom(
                    ai_skin_mask.astype(cp.float32),
                    (h / ai_skin_mask.shape[0], w / ai_skin_mask.shape[1]),
                    order=1
                ).astype(cp.uint8)
                ai_skin_mask = ai_skin_mask[:h, :w]

            # Intersection: Take minimum of both masks
            # This ensures we only process areas that both agree are skin
            hybrid_mask = cp.minimum(face_region, ai_skin_mask)

            # Step 3: Smooth the edges with Gaussian blur
            hybrid_mask = cupyx.scipy.ndimage.gaussian_filter(
                hybrid_mask.astype(cp.float32),
                sigma=3.0
            )
            hybrid_mask = cp.clip(hybrid_mask, 0, 255).astype(cp.uint8)

            # Debug: Log hybrid mask stats (throttled)
            if not hasattr(self, '_hybrid_log_count'):
                self._hybrid_log_count = 0
            self._hybrid_log_count += 1
            if self._hybrid_log_count % 60 == 1:
                face_mean = float(cp.mean(face_region))
                ai_mean = float(cp.mean(ai_skin_mask))
                hybrid_mean = float(cp.mean(hybrid_mask))
                print(f"[HYBRID-DEBUG] FaceMesh={face_mean:.1f}, AI={ai_mean:.1f}, Hybrid={hybrid_mean:.1f} (intersection)")

            return hybrid_mask
        else:
            # Fallback: Use FaceMesh mask only
            if not hasattr(self, '_fallback_log_count'):
                self._fallback_log_count = 0
            self._fallback_log_count += 1
            if self._fallback_log_count % 300 == 1:
                print(f"[HYBRID-DEBUG] Fallback to FaceMesh-only (AI mask unavailable)")
            return face_region

    def generate_torso_mask(self, body_landmarks, w, h):
        """
        [V40] Skeleton Patch - 몸통 영역 마스크 생성

        ViTPose 랜드마크의 어깨-골반을 연결한 사각형 영역을 채워서
        AI 마스크의 내부 구멍을 메우는 데 사용.

        Args:
            body_landmarks: ViTPose 결과 [(x, y, conf), ...] 또는 [(x, y), ...]
            w, h: 이미지 크기

        Returns:
            torso_mask: 몸통 영역 마스크 (np.ndarray, uint8) 또는 None
        """
        if body_landmarks is None:
            return None

        # body_landmarks가 .get()을 가진 객체인 경우 처리
        if hasattr(body_landmarks, 'get'):
            body_landmarks = body_landmarks.get()

        if len(body_landmarks) < 13:
            return None

        # 랜드마크 인덱스: 5=왼쪽어깨, 6=오른쪽어깨, 11=왼쪽골반, 12=오른쪽골반
        indices = [5, 6, 12, 11]  # 시계방향 사각형
        points = []

        for idx in indices:
            if idx >= len(body_landmarks):
                return None

            pt = body_landmarks[idx]

            # confidence 체크 (3번째 값이 있는 경우)
            if len(pt) >= 3:
                if pt[2] < 0.3:  # 너무 낮으면 스킵
                    return None

            points.append([int(pt[0]), int(pt[1])])

        # 빈 마스크 생성
        mask = np.zeros((h, w), dtype=np.uint8)

        # 폴리곤 채우기
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        # 경계 부드럽게 (가우시안 블러)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        return mask

    def generate_teeth_mask(self, landmarks, w, h):
        """
        [치아 미백] 입술 안쪽 폴리곤 기반 치아 영역 마스크 생성

        LIPS_INNER_POLY는 입이 벌어졌을 때 치아가 보이는 영역과 일치.
        이 마스크는 CUDA 커널에서 밝기/색상 조건과 함께 사용되어
        입술(붉은색)과 혀(어두움)를 자동 제외함.

        Args:
            landmarks: Face landmarks array (numpy)
            w, h: Image dimensions

        Returns:
            teeth_mask: 치아 영역 마스크 (CuPy GPU array 또는 numpy array)
                       None if landmarks/indices unavailable
        """
        if landmarks is None:
            return None

        if len(self.LIPS_INNER_INDICES) == 0:
            return None

        # 빈 마스크 생성
        mask = np.zeros((h, w), dtype=np.uint8)

        # 입술 안쪽 폴리곤 좌표 추출
        pts = landmarks[self.LIPS_INNER_INDICES].astype(np.int32)

        # 폴리곤 채우기
        cv2.fillPoly(mask, [pts], 255)

        # 경계 부드럽게 (가우시안 블러)
        mask = cv2.GaussianBlur(mask, (7, 7), 3)

        # GPU 변환
        if HAS_CUDA:
            return cp.asarray(mask)
        return mask
