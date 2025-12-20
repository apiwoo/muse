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

    # Exclusion zones
    EYE_L_INDICES = FaceMesh.POLYGON_INDICES.get("EYE_L_POLY", [])
    EYE_R_INDICES = FaceMesh.POLYGON_INDICES.get("EYE_R_POLY", [])
    BROW_L_INDICES = FaceMesh.POLYGON_INDICES.get("BROW_L_POLY", [])
    BROW_R_INDICES = FaceMesh.POLYGON_INDICES.get("BROW_R_POLY", [])
    LIPS_INDICES = FaceMesh.POLYGON_INDICES.get("LIPS_OUTER_POLY", [])

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

        # 1. Fill face oval
        face_pts = landmarks[self.FACE_OVAL_INDICES].astype(np.int32)
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
