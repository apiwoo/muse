# Project MUSE - ai/parsing/skin_parser.py
# BiSeNet V2 Face Parsing TensorRT Wrapper
# Provides pixel-level skin mask detection for hybrid masking
# (C) 2025 MUSE Corp. All rights reserved.

"""
SkinParser: BiSeNet V2 based face parsing for precise skin detection.

This module uses a pretrained BiSeNet V2 model (CelebAMask-HQ) to detect
skin regions at pixel level, enabling hybrid masking with FaceMesh polygons.

CelebAMask-HQ Label Map:
    0: Background
    1: Skin (extracted)
    2: Nose
    3: Eye glasses
    4: Left eye
    5: Right eye
    6: Left eyebrow
    7: Right eyebrow
    8: Left ear
    9: Right ear
    10: Mouth (inner)
    11: Upper lip
    12: Lower lip
    13: Hair
    ...
"""

import os
import numpy as np

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    print("[WARNING] TensorRT not found. SkinParser will be disabled.")

try:
    import cupy as cp
    import cupyx.scipy.ndimage
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("[WARNING] CuPy not found. SkinParser will be disabled.")


class SkinParser:
    """
    BiSeNet V2 Face Parsing TensorRT Wrapper.

    Provides pixel-level skin mask detection for hybrid masking approach.
    Combines AI precision with FaceMesh polygon stability.

    Features:
    - TensorRT FP16 acceleration (~3-5ms inference)
    - CelebAMask-HQ pretrained weights
    - Extracts only skin class (label 1)
    - Auto-resize to original frame resolution
    - Graceful fallback if engine not found

    Usage:
        parser = SkinParser("assets/models/parsing/bisenet_v2_fp16.engine")
        skin_mask = parser.infer(frame_gpu)  # Returns CuPy uint8 mask (0-255)
    """

    # Model input size (BiSeNet V2 standard)
    INPUT_H = 512
    INPUT_W = 512

    # CelebAMask-HQ skin class index
    SKIN_CLASS = 1

    def __init__(self, engine_path, lazy=False):
        """
        Initialize SkinParser with TensorRT engine.

        Args:
            engine_path: Path to BiSeNet V2 TensorRT engine file
            lazy: If True, defer engine loading until first inference
        """
        self.engine_path = engine_path
        self.is_ready = False

        # TensorRT objects (initialized in load())
        self.logger = None
        self.engine = None
        self.context = None

        # Buffers
        self.d_input = None
        self.d_output = None
        self.input_name = None
        self.output_name = None

        # Preprocessing constants (ImageNet normalization)
        self.mean = None
        self.std = None

        if not lazy:
            self.load()

    def load(self):
        """Load TensorRT engine and allocate buffers."""
        if self.is_ready:
            return True

        if not HAS_TRT or not HAS_CUDA:
            print("[SKIN] TensorRT/CuPy not available. SkinParser disabled.")
            return False

        if not os.path.exists(self.engine_path):
            print(f"[SKIN] Engine not found: {self.engine_path}")
            return False

        try:
            print(f"[SKIN] Loading BiSeNet V2 Engine: {os.path.basename(self.engine_path)}...")

            self.logger = trt.Logger(trt.Logger.WARNING)

            with open(self.engine_path, "rb") as f:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())

            if self.engine is None:
                print("[SKIN] Failed to deserialize engine.")
                return False

            self.context = self.engine.create_execution_context()

            # Get tensor names
            self.input_name = self.engine.get_tensor_name(0)
            self.output_name = self.engine.get_tensor_name(1)

            # Allocate GPU buffers
            # Input: NCHW (1, 3, 512, 512)
            self.d_input = cp.zeros((1, 3, self.INPUT_H, self.INPUT_W), dtype=cp.float32)

            # Output: N x num_classes x H x W (BiSeNet outputs class logits)
            # Assuming 19 classes for CelebAMask-HQ
            num_classes = 19
            self.d_output = cp.zeros((1, num_classes, self.INPUT_H, self.INPUT_W), dtype=cp.float32)

            # Set tensor addresses (TensorRT 10.x API)
            self.context.set_tensor_address(self.input_name, int(self.d_input.data.ptr))
            self.context.set_tensor_address(self.output_name, int(self.d_output.data.ptr))

            # Preprocessing constants
            self.mean = cp.array([0.485, 0.456, 0.406], dtype=cp.float32).reshape(1, 3, 1, 1)
            self.std = cp.array([0.229, 0.224, 0.225], dtype=cp.float32).reshape(1, 3, 1, 1)

            self.is_ready = True
            print(f"[SKIN] BiSeNet V2 loaded successfully.")
            return True

        except Exception as e:
            print(f"[SKIN] Engine load failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _preprocess(self, frame_gpu):
        """
        Preprocess frame for BiSeNet V2 inference.

        Args:
            frame_gpu: CuPy BGR image (H, W, 3), uint8

        Returns:
            Preprocessed tensor (1, 3, 512, 512), float32
        """
        h, w = frame_gpu.shape[:2]

        # Resize to model input size using GPU
        zoom_h = self.INPUT_H / h
        zoom_w = self.INPUT_W / w

        frame_resized = cupyx.scipy.ndimage.zoom(
            frame_gpu.astype(cp.float32),
            (zoom_h, zoom_w, 1),
            order=1
        )

        # BGR -> RGB
        frame_rgb = frame_resized[:, :, ::-1]

        # Normalize to [0, 1]
        frame_norm = frame_rgb / 255.0

        # HWC -> NCHW
        frame_nchw = frame_norm.transpose(2, 0, 1).reshape(1, 3, self.INPUT_H, self.INPUT_W)

        # ImageNet normalization
        frame_nchw = (frame_nchw - self.mean) / self.std

        return frame_nchw.astype(cp.float32)

    def _postprocess(self, output, orig_h, orig_w):
        """
        Extract skin mask from BiSeNet output.

        Args:
            output: Model output (1, num_classes, 512, 512)
            orig_h, orig_w: Original frame dimensions

        Returns:
            Skin mask (H, W), uint8, values 0-255
        """
        # Argmax to get class predictions
        class_map = cp.argmax(output[0], axis=0)  # (512, 512)

        # Extract skin class (label 1)
        skin_mask_small = (class_map == self.SKIN_CLASS).astype(cp.float32)

        # Resize to original resolution
        zoom_h = orig_h / self.INPUT_H
        zoom_w = orig_w / self.INPUT_W

        skin_mask = cupyx.scipy.ndimage.zoom(
            skin_mask_small,
            (zoom_h, zoom_w),
            order=1  # Bilinear interpolation
        )

        # Handle potential size mismatch from zoom
        mh, mw = skin_mask.shape
        if mh != orig_h or mw != orig_w:
            skin_mask = skin_mask[:orig_h, :orig_w]

        # Convert to uint8 (0-255)
        skin_mask = (skin_mask * 255.0).clip(0, 255).astype(cp.uint8)

        return skin_mask

    def infer(self, frame_gpu):
        """
        Run inference on a frame to detect skin regions.

        Args:
            frame_gpu: CuPy BGR image (H, W, 3), uint8

        Returns:
            Skin mask (H, W), CuPy uint8, values 0-255
            Returns None if inference fails or engine not ready
        """
        if not self.is_ready:
            return None

        if frame_gpu is None:
            return None

        try:
            h, w = frame_gpu.shape[:2]

            # Preprocess
            input_tensor = self._preprocess(frame_gpu)

            # Copy to input buffer
            cp.copyto(self.d_input, input_tensor)

            # Set dynamic input shape if needed
            if self.input_name:
                self.context.set_input_shape(self.input_name, input_tensor.shape)

            # Run inference (TensorRT 10.x API)
            self.context.execute_async_v3(stream_handle=0)

            # Synchronize
            cp.cuda.Stream.null.synchronize()

            # Postprocess
            skin_mask = self._postprocess(self.d_output, h, w)

            return skin_mask

        except Exception as e:
            print(f"[SKIN] Inference failed: {e}")
            return None

    def infer_batch(self, frames_gpu):
        """
        Batch inference (for future optimization).
        Currently processes frames one by one.

        Args:
            frames_gpu: List of CuPy BGR images

        Returns:
            List of skin masks
        """
        return [self.infer(f) for f in frames_gpu]

    def get_skin_and_face_mask(self, frame_gpu):
        """
        Get both skin mask and full face region mask.
        Useful for advanced compositing.

        Args:
            frame_gpu: CuPy BGR image (H, W, 3), uint8

        Returns:
            Tuple of (skin_mask, face_mask) or (None, None) if failed
        """
        if not self.is_ready or frame_gpu is None:
            return None, None

        try:
            h, w = frame_gpu.shape[:2]

            # Preprocess and infer
            input_tensor = self._preprocess(frame_gpu)
            cp.copyto(self.d_input, input_tensor)

            if self.input_name:
                self.context.set_input_shape(self.input_name, input_tensor.shape)

            self.context.execute_async_v3(stream_handle=0)
            cp.cuda.Stream.null.synchronize()

            # Get class predictions
            class_map = cp.argmax(self.d_output[0], axis=0)

            # Skin mask (class 1)
            skin_small = (class_map == 1).astype(cp.float32)

            # Face mask (skin + nose + lips + eyes + eyebrows)
            # Classes: 1=skin, 2=nose, 4-5=eyes, 6-7=brows, 10-12=mouth/lips
            face_classes = [1, 2, 4, 5, 6, 7, 10, 11, 12]
            face_small = cp.zeros_like(class_map, dtype=cp.float32)
            for c in face_classes:
                face_small += (class_map == c).astype(cp.float32)
            face_small = cp.clip(face_small, 0, 1)

            # Resize both to original resolution
            zoom_h = h / self.INPUT_H
            zoom_w = w / self.INPUT_W

            skin_mask = cupyx.scipy.ndimage.zoom(skin_small, (zoom_h, zoom_w), order=1)
            face_mask = cupyx.scipy.ndimage.zoom(face_small, (zoom_h, zoom_w), order=1)

            # Handle size mismatch
            skin_mask = skin_mask[:h, :w]
            face_mask = face_mask[:h, :w]

            # Convert to uint8
            skin_mask = (skin_mask * 255.0).clip(0, 255).astype(cp.uint8)
            face_mask = (face_mask * 255.0).clip(0, 255).astype(cp.uint8)

            return skin_mask, face_mask

        except Exception as e:
            print(f"[SKIN] get_skin_and_face_mask failed: {e}")
            return None, None

    def release(self):
        """Release GPU resources."""
        self.d_input = None
        self.d_output = None
        self.context = None
        self.engine = None
        self.is_ready = False
        print("[SKIN] SkinParser resources released.")
