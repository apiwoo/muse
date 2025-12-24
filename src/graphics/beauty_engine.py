# Project MUSE - beauty_engine.py
# V47: 배경 왜곡 차단 & 번개 현상 방어
# - 신규: Erosion 기반 마스크 수축 (iterations=2, sigma=1.5)
# - 신규: 커널 임계값 조정 (CERTAIN_PERSON 0.3→0.10, CERTAIN_BG 0.05→0.02)
# - 유지: bypass=True (Stabilizer 비활성화)
# - 유지: FrameSyncBuffer 비활성화
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import os
from ai.tracking.facemesh import FaceMesh

# Import modular components
from graphics.stabilizers import LandmarkStabilizer, MaskStabilizer, WarpGridStabilizer
from graphics.buffers import FrameSyncBuffer
from graphics.mask_manager import MaskManager

# Import AI Skin Parser (optional - graceful fallback if not available)
try:
    from ai.parsing.skin_parser import SkinParser
    HAS_SKIN_PARSER = True
except ImportError:
    SkinParser = None
    HAS_SKIN_PARSER = False

# Import Kernels & Logic
from graphics.kernels.cuda_kernels import (
    WARP_KERNEL_CODE, COMPOSITE_KERNEL_CODE, SKIN_SMOOTH_KERNEL_CODE,
    BILATERAL_SMOOTH_KERNEL_CODE, GPU_RESIZE_KERNEL_CODE,
    GPU_MASK_RESIZE_KERNEL_CODE, FINAL_BLEND_KERNEL_CODE,
    # V4: Forward Mask based Composite
    FORWARD_WARP_MASK_KERNEL_CODE, MASK_DILATE_KERNEL_CODE, SIMPLE_COMPOSITE_KERNEL_CODE,
    # V5: Void Fill Composite (동기화된 트리플 레이어)
    VOID_FILL_COMPOSITE_KERNEL_CODE,
    # V29: High-Fidelity Skin Smoothing (Guided Filter + LAB Color Space)
    GUIDED_FILTER_KERNEL_CODE, FAST_SKIN_SMOOTH_KERNEL_CODE, LAB_SKIN_SMOOTH_KERNEL_CODE,
    # V31: Dual-Pass Smooth Kernel (Wide/Fine 합성)
    DUAL_PASS_SMOOTH_KERNEL_CODE,
    # V34: Background Warp Prevention & Clean 2-Layer Composite
    MODULATE_DISPLACEMENT_KERNEL_CODE, LAYERED_COMPOSITE_KERNEL_CODE,
    # V36: Warp Grid Based Mask (근본적 재설계)
    WARP_MASK_FROM_GRID_KERNEL_CODE, CLEAN_COMPOSITE_KERNEL_CODE,
    # V40: Skeleton Patch (AI Mask + Torso Mask)
    MASK_COMBINE_KERNEL_CODE,
    # V41: Logical Void Fill (Time-Locked Sync)
    LOGICAL_VOID_FILL_KERNEL_CODE,
    # V43: Inverse Warp Validity (번개 현상 근본 해결)
    INVERSE_WARP_VALIDITY_KERNEL_CODE,
    # V44: Simple Void Fill (Frame-Independent)
    SIMPLE_VOID_FILL_KERNEL_CODE
)
from graphics.processors.morph_logic import MorphLogic

try:
    import cupy as cp
    import cupyx.scipy.ndimage
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("[WARNING] [BeautyEngine] CuPy not found. Fallback to CPU Mode.")


# ==============================================================================
# [메인 클래스] BeautyEngine - V47 배경 왜곡 차단 & 번개 현상 방어
# ==============================================================================
class BeautyEngine:
    """
    V47 Beauty Processing Engine - 배경 왜곡 차단 & 번개 현상 방어

    Key Changes (V47):
    - Erosion 기반 배경 왜곡 차단:
      * binary_erosion(iterations=2)로 마스크 2픽셀 수축
      * gaussian_filter(sigma=1.5)로 경계 부드럽게
      * 배경 픽셀에 워핑 변위 0 적용

    - 커널 임계값 조정 (번개 현상 방어):
      * CERTAIN_PERSON: 0.3 → 0.10 (10%만 넘어도 사람 판정)
      * CERTAIN_BG: 0.05 → 0.02 (진짜 완전한 배경만)
      * ORIGIN_THRESHOLD: 0.2 → 0.10

    Preserved from V46.2:
    - 화면 밖 keypoint 검증 (margin=0, min_conf=0.3)

    Preserved from V44-V45.1:
    - Stabilizer bypass=True (프레임 독립 처리)
    - FrameSyncBuffer 비활성화
    - Simple Void Fill Kernel

    Result: 배경 휘어짐 없음 + 번개 현상 없음
    """

    def __init__(self, profiles=[]):
        print("[BEAUTY] [BeautyEngine] V47 Erosion + Threshold Defense")
        self.map_scale = 0.25
        self.cache_w = 0
        self.cache_h = 0
        self.gpu_initialized = False

        self.gpu_dx = None
        self.gpu_dy = None

        # [V44] 프레임 독립 처리 - 모든 Stabilizer bypass
        self.body_stabilizer = LandmarkStabilizer(
            min_cutoff=0.005,
            base_beta=0.5,
            high_speed_beta=50.0,
            bypass=True  # [V44] Frame-independent mode
        )
        self.face_stabilizer = LandmarkStabilizer(
            min_cutoff=0.1,
            base_beta=3.0,
            high_speed_beta=30.0,
            bypass=True  # [V44] Frame-independent mode
        )

        # V28.0: Mask Stabilizer (마스크-워핑 시간 동기화)
        self.mask_stabilizer = None  # Initialized when HAS_CUDA

        # V33: Warp Grid Stabilizer with Adaptive EMA (동적 alpha)
        self.warp_grid_stabilizer = None  # Initialized when HAS_CUDA

        # V33: Frame Sync Buffer (AI 지연 보상)
        self.frame_sync_buffer = None  # Initialized when HAS_CUDA
        # [V45] AI 마스크 지연 프레임 수
        # - 0: 지연 보상 없음 (현재 프레임 직접 사용)
        # - 1: 1프레임 전 (권장 - 대부분의 하드웨어)
        # - 2: 2프레임 전 (느린 GPU에서 권장)
        # 실제 지연은 GPU 성능에 따라 다름, 테스트 후 조정
        self.ai_latency_frames = 1

        # Background management (기존 유지)
        self.bg_buffers = {}
        self.active_profile = 'default'
        self.bg_gpu = None
        self.has_bg = False

        # Morph Logic (기존 유지)
        self.morph_logic = MorphLogic()

        # V25.0 Components (simplified - CPU bilateral filter based)
        self.mask_manager = MaskManager()

        # Paths
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.root_dir, "recorded_data", "personal_data")

        # V29/V31: High-Fidelity processing buffers (원본 해상도, 다운스케일 제거)
        self.guided_result = None  # Guided Filter 결과 (저주파)
        self.freq_sep_result = None  # Frequency Separation 결과
        self.v29_frame_count = 0  # 디버그 로그용 프레임 카운터

        # V31: Dual-Pass 피부 보정 버퍼
        self.wide_smooth_result = None  # Wide Pass 결과 (radius 15)
        self.fine_smooth_result = None  # Fine Pass 결과 (radius 5)

        # V39: AI Skin Parser for hybrid masking
        self.skin_parser = None
        self.skin_parser_enabled = True  # Can be disabled via settings

        # V4: Forward Mask buffers (순방향 마스크 기반 합성)
        self.forward_mask_gpu = None
        self.forward_mask_dilated_gpu = None

        # V28.0: Stabilized mask buffer (동기화된 마스크)
        self.stabilized_mask_gpu = None

        # V36: Warp Grid Based Mask buffers
        self.warp_grid_mask_gpu = None           # 워핑 강도 기반 마스크
        self.warp_grid_mask_forward_gpu = None   # 순방향 워핑된 마스크

        # V40: Skeleton Patch buffers
        self.torso_mask_gpu = None               # 몸통 영역 마스크
        self.combined_mask_gpu = None            # AI + Torso 병합 마스크

        if HAS_CUDA:
            # [V44] 프레임 독립 처리 - Stabilizer bypass
            self.mask_stabilizer = MaskStabilizer(
                base_alpha=0.12,
                fast_alpha=0.85,
                diff_threshold=0.04,
                bypass=True  # [V44] Frame-independent mode
            )
            # [V44] 프레임 독립 처리 - Stabilizer bypass
            self.warp_grid_stabilizer = WarpGridStabilizer(
                base_alpha=0.05,
                max_alpha=0.98,
                delta_scale=15.0,
                bypass=True  # [V44] Frame-independent mode
            )
            # [V33] Frame Sync Buffer for AI latency compensation
            self.frame_sync_buffer = FrameSyncBuffer(max_size=3)
            self.stream = cp.cuda.Stream(non_blocking=True)
            # Core kernels (warping, compositing)
            self.warp_kernel = cp.RawKernel(WARP_KERNEL_CODE, 'warp_kernel')
            self.composite_kernel = cp.RawKernel(COMPOSITE_KERNEL_CODE, 'composite_kernel')
            self.skin_kernel = cp.RawKernel(SKIN_SMOOTH_KERNEL_CODE, 'skin_smooth_kernel')

            # YY-Style kernels (bilateral smoothing) - Legacy
            self.bilateral_kernel = cp.RawKernel(BILATERAL_SMOOTH_KERNEL_CODE, 'bilateral_smooth_kernel')
            self.resize_kernel = cp.RawKernel(GPU_RESIZE_KERNEL_CODE, 'gpu_resize_kernel')
            self.mask_resize_kernel = cp.RawKernel(GPU_MASK_RESIZE_KERNEL_CODE, 'gpu_mask_resize_kernel')
            self.blend_kernel = cp.RawKernel(FINAL_BLEND_KERNEL_CODE, 'final_blend_kernel')

            # V29: High-Fidelity kernels (Guided Filter + LAB Color Space)
            self.guided_filter_kernel = cp.RawKernel(GUIDED_FILTER_KERNEL_CODE, 'guided_filter_kernel')
            self.freq_separation_kernel = cp.RawKernel(FAST_SKIN_SMOOTH_KERNEL_CODE, 'fast_skin_smooth_kernel')
            self.lab_smooth_kernel = cp.RawKernel(LAB_SKIN_SMOOTH_KERNEL_CODE, 'lab_skin_smooth_kernel')

            # V31: Dual-Pass Smooth kernel (Wide/Fine 합성)
            self.dual_pass_smooth_kernel = cp.RawKernel(DUAL_PASS_SMOOTH_KERNEL_CODE, 'dual_pass_smooth_kernel')

            # V4: Forward Mask based Composite (순방향 마스크 기반 합성)
            self.forward_warp_mask_kernel = cp.RawKernel(FORWARD_WARP_MASK_KERNEL_CODE, 'forward_warp_mask_kernel')
            self.mask_dilate_kernel = cp.RawKernel(MASK_DILATE_KERNEL_CODE, 'mask_dilate_kernel')
            self.simple_composite_kernel = cp.RawKernel(SIMPLE_COMPOSITE_KERNEL_CODE, 'simple_composite_kernel')

            # V28.0: Void Fill Composite (동기화된 트리플 레이어)
            self.void_fill_composite_kernel = cp.RawKernel(VOID_FILL_COMPOSITE_KERNEL_CODE, 'void_fill_composite_kernel')

            # V34: Background Warp Prevention & Clean 2-Layer Composite
            self.modulate_displacement_kernel = cp.RawKernel(MODULATE_DISPLACEMENT_KERNEL_CODE, 'modulate_displacement_kernel')
            self.layered_composite_kernel = cp.RawKernel(LAYERED_COMPOSITE_KERNEL_CODE, 'layered_composite_kernel')

            # V36: Warp Grid Based Mask (근본적 재설계)
            self.warp_mask_from_grid_kernel = cp.RawKernel(WARP_MASK_FROM_GRID_KERNEL_CODE, 'warp_mask_from_grid_kernel')
            self.clean_composite_kernel = cp.RawKernel(CLEAN_COMPOSITE_KERNEL_CODE, 'clean_composite_kernel')

            # V40: Skeleton Patch (AI Mask + Torso Mask)
            self.mask_combine_kernel = cp.RawKernel(MASK_COMBINE_KERNEL_CODE, 'mask_combine_kernel')

            # V41: Logical Void Fill (Time-Locked Sync)
            self.logical_void_fill_kernel = cp.RawKernel(LOGICAL_VOID_FILL_KERNEL_CODE, 'logical_void_fill_kernel')

            # V43: Inverse Warp Validity (번개 현상 근본 해결)
            self.inverse_warp_validity_kernel = cp.RawKernel(
                INVERSE_WARP_VALIDITY_KERNEL_CODE,
                'inverse_warp_validity_kernel'
            )

            # V44: Simple Void Fill (Frame-Independent)
            self.simple_void_fill_kernel = cp.RawKernel(
                SIMPLE_VOID_FILL_KERNEL_CODE,
                'simple_void_fill_kernel'
            )

            self._warmup_kernels()
            self._load_all_backgrounds(profiles)

    def _warmup_kernels(self):
        """Pre-compile CUDA kernels"""
        print("   [INIT] Warming up CUDA Kernels...")
        try:
            h, w = 64, 64
            dummy_src = cp.zeros((h, w, 3), dtype=cp.uint8)
            dummy_dst = cp.zeros_like(dummy_src)
            dummy_exclusion = cp.zeros(15, dtype=cp.float32)

            # Legacy kernel warmup
            self.skin_kernel(
                (2, 2), (32, 32),
                (dummy_src, dummy_dst, w, h, cp.float32(0.5),
                 cp.float32(32), cp.float32(32), cp.float32(10),
                 cp.float32(128), cp.float32(128), cp.float32(128),
                 cp.float32(0.0),
                 dummy_exclusion)
            )

            cp.cuda.Stream.null.synchronize()
            print("   [INIT] Core Kernels Compiled")
        except Exception as e:
            print(f"   [WARNING] Warm-up failed: {e}")

    def _init_skin_parser(self):
        """
        [V39] Initialize AI Skin Parser for hybrid masking.

        Lazy initialization of BiSeNet V2 based skin parsing.
        Falls back to FaceMesh-only masking if engine not found.
        """
        if self.skin_parser is not None:
            return  # Already initialized

        if not self.skin_parser_enabled:
            return  # Disabled by user

        if not HAS_SKIN_PARSER:
            print("[BEAUTY] SkinParser module not available, using FaceMesh only")
            return

        try:
            engine_path = os.path.join(
                self.root_dir, "assets", "models", "parsing", "bisenet_v2_fp16.engine"
            )

            if os.path.exists(engine_path):
                self.skin_parser = SkinParser(engine_path)
                if self.skin_parser.is_ready:
                    print("[BEAUTY] SkinParser loaded successfully (Hybrid Masking enabled)")
                else:
                    print("[BEAUTY] SkinParser failed to initialize, using FaceMesh only")
                    self.skin_parser = None
            else:
                print(f"[BEAUTY] SkinParser engine not found at {engine_path}")
                print("[BEAUTY] Using FaceMesh-only masking (run tools/convert_bisenet.py to enable)")

        except Exception as e:
            print(f"[BEAUTY] SkinParser init failed: {e}")
            self.skin_parser = None
            self.skin_parser_enabled = False

    # ==========================================================================
    # Background Management (기존 100% 유지)
    # ==========================================================================
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

    def _init_bg_buffers(self, w, h, tmpl):
        for p, data in self.bg_buffers.items():
            if data['gpu'] is None or (data['gpu'].shape[1] != w or data['gpu'].shape[0] != h):
                if data['cpu'] is not None:
                    rz = cv2.resize(data['cpu'], (w, h))
                    data['gpu'] = cp.asarray(rz)
                else:
                    data['gpu'] = cp.zeros_like(tmpl)

    # ==========================================================================
    # Exclusion Zones (레거시 호환용 - 원형 기반)
    # ==========================================================================
    def _calculate_exclusion_zones(self, lm):
        """Legacy circular exclusion zones for backward compatibility"""
        zones = []
        target_keys = ['EYE_L', 'EYE_R', 'BROW_L', 'BROW_R', 'LIPS']

        for key in target_keys:
            if key in FaceMesh.FACE_INDICES:
                indices = FaceMesh.FACE_INDICES[key]
                pts = lm[indices]

                center = np.mean(pts, axis=0)
                cx, cy = center[0], center[1]

                dists = np.sqrt(np.sum((pts - center)**2, axis=1))
                max_dist = np.max(dists)

                padding = 0.9
                radius = max_dist * padding

                zones.extend([cx, cy, radius])
            else:
                zones.extend([0, 0, 0])

        return np.array(zones, dtype=np.float32)

    # ==========================================================================
    # V31: Dual-Pass High-Fidelity Skin Smoothing
    # - Wide Pass (radius 15, epsilon 0.02): 피부톤 전체를 도자기처럼 균일화
    # - Fine Pass (radius 5, epsilon 0.008): 미세 디테일 보존
    # - 두 결과를 비선형 곡선(pow 0.3)으로 합성하여 엣지 보존 극대화
    # ==========================================================================
    def _process_skin_yy_style(self, frame_gpu, landmarks, params):
        """
        V31: Dual-Pass High-Fidelity skin smoothing

        Key improvements over V30 (Single-Pass):
        - Wide Pass: 넓은 반경으로 피부톤 전체를 균일화 (도자기 질감)
        - Fine Pass: 좁은 반경으로 눈썹/속눈썹 등 미세 디테일 유지
        - Non-linear Detail Curve: powf(edge_factor, 0.3f)로 엣지 복원력 기하급수적 강화
        - Contrast Boost: 1.05f로 투명감 있는 피부 표현

        Result: 피부는 매끈, 눈코입은 초선명
        """
        h, w = frame_gpu.shape[:2]

        # Get parameters
        skin_strength = params.get('skin_smooth', 0.0)
        skin_tone_val = params.get('skin_tone', 0.0)
        color_temp = params.get('color_temperature', 0.0)
        color_tint = params.get('color_tint', 0.0)

        # Skip if no processing needed
        has_skin = skin_strength > 0.01
        has_tone = abs(skin_tone_val) > 0.01
        has_color = abs(color_temp) > 0.01 or abs(color_tint) > 0.01

        if not has_skin and not has_tone and not has_color:
            return frame_gpu, None

        # [V39] AI Skin Parsing for hybrid masking
        ai_skin_mask = None
        if self.skin_parser is not None and self.skin_parser.is_ready:
            try:
                ai_skin_mask = self.skin_parser.infer(frame_gpu)
                # Debug: Log AI mask stats every 60 frames
                if ai_skin_mask is not None and self.v29_frame_count % 60 == 0:
                    mask_mean = float(cp.mean(ai_skin_mask))
                    mask_max = float(cp.max(ai_skin_mask))
                    mask_coverage = float(cp.sum(ai_skin_mask > 128)) / (w * h) * 100
                    print(f"[SKIN-DEBUG] AI Mask: mean={mask_mean:.1f}, max={mask_max:.0f}, coverage={mask_coverage:.1f}%")
            except Exception as e:
                if self.v29_frame_count % 300 == 1:  # Log every 10 seconds at 30fps
                    print(f"[BEAUTY] SkinParser inference failed: {e}")
        elif self.v29_frame_count % 300 == 0:
            # Log fallback mode periodically
            print(f"[SKIN-DEBUG] Using FaceMesh-only (SkinParser not ready)")

        # Generate skin mask using hybrid approach (AI + FaceMesh)
        # Falls back to FaceMesh-only if AI mask is None
        skin_mask_gpu = self.mask_manager.generate_hybrid_mask(
            landmarks, ai_skin_mask, w, h, padding_ratio=1.25
        )

        if skin_mask_gpu is None:
            return frame_gpu, None

        # 디버그 로그 (60프레임마다 1회)
        self.v29_frame_count += 1
        should_log = (self.v29_frame_count % 60 == 1)

        # ===== V31 Dual-Pass Skin Smoothing =====
        if has_skin:
            # 원본 평균 저장 (디버그용)
            orig_mean = float(cp.mean(frame_gpu)) if should_log else 0

            # Initialize or resize buffers (원본 해상도)
            if (self.wide_smooth_result is None or
                self.wide_smooth_result.shape[0] != h or
                self.wide_smooth_result.shape[1] != w):
                self.wide_smooth_result = cp.zeros((h, w, 3), dtype=cp.uint8)
                self.fine_smooth_result = cp.zeros((h, w, 3), dtype=cp.uint8)
                self.freq_sep_result = cp.zeros((h, w, 3), dtype=cp.uint8)
                print(f"[V31] Dual-Pass 버퍼 초기화: {w}x{h}")

            block_dim = (16, 16)
            grid_dim = ((w + block_dim[0] - 1) // block_dim[0],
                        (h + block_dim[1] - 1) // block_dim[1])

            # ============================================
            # [Step 1] Wide Pass: 넓은 반경으로 피부톤 균일화
            # - radius 15, epsilon 0.02
            # - 피부 전체를 부드럽게 밀어버림 (도자기 효과)
            # ============================================
            wide_radius = 15
            wide_epsilon = 0.02

            try:
                self.guided_filter_kernel(
                    grid_dim, block_dim,
                    (frame_gpu, frame_gpu, self.wide_smooth_result, skin_mask_gpu,
                     cp.int32(w), cp.int32(h),
                     cp.int32(wide_radius),
                     cp.float32(wide_epsilon))
                )
            except Exception as e:
                print(f"[V31 ERROR] Wide Pass 실패: {e}")
                return frame_gpu, skin_mask_gpu

            # ============================================
            # [Step 2] Fine Pass: 좁은 반경으로 디테일 보존
            # - radius 5, epsilon 0.008
            # - 속눈썹, 눈동자 등 미세 엣지 유지
            # ============================================
            fine_radius = 5
            fine_epsilon = 0.008

            try:
                self.guided_filter_kernel(
                    grid_dim, block_dim,
                    (frame_gpu, frame_gpu, self.fine_smooth_result, skin_mask_gpu,
                     cp.int32(w), cp.int32(h),
                     cp.int32(fine_radius),
                     cp.float32(fine_epsilon))
                )
            except Exception as e:
                print(f"[V31 ERROR] Fine Pass 실패: {e}")
                return frame_gpu, skin_mask_gpu

            # ============================================
            # [Step 3] Dual-Pass 합성 (비선형 디테일 곡선)
            # - Wide/Fine 결과를 입력받아 합성
            # - powf(edge_factor, 0.3f)로 엣지 영역 복원력 강화
            # - Contrast 부스팅 1.05f로 투명감 표현
            # ============================================
            blend_strength = min(skin_strength * 1.5, 0.90)  # 최대 90%

            try:
                self.dual_pass_smooth_kernel(
                    grid_dim, block_dim,
                    (frame_gpu, self.wide_smooth_result, self.fine_smooth_result,
                     skin_mask_gpu, self.freq_sep_result,
                     cp.int32(w), cp.int32(h),
                     cp.float32(skin_strength),
                     cp.float32(blend_strength))
                )
            except Exception as e:
                print(f"[V31 ERROR] Dual-Pass Smooth 실패: {e}")
                return frame_gpu, skin_mask_gpu

            # Step 4: Color Grading 적용 (선택적)
            if has_color:
                result_gpu = cp.empty_like(frame_gpu)
                self.blend_kernel(
                    grid_dim, block_dim,
                    (self.freq_sep_result, self.freq_sep_result, skin_mask_gpu, result_gpu,
                     cp.int32(w), cp.int32(h),
                     cp.float32(0.0),
                     cp.float32(color_temp),
                     cp.float32(color_tint))
                )
                frame_gpu = result_gpu
            else:
                frame_gpu = cp.asarray(self.freq_sep_result)

            # 디버그 로그 (60프레임마다)
            if should_log:
                result_mean = float(cp.mean(frame_gpu))
                diff = abs(result_mean - orig_mean)
                print(f"[V31-DP] str={skin_strength:.2f}, blend={blend_strength:.2f}, diff={diff:.2f}")

        # ===== Skin Tone / Color Only (No Smoothing) =====
        elif has_tone or has_color:
            result_gpu = cp.empty_like(frame_gpu)
            grid_dim = ((w + 15) // 16, (h + 15) // 16)

            self.blend_kernel(
                grid_dim, (16, 16),
                (frame_gpu, frame_gpu, skin_mask_gpu, result_gpu,
                 w, h,
                 cp.float32(0.0),  # No smoothing blend
                 cp.float32(color_temp),
                 cp.float32(color_tint))
            )
            frame_gpu = result_gpu

        return frame_gpu, skin_mask_gpu

    # ==========================================================================
    # V45.1: Body Keypoint Validation
    # ==========================================================================
    def _are_keypoints_valid(self, kpts, indices, w, h, min_conf=0.3):
        """
        [V46.2] 화면 밖 keypoint 검증 - 정상화
        - margin: 0 (정확히 화면 내부만)
        - min_conf: 0.3 (원래대로)

        Args:
            kpts: body_landmarks 배열 (N, 3) - [x, y, confidence]
            indices: 검사할 keypoint 인덱스 리스트 (예: [11, 12] for hips)
            w, h: 화면 크기
            min_conf: 최소 신뢰도 (기본 0.3)

        Returns:
            bool: 모든 keypoint가 유효하면 True
        """
        if kpts is None:
            return False

        for idx in indices:
            if idx >= len(kpts):
                return False

            pt = kpts[idx]
            x, y = pt[0], pt[1]

            # [V46.2] 정확히 화면 내부만 허용 (margin = 0)
            if x < 0 or x > w:
                return False
            if y < 0 or y > h:
                return False

            # confidence 체크 (3번째 값이 있는 경우)
            if len(pt) >= 3:
                conf = pt[2]
                if conf < min_conf:
                    return False

        return True

    # ==========================================================================
    # V39: Mask-Warp Spatial Alignment
    # ==========================================================================
    def _align_mask_to_landmarks(self, mask_gpu, current_body, stabilized_body):
        """
        [V39] AI 마스크를 안정화된 랜드마크 위치에 공간적으로 정렬

        핵심: 시간적 지연은 그대로 두고, 공간적으로만 마스크를 이동
        - AI 마스크: 현재 프레임의 사람 위치
        - 워핑 그리드: 안정화된 랜드마크 기준 (지연됨)
        - 이 불일치가 "번개/문신" 현상의 원인

        해결책: 마스크를 워핑 기준(안정화된 랜드마크)에 맞게 이동

        Args:
            mask_gpu: 현재 프레임의 AI 마스크 (CuPy array, uint8)
            current_body: 현재 프레임의 바디 랜드마크 (numpy array, Nx2)
            stabilized_body: 안정화된 바디 랜드마크 (numpy array, Nx2)

        Returns:
            aligned_mask: 공간 정렬된 마스크 (CuPy array, uint8)
        """
        if mask_gpu is None:
            return mask_gpu

        if current_body is None or stabilized_body is None:
            return mask_gpu

        # 랜드마크 배열 크기 확인
        if len(current_body) == 0 or len(stabilized_body) == 0:
            return mask_gpu

        try:
            # 1. 현재 랜드마크 중심 계산
            current_center = np.mean(current_body, axis=0)  # (x, y)

            # 2. 안정화된 랜드마크 중심 계산
            stable_center = np.mean(stabilized_body, axis=0)  # (x, y)

            # 3. 오프셋 계산 (안정화 중심 - 현재 중심)
            # 마스크를 현재 위치에서 안정화된 위치로 이동
            offset_x = stable_center[0] - current_center[0]
            offset_y = stable_center[1] - current_center[1]

            # 4. 오프셋이 너무 작으면 스킵 (불필요한 연산 방지)
            MIN_OFFSET = 0.5  # 0.5픽셀 미만은 무시
            if abs(offset_x) < MIN_OFFSET and abs(offset_y) < MIN_OFFSET:
                return mask_gpu

            # 5. 오프셋이 너무 크면 제한 (갑작스러운 점프 방지)
            # 해상도 비례: height * 0.05 (1080p에서 약 54픽셀)
            h = mask_gpu.shape[0]
            MAX_OFFSET = h * 0.05
            offset_x = np.clip(offset_x, -MAX_OFFSET, MAX_OFFSET)
            offset_y = np.clip(offset_y, -MAX_OFFSET, MAX_OFFSET)

            # 6. 마스크 이동 (CuPy scipy.ndimage.shift 사용)
            # shift 순서: (y, x) - numpy/cupy 배열 인덱싱 순서
            aligned_mask = cupyx.scipy.ndimage.shift(
                mask_gpu.astype(cp.float32),
                shift=(offset_y, offset_x),
                order=1,  # 선형 보간
                mode='constant',
                cval=0  # 범위 밖은 0
            )

            return cp.clip(aligned_mask, 0, 255).astype(cp.uint8)

        except Exception as e:
            # 오류 발생 시 원본 마스크 반환 (안전한 폴백)
            print(f"[V39 WARNING] Mask alignment failed: {e}")
            return mask_gpu

    # ==========================================================================
    # V36: Warp Grid Based Mask Generation
    # ==========================================================================
    def _generate_warp_mask_from_grid(self, w, h, sw, sh, scale, threshold=0.5):
        """
        [V36] 워핑 그리드에서 마스크 생성

        워핑 강도(|dx|+|dy|)가 threshold 이상인 영역을 마스크로 추출.
        워핑 그리드와 완벽히 동기화되어 시간 지연 문제 해결.

        Args:
            w, h: 전체 해상도
            sw, sh: 워핑 그리드 해상도
            scale: 스케일 배율
            threshold: 워핑 강도 임계값 (픽셀 단위)

        Returns:
            warp_mask_gpu: 워핑 영역 마스크 (CuPy array, uint8)
        """
        # 버퍼 초기화
        if self.warp_grid_mask_gpu is None or self.warp_grid_mask_gpu.shape != (h, w):
            self.warp_grid_mask_gpu = cp.zeros((h, w), dtype=cp.uint8)
            self.warp_grid_mask_forward_gpu = cp.zeros((h, w), dtype=cp.uint8)

        block_dim = (16, 16)
        grid_dim = ((w + 15) // 16, (h + 15) // 16)

        # Step 1: 워핑 그리드에서 마스크 생성
        self.warp_mask_from_grid_kernel(
            grid_dim, block_dim,
            (self.gpu_dx, self.gpu_dy, self.warp_grid_mask_gpu,
             w, h, sw, sh, scale, cp.float32(threshold))
        )

        # Step 2: 가우시안 블러로 경계 부드럽게
        smoothed = cupyx.scipy.ndimage.gaussian_filter(
            self.warp_grid_mask_gpu.astype(cp.float32), sigma=3.0
        )
        self.warp_grid_mask_gpu = cp.clip(smoothed, 0, 255).astype(cp.uint8)

        # Step 3: 순방향 워핑으로 최종 사람 영역 계산
        self.warp_grid_mask_forward_gpu.fill(0)
        self.forward_warp_mask_kernel(
            grid_dim, block_dim,
            (self.warp_grid_mask_gpu, self.warp_grid_mask_forward_gpu,
             self.gpu_dx, self.gpu_dy,
             w, h, sw, sh, scale)
        )

        # Step 4: 홀 채우기 (Dilate)
        temp_mask = cp.zeros_like(self.warp_grid_mask_forward_gpu)
        self.mask_dilate_kernel(
            grid_dim, block_dim,
            (self.warp_grid_mask_forward_gpu, temp_mask,
             w, h, 2)
        )

        # Step 5: 최종 블러
        final_mask = cupyx.scipy.ndimage.gaussian_filter(
            temp_mask.astype(cp.float32), sigma=2.0
        )
        final_mask = cp.clip(final_mask, 0, 255).astype(cp.uint8)

        return final_mask

    # ==========================================================================
    # Main Processing Pipeline
    # ==========================================================================
    def process(self, frame, faces, body_landmarks=None, params=None, mask=None, frame_cpu=None, bg_stable=False):
        """
        Main processing pipeline

        :param frame: Input frame (CPU or GPU)
        :param faces: List of FaceResult from FaceMesh
        :param body_landmarks: Body keypoints from pose tracker
        :param params: Processing parameters dict
        :param mask: Alpha mask for compositing
        :param frame_cpu: Optional CPU frame to avoid GPU->CPU transfer
        :param bg_stable: [V6] 정적 배경 안정성 플래그 (True: 슬리밍 합성 가능)
        :return: Processed frame
        """
        if frame is None or not HAS_CUDA:
            return frame
        if params is None:
            params = {}

        # Store CPU frame if provided (optimization)
        self._frame_cpu_cache = frame_cpu

        is_gpu_input = hasattr(frame, 'device')

        with self.stream:
            if is_gpu_input:
                frame_gpu = frame
                h, w = frame.shape[:2]
            else:
                h, w = frame.shape[:2]
                frame_gpu = cp.asarray(frame)

            # Initialize buffers on size change
            if self.cache_w != w or self.cache_h != h:
                self.cache_w, self.cache_h = w, h
                self.gpu_initialized = False
                self._init_bg_buffers(w, h, frame_gpu)

                self.body_stabilizer.reset()
                self.face_stabilizer.reset()
                # V28.0: Reset MaskStabilizer on size change
                if self.mask_stabilizer is not None:
                    self.mask_stabilizer.reset()
                # V31: Reset WarpGridStabilizer on size change
                if self.warp_grid_stabilizer is not None:
                    self.warp_grid_stabilizer.reset()
                # V33: Reset FrameSyncBuffer on size change
                if self.frame_sync_buffer is not None:
                    self.frame_sync_buffer.reset()

                if self.active_profile in self.bg_buffers and self.bg_buffers[self.active_profile]['gpu'] is not None:
                    self.bg_gpu = self.bg_buffers[self.active_profile]['gpu']
                    self.has_bg = True

            sw, sh = int(w * self.map_scale), int(h * self.map_scale)
            if not self.gpu_initialized:
                self.gpu_dx = cp.zeros((sh, sw), dtype=cp.float32)
                self.gpu_dy = cp.zeros((sh, sw), dtype=cp.float32)
                self.gpu_initialized = True

            if self.bg_gpu is None:
                self.bg_gpu = cp.copy(frame_gpu)
                self.has_bg = True

            # ==================================================================
            # [V45.1] Frame Independence 유지 - FrameSyncBuffer 비활성화
            # - V45에서 복구했으나 오히려 떨림 악화
            # - 원인: AI 지연이 가변적이라 과거 프레임-현재 마스크 불일치 발생
            # - 결론: 현재 프레임 직접 사용이 가장 안정적
            # ==================================================================
            synced_frame = frame_gpu  # 현재 프레임 직접 사용

            # [V45.1 DISABLED] FrameSyncBuffer - 가변 지연으로 인해 비활성화
            # if self.frame_sync_buffer is not None:
            #     self.frame_sync_buffer.push(frame_gpu)
            #     if mask is not None and bg_stable:
            #         _synced = self.frame_sync_buffer.get_synced_frame(
            #             delay_frames=self.ai_latency_frames
            #         )
            #         if _synced is not None:
            #             synced_frame = _synced

            # Mask handling
            # [V45] 배경합성 조건 정교화
            # - 실루엣이 "안쪽으로 수축"하는 기능에서만 배경합성 활성화
            # - hip_widen(골반늘리기)은 "바깥으로 확장"이므로 Void가 발생하지 않음
            # - ribcage_slim(갈비뼈줄이기) 추가: 수축이므로 Void 발생
            # - nose_slim, eye_scale, head_scale은 얼굴 내부 변형이므로 배경합성 불필요
            is_slimming_enabled = (
                params.get('face_v', 0) > 0 or
                params.get('waist_slim', 0) > 0 or
                params.get('shoulder_narrow', 0) > 0 or
                params.get('ribcage_slim', 0) > 0
            )
            # 주의: hip_widen은 의도적으로 제외됨 (확장은 Void를 만들지 않음)

            if self.has_bg and mask is not None and bg_stable and is_slimming_enabled:
                if hasattr(mask, 'device'):
                    mask_gpu = mask
                else:
                    mask_gpu = cp.asarray(mask)
                if mask_gpu.dtype == cp.float32 or mask_gpu.dtype == cp.float16:
                    mask_gpu = (mask_gpu * 255.0).astype(cp.uint8)
                use_bg = 1
            else:
                mask_gpu = cp.zeros((h, w), dtype=cp.uint8)
                use_bg = 0

            # ==================================================================
            # [Step 1] Skin Processing
            # [V35] 동기화된 프레임(synced_frame)을 베이스로 사용
            # ==================================================================
            source_for_warp = synced_frame
            skin_mask_debug = None

            if faces:
                raw_face = faces[0].landmarks
                stable_face = self.face_stabilizer.update(raw_face)

                # YY-Style Pipeline (Bilateral Filter based)
                # [V35] synced_frame을 입력으로 사용하여 시점 통일
                source_for_warp, skin_mask_debug = self._process_skin_yy_style(
                    synced_frame, stable_face, params
                )

            # ==================================================================
            # [Step 2] Morph Logic (기존 100% 유지)
            # ==================================================================
            self.morph_logic.clear()

            # [V30] bg_stable 감쇄 계수: 배경이 불안정하면 슬리밍 강도 50% 감쇄
            slim_damping = 1.0 if bg_stable else 0.5

            raw_body = body_landmarks.get() if hasattr(body_landmarks, 'get') else body_landmarks
            if raw_body is not None:
                kpts_xy = raw_body[:, :2]
                stable_kpts = self.body_stabilizer.update(kpts_xy)
                scaled_body = stable_kpts * self.map_scale

                # [V46.1] keypoint 유효성 검사를 위한 원본 좌표 보존
                # ★ 핵심 변경: raw_body 사용 (confidence 정보 포함)
                original_kpts = raw_body  # (N, 3) 배열: [x, y, confidence]

                # [V45.1] 각 보정별 필요 keypoint 인덱스 정의
                # 어깨: 5(왼쪽어깨), 6(오른쪽어깨)
                # 갈비뼈: 5, 6, 11, 12 (어깨 + 골반)
                # 허리: 5, 6, 11, 12 (어깨 + 골반)
                # 골반: 11(왼쪽골반), 12(오른쪽골반)

                # [V30] 슬리밍 파라미터에 damping 적용 + [V45.1] 유효성 검사
                if params.get('shoulder_narrow', 0) > 0:
                    if self._are_keypoints_valid(original_kpts, [5, 6], w, h):
                        self.morph_logic.collect_shoulder_params(scaled_body, params['shoulder_narrow'] * slim_damping)

                if params.get('ribcage_slim', 0) > 0:
                    if self._are_keypoints_valid(original_kpts, [5, 6, 11, 12], w, h):
                        self.morph_logic.collect_ribcage_params(scaled_body, params['ribcage_slim'] * slim_damping)

                if params.get('waist_slim', 0) > 0:
                    if self._are_keypoints_valid(original_kpts, [5, 6, 11, 12], w, h):
                        self.morph_logic.collect_waist_params(scaled_body, params['waist_slim'] * slim_damping)

                if params.get('hip_widen', 0) > 0:
                    if self._are_keypoints_valid(original_kpts, [11, 12], w, h):
                        self.morph_logic.collect_hip_params(scaled_body, params['hip_widen'] * slim_damping)

            if faces:
                lm_small = stable_face * self.map_scale

                face_v = params.get('face_v', 0)
                eye_scale = params.get('eye_scale', 0)
                head_scale = params.get('head_scale', 0)
                nose_slim = params.get('nose_slim', 0)

                if face_v > 0:
                    self.morph_logic.collect_face_contour_params(lm_small, face_v)
                if eye_scale > 0:
                    self.morph_logic.collect_eyes_params(lm_small, eye_scale)
                if head_scale != 0:
                    self.morph_logic.collect_head_params(lm_small, head_scale)
                if nose_slim > 0:
                    self.morph_logic.collect_nose_params(lm_small, nose_slim)

            # ==================================================================
            # [Step 3] Warping (기존 100% 유지)
            # ==================================================================
            self.gpu_dx.fill(0)
            self.gpu_dy.fill(0)

            warp_params = self.morph_logic.get_params()
            if len(warp_params) > 0:
                params_arr = np.array(warp_params, dtype=np.float32)
                params_gpu = cp.asarray(params_arr)

                block_dim = (16, 16)
                grid_dim = ((sw + block_dim[0] - 1) // block_dim[0],
                            (sh + block_dim[1] - 1) // block_dim[1])

                self.warp_kernel(grid_dim, block_dim,
                    (self.gpu_dx, self.gpu_dy, params_gpu, len(warp_params), sw, sh))

                # [V31] 그리드 블러 강화: sigma 1.0 → 2.0 (경계면 부드러움 향상)
                cupyx.scipy.ndimage.gaussian_filter(self.gpu_dx, sigma=2.0, output=self.gpu_dx)
                cupyx.scipy.ndimage.gaussian_filter(self.gpu_dy, sigma=2.0, output=self.gpu_dy)

                # [V44.5] WarpGridStabilizer 호출 비활성화 - 움직임 추적 지연 해결
                # if self.warp_grid_stabilizer is not None:
                #     self.gpu_dx, self.gpu_dy = self.warp_grid_stabilizer.update(self.gpu_dx, self.gpu_dy)

                # ==============================================================
                # [V34] Grid Modulation - 배경 워핑 원천 차단
                # - 인물 마스크를 small scale로 리사이즈
                # - dx, dy 그리드에 마스크를 곱하여 배경 영역 변위 = 0
                # ==============================================================
                if use_bg and mask_gpu is not None:
                    # 마스크를 small scale로 리사이즈
                    mask_small_gpu = cupyx.scipy.ndimage.zoom(
                        mask_gpu.astype(cp.float32),
                        (sh / h, sw / w),
                        order=1
                    ).astype(cp.uint8)

                    # ==============================================================
                    # [V47] Erosion 기반 배경 왜곡 차단
                    # - 마스크를 2픽셀 수축하여 워핑 영향권을 사람 안쪽으로 제한
                    # - 배경 픽셀에는 워핑 변위가 0이 되어 물리적으로 왜곡 불가능
                    # ==============================================================
                    # 1단계: 마스크 이진화
                    mask_float = mask_small_gpu.astype(cp.float32) / 255.0
                    mask_binary = (mask_float > 0.5).astype(cp.float32)

                    # 2단계: Binary Erosion으로 2픽셀 수축
                    mask_eroded = cupyx.scipy.ndimage.binary_erosion(
                        mask_binary.astype(cp.bool_), iterations=2
                    ).astype(cp.float32)

                    # 3단계: 약한 Gaussian blur로 경계 부드럽게
                    mask_smoothed = cupyx.scipy.ndimage.gaussian_filter(mask_eroded, sigma=1.5)

                    # 4단계: uint8로 변환
                    mask_for_modulation = (mask_smoothed * 255).astype(cp.uint8)

                    # 그리드 모듈레이션 적용
                    small_block_dim = (16, 16)
                    small_grid_dim = ((sw + small_block_dim[0] - 1) // small_block_dim[0],
                                      (sh + small_block_dim[1] - 1) // small_block_dim[1])

                    self.modulate_displacement_kernel(
                        small_grid_dim, small_block_dim,
                        (self.gpu_dx, self.gpu_dy, mask_for_modulation, sw, sh)
                    )

            # ==================================================================
            # [Step 4] V36: Void Only Fill 합성
            # - 기본: 워핑된 원본 프레임
            # - Void 영역만: 저장된 배경으로 패치 (슬리밍으로 비어진 공간만)
            # - 배경 전체 교체 X, Void만 채움 O
            # ==================================================================
            block_dim = (16, 16)
            grid_dim = ((w + block_dim[0] - 1) // block_dim[0],
                        (h + block_dim[1] - 1) // block_dim[1])
            scale = int(1.0 / self.map_scale)

            # 4-1. 버퍼 초기화
            if self.forward_mask_gpu is None or self.forward_mask_gpu.shape != (h, w):
                self.forward_mask_gpu = cp.zeros((h, w), dtype=cp.uint8)
                self.forward_mask_dilated_gpu = cp.zeros((h, w), dtype=cp.uint8)

            # ==============================================================
            # [V40] Skeleton Patch - AI 마스크 + Torso 마스크 병합
            # ==============================================================
            # 전략: Final_Mask = Max(AI_Mask, Torso_Mask)
            # - AI 마스크: 정밀한 외곽선 (배경 침범 방지)
            # - Torso 마스크: 내부 구멍 메움 (번개 현상 방지)
            # ==============================================================
            if use_bg and mask_gpu is not None and raw_body is not None:
                # 1. Torso 마스크 생성 (어깨-골반 영역)
                torso_mask = self.mask_manager.generate_torso_mask(raw_body, w, h)

                if torso_mask is not None:
                    # 2. GPU 업로드
                    torso_mask_gpu = cp.asarray(torso_mask)

                    # 3. 병합 버퍼 준비
                    if self.combined_mask_gpu is None or self.combined_mask_gpu.shape != (h, w):
                        self.combined_mask_gpu = cp.zeros((h, w), dtype=cp.uint8)

                    # 4. 마스크 병합 (픽셀별 Max)
                    self.mask_combine_kernel(
                        grid_dim, block_dim,
                        (mask_gpu, torso_mask_gpu, self.combined_mask_gpu, w, h)
                    )

                    # 5. 병합된 마스크를 메인 마스크로 사용
                    mask_gpu = self.combined_mask_gpu

            # ==============================================================
            # [V40] 원본 마스크 준비 (mask_orig: 사람이 "있었던" 영역)
            # - AI 마스크 기반으로 복원 (V38.1 방식)
            # - Skeleton Patch로 내부 구멍이 메워진 상태
            # ==============================================================
            if use_bg and mask_gpu is not None:
                # AI 마스크 경계 블러 처리
                mask_orig = cupyx.scipy.ndimage.gaussian_filter(
                    mask_gpu.astype(cp.float32), sigma=1.5
                ).astype(cp.uint8)
            else:
                # AI 마스크가 없으면 워핑 그리드에서 생성
                mask_orig = self._generate_warp_mask_from_grid(
                    w, h, sw, sh, scale, threshold=0.5
                )

            # [V41 LEGACY] 순방향 워핑된 마스크 (mask_fwd) 생성
            # - V41에서는 logical_void_fill_kernel이 단일 마스크만 사용
            # - 롤백 시 clean_composite_kernel에서 필요하므로 코드 유지
            # - 슬리밍으로 인해 수축된 영역을 나타냄
            self.forward_mask_gpu.fill(0)

            self.forward_warp_mask_kernel(
                grid_dim, block_dim,
                (mask_orig, self.forward_mask_gpu,
                 self.gpu_dx, self.gpu_dy,
                 w, h, sw, sh, scale)
            )

            # 4-4. 마스크 홀 채우기 (Dilate) - [V41 LEGACY]
            self.mask_dilate_kernel(
                grid_dim, block_dim,
                (self.forward_mask_gpu, self.forward_mask_dilated_gpu,
                 w, h, 2)  # [V40] radius 2 (V38.1 복원)
            )

            # 4-5. 순방향 마스크 평활화 - [V41 LEGACY] 롤백용으로 유지
            mask_fwd = cupyx.scipy.ndimage.gaussian_filter(
                self.forward_mask_dilated_gpu.astype(cp.float32), sigma=1.5
            ).astype(cp.uint8)
            _ = mask_fwd  # [V41] Suppress unused variable warning (kept for rollback)

            # ==============================================================
            # [V44] Simple Void Fill - 프레임 독립 합성
            # - 핵심: 프레임과 마스크가 100% 동일 시점
            # - 3단계 무결성 판정:
            #   * CASE 1: mask_at_warped > 0.4 → 전경 출력 (번개 차단)
            #   * CASE 2: mask_at_origin > 0.2 → 배경 출력 (Void 채움)
            #   * CASE 3: 그 외 → 원본 유지 (왜곡 차단)
            # ==============================================================
            result_gpu = cp.empty_like(frame_gpu)

            # [V44] Simple Void Fill - 프레임 독립 처리
            self.simple_void_fill_kernel(
                grid_dim, block_dim,
                (source_for_warp, self.bg_gpu,
                 mask_orig,
                 result_gpu,
                 self.gpu_dx, self.gpu_dy,
                 w, h, sw, sh, scale, use_bg)
            )

            # [V43 LEGACY] 롤백이 필요한 경우 아래 코드로 교체:
            # self.inverse_warp_validity_kernel(
            #     grid_dim, block_dim,
            #     (source_for_warp, self.bg_gpu,
            #      mask_orig,
            #      result_gpu,
            #      self.gpu_dx, self.gpu_dy,
            #      w, h, sw, sh, scale, use_bg)
            # )

            # ==================================================================
            # [Debug Visualization]
            # ==================================================================
            if params.get('show_body_debug', False):
                if hasattr(result_gpu, 'get'):
                    debug_img = result_gpu.get()
                else:
                    debug_img = cp.asnumpy(result_gpu)

                # Show skin mask boundary if available
                if skin_mask_debug is not None:
                    mask_np = cp.asnumpy(skin_mask_debug)
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)

                # Show processing mode
                cv2.putText(debug_img, "V44 Frame-Independent Void Fill", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                result_gpu = cp.asarray(debug_img)

            # Return result
            if is_gpu_input:
                return result_gpu
            else:
                return result_gpu.get()

