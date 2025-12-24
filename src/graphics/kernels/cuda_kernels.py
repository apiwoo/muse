# Project MUSE - cuda_kernels.py
# V47: 리팩토링 - 커널 파일 분리 및 레거시 코드 제거
# 기존 import 호환성을 위한 re-export 모듈
# (C) 2025 MUSE Corp. All rights reserved.

__all__ = [
    # Warp kernels
    'WARP_KERNEL_CODE',
    'MODULATE_DISPLACEMENT_KERNEL_CODE',
    'FORWARD_WARP_MASK_KERNEL_CODE',
    'MASK_DILATE_KERNEL_CODE',
    'MASK_COMBINE_KERNEL_CODE',
    'WARP_MASK_FROM_GRID_KERNEL_CODE',
    # Composite kernels
    'SIMPLE_VOID_FILL_KERNEL_CODE',
    # Skin kernels
    'GUIDED_FILTER_KERNEL_CODE',
    'FAST_SKIN_SMOOTH_KERNEL_CODE',
    'DUAL_PASS_SMOOTH_KERNEL_CODE',
    # Utils kernels
    'GPU_RESIZE_KERNEL_CODE',
    'GPU_MASK_RESIZE_KERNEL_CODE',
    'FINAL_BLEND_KERNEL_CODE',
]

# ==============================================================================
# 워핑 관련 커널 (warp_kernels.py)
# ==============================================================================
from graphics.kernels.warp_kernels import (
    WARP_KERNEL_CODE,
    MODULATE_DISPLACEMENT_KERNEL_CODE,
    FORWARD_WARP_MASK_KERNEL_CODE,
    MASK_DILATE_KERNEL_CODE,
    MASK_COMBINE_KERNEL_CODE,
    WARP_MASK_FROM_GRID_KERNEL_CODE,
)

# ==============================================================================
# 합성 관련 커널 (composite_kernels.py)
# ==============================================================================
from graphics.kernels.composite_kernels import (
    SIMPLE_VOID_FILL_KERNEL_CODE,
)

# ==============================================================================
# 피부 보정 관련 커널 (skin_kernels.py)
# ==============================================================================
from graphics.kernels.skin_kernels import (
    GUIDED_FILTER_KERNEL_CODE,
    FAST_SKIN_SMOOTH_KERNEL_CODE,
    DUAL_PASS_SMOOTH_KERNEL_CODE,
)

# ==============================================================================
# 유틸리티 커널 (utils_kernels.py)
# ==============================================================================
from graphics.kernels.utils_kernels import (
    GPU_RESIZE_KERNEL_CODE,
    GPU_MASK_RESIZE_KERNEL_CODE,
    FINAL_BLEND_KERNEL_CODE,
)

# ==============================================================================
# 레거시 호환용 Alias (경고 없이 작동)
# - 실제 커널 코드는 제거되었지만 import 에러 방지용
# - beauty_engine.py에서 import만 하고 사용하지 않는 커널들
# ==============================================================================

# V4 레거시: 순방향 마스크 기반 합성 (V44로 대체됨)
# SIMPLE_COMPOSITE_KERNEL_CODE - 제거됨

# V5 레거시: Void Fill Composite (V44로 대체됨)
# VOID_FILL_COMPOSITE_KERNEL_CODE - 제거됨

# V34 레거시: 2-Layer Composite (V44로 대체됨)
# LAYERED_COMPOSITE_KERNEL_CODE - 제거됨

# V36 레거시: Clean Composite (V44로 대체됨)
# CLEAN_COMPOSITE_KERNEL_CODE - 제거됨

# V41 레거시: Logical Void Fill (V44로 대체됨)
# LOGICAL_VOID_FILL_KERNEL_CODE - 제거됨

# V43 레거시: Inverse Warp Validity (V44로 대체됨)
# INVERSE_WARP_VALIDITY_KERNEL_CODE - 제거됨

# 기타 레거시:
# COMPOSITE_KERNEL_CODE - 제거됨
# SKIN_SMOOTH_KERNEL_CODE - 제거됨 (warmup 코드도 수정 필요)
# BILATERAL_SMOOTH_KERNEL_CODE - 제거됨
# LAB_SKIN_SMOOTH_KERNEL_CODE - 제거됨
