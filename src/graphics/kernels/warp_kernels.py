# Project MUSE - warp_kernels.py
# 워핑 및 마스크 관련 CUDA 커널
# (C) 2025 MUSE Corp. All rights reserved.

# ==============================================================================
# [KERNEL 1] Grid Generation (TPS Logic)
# ==============================================================================
WARP_KERNEL_CODE = r'''
extern "C" __global__
void warp_kernel(
    float* dx, float* dy,
    const float* params,
    int num_points,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;

    float acc_dx = 0.0f;
    float acc_dy = 0.0f;

    for (int i = 0; i < num_points; ++i) {
        int base = i * 7;
        float cx = params[base + 0];
        float cy = params[base + 1];
        float r = params[base + 2];
        float s = params[base + 3];
        float vx = params[base + 4];
        float vy = params[base + 5];
        int mode = (int)params[base + 6];

        float diff_x = x - cx;
        float diff_y = y - cy;
        float dist_sq = diff_x * diff_x + diff_y * diff_y;

        if (dist_sq < (r * r)) {
            float dist = sqrtf(dist_sq);
            float factor = (1.0f - dist / r);
            factor = factor * factor * s;
            if (mode == 0) { // Expand/Shrink
                acc_dx -= diff_x * factor;
                acc_dy -= diff_y * factor;
            } else if (mode == 1) { // Vector Shift
                acc_dx -= vx * factor * r * 0.5f;
                acc_dy -= vy * factor * r * 0.5f;
            } else if (mode == 2) { // Inverse
                acc_dx += diff_x * factor;
                acc_dy += diff_y * factor;
            }
        }
    }
    dx[idx] += acc_dx;
    dy[idx] += acc_dy;
}
'''

# ==============================================================================
# [KERNEL 1.5] Displacement Grid Modulation (V34 - Background Warp Prevention)
# ==============================================================================
MODULATE_DISPLACEMENT_KERNEL_CODE = r'''
extern "C" __global__
void modulate_displacement_kernel(
    float* dx,                       // In/Out: X 변위 맵 (small scale)
    float* dy,                       // In/Out: Y 변위 맵 (small scale)
    const unsigned char* mask,       // 인물 마스크 (small scale로 리사이즈됨, 0-255)
    int small_width, int small_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= small_width || y >= small_height) return;

    int idx = y * small_width + x;

    // 마스크 값을 0.0 ~ 1.0 범위로 변환
    float alpha = (float)mask[idx] / 255.0f;

    // 변위 그리드에 마스크를 곱함
    dx[idx] *= alpha;
    dy[idx] *= alpha;
}
'''

# ==============================================================================
# [KERNEL 17] Forward Warp Mask (순방향 마스크 워핑)
# ==============================================================================
FORWARD_WARP_MASK_KERNEL_CODE = r'''
// Custom atomicMax for unsigned char using CAS
__device__ __forceinline__ unsigned char atomicMaxUChar(unsigned char* address, unsigned char val) {
    // Get the 4-byte aligned address containing our byte
    unsigned int* base_address = (unsigned int*)((size_t)address & ~3);
    unsigned int byte_offset = ((size_t)address & 3) * 8;  // bit offset within the int
    unsigned int mask = 0xFF << byte_offset;

    unsigned int old_val, new_val, assumed;
    old_val = *base_address;

    do {
        assumed = old_val;
        unsigned char current_byte = (unsigned char)((assumed >> byte_offset) & 0xFF);
        unsigned char max_byte = (val > current_byte) ? val : current_byte;
        new_val = (assumed & ~mask) | ((unsigned int)max_byte << byte_offset);
        old_val = atomicCAS(base_address, assumed, new_val);
    } while (assumed != old_val);

    return (unsigned char)((old_val >> byte_offset) & 0xFF);
}

extern "C" __global__
void forward_warp_mask_kernel(
    const unsigned char* mask_in,    // 원본 마스크 (1채널, 0-255)
    unsigned char* mask_out,         // 순방향 워핑된 마스크 (1채널)
    const float* dx,                 // X 변위 맵 (small scale)
    const float* dy,                 // Y 변위 맵 (small scale)
    int width, int height,
    int small_width, int small_height,
    int scale
) {
    // 입력 좌표 (u, v) - 순방향이므로 입력 기준 순회
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;

    int in_idx = v * width + u;
    unsigned char mask_val = mask_in[in_idx];

    // 마스크가 0이면 스킵 (배경은 이동할 필요 없음)
    if (mask_val == 0) return;

    // 워핑 변위 가져오기
    int su = u / scale;
    int sv = v / scale;
    if (su >= small_width) su = small_width - 1;
    if (sv >= small_height) sv = small_height - 1;
    int s_idx = sv * small_width + su;

    // 순방향: 출력 위치 = 입력 위치 + 변위
    float shift_x = -dx[s_idx] * (float)scale;
    float shift_y = -dy[s_idx] * (float)scale;

    int out_x = (int)roundf((float)u + shift_x);
    int out_y = (int)roundf((float)v + shift_y);

    // 범위 체크
    if (out_x < 0 || out_x >= width || out_y < 0 || out_y >= height) return;

    int out_idx = out_y * width + out_x;

    // Custom Atomic Max: 여러 입력 픽셀이 같은 출력에 쓸 수 있음
    atomicMaxUChar(&mask_out[out_idx], mask_val);
}
'''

# ==============================================================================
# [KERNEL 18] Mask Dilate (마스크 확장으로 홀 채우기)
# ==============================================================================
MASK_DILATE_KERNEL_CODE = r'''
extern "C" __global__
void mask_dilate_kernel(
    const unsigned char* mask_in,
    unsigned char* mask_out,
    int width, int height,
    int radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    unsigned char max_val = 0;

    // radius 내 최대값 탐색
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                unsigned char val = mask_in[ny * width + nx];
                if (val > max_val) max_val = val;
            }
        }
    }

    mask_out[idx] = max_val;
}
'''

# ==============================================================================
# [KERNEL V40] Mask Combine (Skeleton Patch)
# ==============================================================================
MASK_COMBINE_KERNEL_CODE = r'''
extern "C" __global__
void mask_combine_kernel(
    const unsigned char* mask_a,     // AI 마스크 (정밀한 외곽선)
    const unsigned char* mask_b,     // Torso 마스크 (내부 채움)
    unsigned char* dst,              // 출력: 병합된 마스크
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // 픽셀별 최댓값 (Max)
    unsigned char a = mask_a[idx];
    unsigned char b = mask_b[idx];
    dst[idx] = (a > b) ? a : b;
}
'''

# ==============================================================================
# [KERNEL V36] Warp Mask From Grid
# ==============================================================================
WARP_MASK_FROM_GRID_KERNEL_CODE = r'''
extern "C" __global__
void warp_mask_from_grid_kernel(
    const float* dx_small,           // 워핑 그리드 X (small scale)
    const float* dy_small,           // 워핑 그리드 Y (small scale)
    unsigned char* warp_mask,        // 출력: 워핑 영역 마스크 (full scale)
    int width, int height,           // 전체 해상도
    int small_width, int small_height,  // 워핑 그리드 해상도
    int scale,                       // 스케일 배율
    float threshold                  // 워핑 강도 임계값 (권장: 0.5)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 워핑 그리드 좌표로 변환
    int sx = x / scale;
    int sy = y / scale;
    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;
    int s_idx = sy * small_width + sx;

    // 워핑 강도 계산 (픽셀 단위)
    float dx_val = dx_small[s_idx] * (float)scale;
    float dy_val = dy_small[s_idx] * (float)scale;
    float magnitude = sqrtf(dx_val * dx_val + dy_val * dy_val);

    // 임계값 기반 마스크 생성
    float edge0 = threshold * 0.5f;
    float edge1 = threshold * 2.0f;

    float t = (magnitude - edge0) / (edge1 - edge0);
    t = fmaxf(0.0f, fminf(1.0f, t));
    float alpha = t * t * (3.0f - 2.0f * t);  // smoothstep

    int idx = y * width + x;
    warp_mask[idx] = (unsigned char)(alpha * 255.0f);
}
'''
