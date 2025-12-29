# Project MUSE - composite_kernels.py
# 합성 관련 CUDA 커널
# (C) 2025 MUSE Corp. All rights reserved.

# ==============================================================================
# [KERNEL V-FINAL3] Simple Void Fill (Dynamic Threshold)
# ==============================================================================
# 번개 현상 해결 (수정판):
# - 문제: 변위가 워핑 포인트(턱, 허리) 주변에만 존재
# - 이전 오류: "변위 없음 = Void"로 판정 → 사람 전체가 배경으로 대체됨
# - 해결: mask_at_warped 기반 로직 복원 + 변위 있을 때 임계값 낮춤
#
# 핵심 로직:
# - 변위 있는 영역: mask_at_warped > 0.02 이면 사람 (관대하게 → 번개 방지)
# - 변위 없는 영역: mask_at_warped > 0.1 이면 사람 (기존대로)
# - Void: mask_at_warped <= threshold AND mask_here > 0.1 → 배경으로 채움
# ==============================================================================
SIMPLE_VOID_FILL_KERNEL_CODE = r'''
extern "C" __global__
void simple_void_fill_kernel(
    const unsigned char* src,        // 현재 프레임 (피부보정 적용됨)
    const unsigned char* bg,         // 정적 배경 (Clean Plate)
    const unsigned char* mask,       // 현재 프레임의 마스크
    unsigned char* dst,              // 출력
    const float* dx_small,           // 역방향 워핑 그리드
    const float* dy_small,
    int width, int height,
    int small_width, int small_height,
    int scale,
    int use_bg
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx_rgb = idx * 3;

    // === 워핑 변위 계산 ===
    int sx = x / scale;
    int sy = y / scale;
    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;
    int s_idx = sy * small_width + sx;

    float shift_x = dx_small[s_idx] * (float)scale;
    float shift_y = dy_small[s_idx] * (float)scale;

    int u = (int)(x + shift_x);
    int v = (int)(y + shift_y);

    // === 배경 미사용 시: 단순 워핑만 ===
    if (!use_bg) {
        if (u >= 0 && u < width && v >= 0 && v < height) {
            int warped_idx_rgb = (v * width + u) * 3;
            dst[idx_rgb + 0] = src[warped_idx_rgb + 0];
            dst[idx_rgb + 1] = src[warped_idx_rgb + 1];
            dst[idx_rgb + 2] = src[warped_idx_rgb + 2];
        } else {
            dst[idx_rgb + 0] = src[idx_rgb + 0];
            dst[idx_rgb + 1] = src[idx_rgb + 1];
            dst[idx_rgb + 2] = src[idx_rgb + 2];
        }
        return;
    }

    // =========================================================================
    // [V-FINAL3] 동적 임계값 기반 사람 판정
    // =========================================================================
    // 핵심 통찰:
    // 1. 변위 없는 영역: u=x, v=y → mask_at_warped = mask_here
    //    → Void 조건 (mask_at_warped <= th AND mask_here > th) 불가능
    //    → 사람/배경이 올바르게 유지됨
    //
    // 2. 변위 있는 영역: 워핑 포인트(턱, 허리) 주변
    //    → AI 마스크가 부정확할 수 있음 → 번개 발생 가능
    //    → 임계값을 낮춰서 (0.02) 관대하게 사람 판정 → 번개 방지
    //
    // 3. 진짜 Void: 변위 있음 + mask_at_warped < 0.02 + mask_here > 0.1
    //    → 워핑된 좌표가 완전히 배경 → 배경으로 채움
    // =========================================================================

    const float VOID_THRESHOLD = 0.1f;          // Void 판정용 기본 임계값
    const float SAFE_PERSON_THRESHOLD = 0.02f;  // 변위 있을 때 사용 (번개 방지)
    const float DISPLACEMENT_MIN = 0.5f;        // 0.5픽셀 이상이면 "변위 있음"

    // 변위 크기 계산
    float disp_magnitude = sqrtf(shift_x * shift_x + shift_y * shift_y);
    bool has_displacement = (disp_magnitude > DISPLACEMENT_MIN);

    // 마스크 값 조회
    float mask_at_warped = 0.0f;
    if (u >= 0 && u < width && v >= 0 && v < height) {
        mask_at_warped = (float)mask[v * width + u] / 255.0f;
    }
    float mask_here = (float)mask[idx] / 255.0f;

    // 동적 임계값: 변위가 있으면 더 관대하게 사람 판정 (번개 방지)
    float person_threshold = has_displacement ? SAFE_PERSON_THRESHOLD : VOID_THRESHOLD;

    // 워핑 좌표 유효성
    bool warp_in_bounds = (u >= 0 && u < width && v >= 0 && v < height);

    // 3단계 픽셀 결정
    if (warp_in_bounds && mask_at_warped > person_threshold) {
        // =========================================================
        // ★ CASE 1: 사람 영역 → 워핑된 픽셀 출력
        // =========================================================
        // 변위 있는 영역: mask_at_warped > 0.02 (관대하게)
        // 변위 없는 영역: mask_at_warped > 0.1 (기존대로)
        //
        // 변위 없는 영역에서는 u=x, v=y이므로:
        // mask_at_warped = mask_here → 사람은 사람, 배경은 배경으로 유지
        int warped_idx_rgb = (v * width + u) * 3;
        dst[idx_rgb + 0] = src[warped_idx_rgb + 0];
        dst[idx_rgb + 1] = src[warped_idx_rgb + 1];
        dst[idx_rgb + 2] = src[warped_idx_rgb + 2];
    }
    else if (mask_here > VOID_THRESHOLD) {
        // =========================================================
        // ★ CASE 2: Void → 배경으로 채움
        // =========================================================
        // 조건: mask_at_warped <= threshold (사람 아님)
        //       mask_here > 0.1 (원래 사람이었음)
        //
        // 변위 없는 영역에서는 이 조건이 성립할 수 없음:
        // mask_at_warped = mask_here이므로,
        // mask_here > 0.1 → mask_at_warped > 0.1 → CASE 1로 감
        //
        // 오직 변위 있는 영역에서만 Void 발생:
        // 슬리밍으로 사람이 밀려나면서 빈 공간 생김
        dst[idx_rgb + 0] = bg[idx_rgb + 0];
        dst[idx_rgb + 1] = bg[idx_rgb + 1];
        dst[idx_rgb + 2] = bg[idx_rgb + 2];
    }
    else {
        // =========================================================
        // ★ CASE 3: 배경 → 원본 유지
        // =========================================================
        // 조건: mask_at_warped <= threshold AND mask_here <= 0.1
        // 원래 배경이었던 영역 → 현재 좌표(x,y)의 원본 픽셀 유지
        // 배경 왜곡 0%
        dst[idx_rgb + 0] = src[idx_rgb + 0];
        dst[idx_rgb + 1] = src[idx_rgb + 1];
        dst[idx_rgb + 2] = src[idx_rgb + 2];
    }
}
'''


# ==============================================================================
# [KERNEL V48] Forward Warp (순방향 워핑) - 방안 N
# ==============================================================================
# 핵심: 사람 영역만 이동, 배경은 절대 워핑되지 않음
# - 입력 픽셀에서 출력 위치로 이동 (역방향의 반대)
# - depth_buffer로 Z-ordering 처리
# - 구멍은 fill_holes_kernel에서 배경으로 채움
# ==============================================================================
FORWARD_WARP_KERNEL_CODE = r'''
extern "C" __global__
void forward_warp_kernel(
    const unsigned char* src,        // 현재 프레임 (피부보정 적용됨)
    const unsigned char* mask,       // 현재 프레임의 마스크
    unsigned char* dst,              // 출력 (초기화: 0)
    int* depth_buffer,               // Z-buffer (초기화: 0, 정수로 처리)
    const float* dx_small,           // 변위 그리드
    const float* dy_small,
    int width, int height,
    int small_width, int small_height,
    int scale
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float mask_val = (float)mask[idx] / 255.0f;

    // 배경 픽셀은 건너뛰기 (나중에 fill_holes에서 처리)
    // 임계값 0.3: 확실히 사람인 영역만 워핑
    if (mask_val < 0.3f) return;

    // 변위 계산 (small grid에서 보간)
    int sx = x / scale;
    int sy = y / scale;
    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;
    int s_idx = sy * small_width + sx;

    // 변위 부호 반전 (역방향 → 순방향 변환)
    // 역방향: dst[x] = src[x + dx] → 순방향: dst[x - dx] = src[x]
    float shift_x = -dx_small[s_idx] * (float)scale;
    float shift_y = -dy_small[s_idx] * (float)scale;

    // 목적지 좌표 (순방향: 현재 위치 + 반전된 변위)
    int dst_x = (int)roundf((float)x + shift_x);
    int dst_y = (int)roundf((float)y + shift_y);

    // 범위 체크
    if (dst_x < 0 || dst_x >= width || dst_y < 0 || dst_y >= height) return;

    int dst_idx = dst_y * width + dst_x;

    // Z-buffer 체크 (마스크 값이 높은 것 우선)
    // 정수로 변환하여 atomicMax 사용 (0-255 범위)
    int depth_val = (int)(mask_val * 255.0f);

    // atomicMax: 더 높은 마스크 값을 가진 픽셀이 우선
    int old_depth = atomicMax(&depth_buffer[dst_idx], depth_val);

    // 이 픽셀이 최고 우선순위인 경우에만 쓰기
    if (depth_val >= old_depth) {
        int src_rgb = idx * 3;
        int dst_rgb = dst_idx * 3;

        dst[dst_rgb + 0] = src[src_rgb + 0];
        dst[dst_rgb + 1] = src[src_rgb + 1];
        dst[dst_rgb + 2] = src[src_rgb + 2];
    }
}
'''


# ==============================================================================
# [KERNEL V48] Fill Holes (구멍 채우기) - 방안 N
# ==============================================================================
# 핵심: 순방향 워핑 후 빈 영역을 배경으로 채움
# - depth_buffer가 0인 영역 = 구멍 (아무 픽셀도 도달하지 않음)
# - 구멍을 저장된 배경으로 채움
# ==============================================================================
FILL_HOLES_KERNEL_CODE = r'''
extern "C" __global__
void fill_holes_kernel(
    unsigned char* dst,              // 순방향 워핑 결과 (구멍 있음)
    const unsigned char* bg,         // 정적 배경
    const int* depth_buffer,         // Z-buffer (0이면 구멍)
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx_rgb = idx * 3;

    // depth_buffer가 0이면 구멍 → 배경으로 채움
    if (depth_buffer[idx] == 0) {
        dst[idx_rgb + 0] = bg[idx_rgb + 0];
        dst[idx_rgb + 1] = bg[idx_rgb + 1];
        dst[idx_rgb + 2] = bg[idx_rgb + 2];
    }
}
'''


# ==============================================================================
# [KERNEL V48] Forward Warp Splatting (서브픽셀 정확도) - 방안 N 확장
# ==============================================================================
# 핵심: 서브픽셀 위치를 주변 4개 픽셀에 분배
# - 더 부드러운 워핑 결과
# - 구멍 발생 감소
# ==============================================================================
FORWARD_WARP_SPLAT_KERNEL_CODE = r'''
extern "C" __global__
void forward_warp_splat_kernel(
    const unsigned char* src,        // 현재 프레임
    const unsigned char* mask,       // 마스크
    float* dst_accum,                // 누적 버퍼 (float, RGB*3 + weight)
    const float* dx_small,
    const float* dy_small,
    int width, int height,
    int small_width, int small_height,
    int scale
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float mask_val = (float)mask[idx] / 255.0f;

    // 배경 픽셀은 건너뛰기
    if (mask_val < 0.3f) return;

    // 변위 계산
    int sx = x / scale;
    int sy = y / scale;
    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;
    int s_idx = sy * small_width + sx;

    // 변위 부호 반전 (역방향 → 순방향 변환)
    float shift_x = -dx_small[s_idx] * (float)scale;
    float shift_y = -dy_small[s_idx] * (float)scale;

    // 목적지 좌표 (float)
    float dst_xf = (float)x + shift_x;
    float dst_yf = (float)y + shift_y;

    // Splatting: 주변 4개 픽셀에 분배
    int x0 = (int)floorf(dst_xf);
    int y0 = (int)floorf(dst_yf);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = dst_xf - (float)x0;
    float fy = dst_yf - (float)y0;

    // 가중치 계산 (bilinear)
    float w00 = (1.0f - fx) * (1.0f - fy) * mask_val;
    float w10 = fx * (1.0f - fy) * mask_val;
    float w01 = (1.0f - fx) * fy * mask_val;
    float w11 = fx * fy * mask_val;

    // 소스 픽셀
    int src_rgb = idx * 3;
    float r = (float)src[src_rgb + 0];
    float g = (float)src[src_rgb + 1];
    float b = (float)src[src_rgb + 2];

    // 각 목적지에 atomic 누적 (4개 채널: R, G, B, weight)
    if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
        int dst_idx = (y0 * width + x0) * 4;
        atomicAdd(&dst_accum[dst_idx + 0], r * w00);
        atomicAdd(&dst_accum[dst_idx + 1], g * w00);
        atomicAdd(&dst_accum[dst_idx + 2], b * w00);
        atomicAdd(&dst_accum[dst_idx + 3], w00);
    }
    if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
        int dst_idx = (y0 * width + x1) * 4;
        atomicAdd(&dst_accum[dst_idx + 0], r * w10);
        atomicAdd(&dst_accum[dst_idx + 1], g * w10);
        atomicAdd(&dst_accum[dst_idx + 2], b * w10);
        atomicAdd(&dst_accum[dst_idx + 3], w10);
    }
    if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
        int dst_idx = (y1 * width + x0) * 4;
        atomicAdd(&dst_accum[dst_idx + 0], r * w01);
        atomicAdd(&dst_accum[dst_idx + 1], g * w01);
        atomicAdd(&dst_accum[dst_idx + 2], b * w01);
        atomicAdd(&dst_accum[dst_idx + 3], w01);
    }
    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
        int dst_idx = (y1 * width + x1) * 4;
        atomicAdd(&dst_accum[dst_idx + 0], r * w11);
        atomicAdd(&dst_accum[dst_idx + 1], g * w11);
        atomicAdd(&dst_accum[dst_idx + 2], b * w11);
        atomicAdd(&dst_accum[dst_idx + 3], w11);
    }
}
'''


# ==============================================================================
# [KERNEL V48] Normalize Splat (Splatting 정규화) - 방안 N 확장
# ==============================================================================
# 핵심: 누적된 값을 weight로 나누어 최종 픽셀 계산
# - weight가 0이면 구멍 → 배경으로 채움
# ==============================================================================
NORMALIZE_SPLAT_KERNEL_CODE = r'''
extern "C" __global__
void normalize_splat_kernel(
    const float* src_accum,          // 누적 버퍼 (RGBW)
    const unsigned char* bg,         // 배경
    unsigned char* dst,              // 최종 출력
    int width, int height,
    float min_weight                 // 최소 weight (이하면 배경)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int accum_idx = idx * 4;
    int rgb_idx = idx * 3;

    float weight = src_accum[accum_idx + 3];

    if (weight > min_weight) {
        // 정규화: 누적값 / weight
        float inv_w = 1.0f / weight;
        dst[rgb_idx + 0] = (unsigned char)fminf(255.0f, src_accum[accum_idx + 0] * inv_w);
        dst[rgb_idx + 1] = (unsigned char)fminf(255.0f, src_accum[accum_idx + 1] * inv_w);
        dst[rgb_idx + 2] = (unsigned char)fminf(255.0f, src_accum[accum_idx + 2] * inv_w);
    } else {
        // 구멍 → 배경으로 채움
        dst[rgb_idx + 0] = bg[rgb_idx + 0];
        dst[rgb_idx + 1] = bg[rgb_idx + 1];
        dst[rgb_idx + 2] = bg[rgb_idx + 2];
    }
}
'''
