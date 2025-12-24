# Project MUSE - composite_kernels.py
# 합성 관련 CUDA 커널
# (C) 2025 MUSE Corp. All rights reserved.

# ==============================================================================
# [KERNEL V44] Simple Void Fill (Frame-Independent)
# ==============================================================================
# 핵심: 프레임과 마스크가 100% 동일 시점 (시간 불일치 불가능)
# 단순 3단계 If-Else로 번개/잔상/왜곡 동시 해결
# ==============================================================================
SIMPLE_VOID_FILL_KERNEL_CODE = r'''
extern "C" __global__
void simple_void_fill_kernel(
    const unsigned char* src,        // 현재 프레임 (피부보정 적용됨)
    const unsigned char* bg,         // 정적 배경 (Clean Plate)
    const unsigned char* mask,       // 현재 프레임의 마스크 (동일 시점)
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
    // [V44.4] Hybrid Void Detection
    // =========================================================================
    // 핵심: mask_at_warped 값에 따라 3단계 분류
    // - 확실히 배경 (< 0.02): 경계 체크 없이 바로 배경 (Void 채움)
    // - 애매함 (0.02 ~ 0.10): 안전하게 전경 (번개 방지)
    // - 확실히 사람 (> 0.10): 무조건 전경
    // =========================================================================

    // 현재 좌표의 마스크
    float mask_here = (float)mask[idx] / 255.0f;

    // 워핑된 좌표의 마스크
    float mask_at_warped = 0.0f;
    if (u >= 0 && u < width && v >= 0 && v < height) {
        mask_at_warped = (float)mask[v * width + u] / 255.0f;
    }

    // 워핑 변위 (슬리밍 영향 확인)
    float warp_magnitude = sqrtf(shift_x * shift_x + shift_y * shift_y);
    bool is_slimming_edge = (warp_magnitude > 1.0f);

    // 픽셀 준비
    float fg_b, fg_g, fg_r;
    if (u >= 0 && u < width && v >= 0 && v < height) {
        int warped_idx_rgb = (v * width + u) * 3;
        fg_b = (float)src[warped_idx_rgb + 0];
        fg_g = (float)src[warped_idx_rgb + 1];
        fg_r = (float)src[warped_idx_rgb + 2];
    } else {
        fg_b = (float)src[idx_rgb + 0];
        fg_g = (float)src[idx_rgb + 1];
        fg_r = (float)src[idx_rgb + 2];
    }

    float bg_b = (float)bg[idx_rgb + 0];
    float bg_g = (float)bg[idx_rgb + 1];
    float bg_r = (float)bg[idx_rgb + 2];

    // 임계값 정의
    // [V47] 번개 현상 방어를 위해 임계값 대폭 낮춤
    const float CERTAIN_BG = 0.02f;       // [V47] 0.05 → 0.02 (진짜 완전한 배경만)
    const float CERTAIN_PERSON = 0.10f;   // [V47] 0.3 → 0.10 (10%만 넘어도 사람)
    const float ORIGIN_THRESHOLD = 0.10f; // [V47] 0.2 → 0.10 (원래 사람이었는지)

    float out_b, out_g, out_r;

    if (mask_at_warped > CERTAIN_PERSON) {
        // ★ CASE 1: 워핑된 위치가 확실히 사람 (10% 초과)
        // → 무조건 전경 출력 (번개 현상 차단)
        out_b = fg_b;
        out_g = fg_g;
        out_r = fg_r;
    }
    else if (mask_at_warped < CERTAIN_BG && mask_here > ORIGIN_THRESHOLD && is_slimming_edge) {
        // ★ CASE 2: 진짜 Void!
        // 조건 1: 워핑된 위치가 확실히 배경 (2% 미만)
        // 조건 2: 원래 이 자리에 사람이 있었음 (10% 초과)
        // 조건 3: 슬리밍으로 인한 변위 있음
        // → 경계 체크 불필요! 바로 배경 출력
        out_b = bg_b;
        out_g = bg_g;
        out_r = bg_r;
    }
    else {
        // ★ CASE 3: 그 외 모든 경우
        // - mask_at_warped = 0.02 ~ 0.10 (애매함) → 안전하게 전경
        // - AI 마스크 결함 → 전경으로 번개 방지
        // - 원래 배경 (mask_here < 0.10) → 전경 (워핑된 원본)
        out_b = fg_b;
        out_g = fg_g;
        out_r = fg_r;
    }

    dst[idx_rgb + 0] = (unsigned char)out_b;
    dst[idx_rgb + 1] = (unsigned char)out_g;
    dst[idx_rgb + 2] = (unsigned char)out_r;
}
'''
