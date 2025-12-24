# Project MUSE - skin_kernels.py
# 피부 보정 관련 CUDA 커널
# (C) 2025 MUSE Corp. All rights reserved.

# ==============================================================================
# [KERNEL 6] Guided Filter (Edge-Aware Smoothing)
# ==============================================================================
GUIDED_FILTER_KERNEL_CODE = r'''
extern "C" __global__
void guided_filter_kernel(
    const unsigned char* src,    // Input image (BGR)
    const unsigned char* guide,  // Guide image (same as src for self-guided)
    unsigned char* dst,          // Output (low-frequency component)
    const unsigned char* mask,   // Skin mask (255=process, 0=skip)
    int width, int height,
    int radius,                  // Filter radius (typically 4-16)
    float epsilon                // Regularization (0.01-0.1, higher = more smoothing)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx3 = idx * 3;

    // Skip non-skin areas
    if (mask[idx] == 0) {
        dst[idx3 + 0] = src[idx3 + 0];
        dst[idx3 + 1] = src[idx3 + 1];
        dst[idx3 + 2] = src[idx3 + 2];
        return;
    }

    // Box filter statistics calculation
    float sum_I_b = 0.0f, sum_I_g = 0.0f, sum_I_r = 0.0f;
    float sum_p_b = 0.0f, sum_p_g = 0.0f, sum_p_r = 0.0f;
    float sum_Ip_b = 0.0f, sum_Ip_g = 0.0f, sum_Ip_r = 0.0f;
    float sum_II_b = 0.0f, sum_II_g = 0.0f, sum_II_r = 0.0f;
    int count = 0;

    // Bounded box filter
    int x_min = max(0, x - radius);
    int x_max = min(width - 1, x + radius);
    int y_min = max(0, y - radius);
    int y_max = min(height - 1, y + radius);

    for (int ny = y_min; ny <= y_max; ny++) {
        for (int nx = x_min; nx <= x_max; nx++) {
            int n_idx3 = (ny * width + nx) * 3;

            float I_b = (float)guide[n_idx3 + 0];
            float I_g = (float)guide[n_idx3 + 1];
            float I_r = (float)guide[n_idx3 + 2];

            float p_b = (float)src[n_idx3 + 0];
            float p_g = (float)src[n_idx3 + 1];
            float p_r = (float)src[n_idx3 + 2];

            sum_I_b += I_b; sum_I_g += I_g; sum_I_r += I_r;
            sum_p_b += p_b; sum_p_g += p_g; sum_p_r += p_r;
            sum_Ip_b += I_b * p_b; sum_Ip_g += I_g * p_g; sum_Ip_r += I_r * p_r;
            sum_II_b += I_b * I_b; sum_II_g += I_g * I_g; sum_II_r += I_r * I_r;
            count++;
        }
    }

    float inv_count = 1.0f / (float)count;

    // Mean values
    float mean_I_b = sum_I_b * inv_count;
    float mean_I_g = sum_I_g * inv_count;
    float mean_I_r = sum_I_r * inv_count;

    float mean_p_b = sum_p_b * inv_count;
    float mean_p_g = sum_p_g * inv_count;
    float mean_p_r = sum_p_r * inv_count;

    // Covariance and variance
    float cov_Ip_b = sum_Ip_b * inv_count - mean_I_b * mean_p_b;
    float cov_Ip_g = sum_Ip_g * inv_count - mean_I_g * mean_p_g;
    float cov_Ip_r = sum_Ip_r * inv_count - mean_I_r * mean_p_r;

    float var_I_b = sum_II_b * inv_count - mean_I_b * mean_I_b;
    float var_I_g = sum_II_g * inv_count - mean_I_g * mean_I_g;
    float var_I_r = sum_II_r * inv_count - mean_I_r * mean_I_r;

    // Guided filter coefficients
    float eps_scaled = epsilon * 255.0f * 255.0f;

    float a_b = cov_Ip_b / (var_I_b + eps_scaled);
    float a_g = cov_Ip_g / (var_I_g + eps_scaled);
    float a_r = cov_Ip_r / (var_I_r + eps_scaled);

    float b_b = mean_p_b - a_b * mean_I_b;
    float b_g = mean_p_g - a_g * mean_I_g;
    float b_r = mean_p_r - a_r * mean_I_r;

    // Output: q = a * I + b
    float I_b = (float)guide[idx3 + 0];
    float I_g = (float)guide[idx3 + 1];
    float I_r = (float)guide[idx3 + 2];

    float q_b = a_b * I_b + b_b;
    float q_g = a_g * I_g + b_g;
    float q_r = a_r * I_r + b_r;

    // Clamp to valid range
    dst[idx3 + 0] = (unsigned char)fminf(fmaxf(q_b, 0.0f), 255.0f);
    dst[idx3 + 1] = (unsigned char)fminf(fmaxf(q_g, 0.0f), 255.0f);
    dst[idx3 + 2] = (unsigned char)fminf(fmaxf(q_r, 0.0f), 255.0f);
}
'''


# ==============================================================================
# [KERNEL 7] Fast Skin Smooth Blend
# ==============================================================================
FAST_SKIN_SMOOTH_KERNEL_CODE = r'''
extern "C" __global__
void fast_skin_smooth_kernel(
    const unsigned char* src,          // Input image (BGR)
    const unsigned char* smoothed,     // Pre-blurred image (BGR)
    const unsigned char* mask,         // Skin mask (0-255)
    unsigned char* dst,                // Output image (BGR)
    int width, int height,
    float detail_preserve,             // 0.0 = full blur, 1.0 = no blur
    float blend_strength               // Overall blend strength (0-1)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx3 = idx * 3;

    float mask_val = (float)mask[idx] / 255.0f * blend_strength;

    if (mask_val < 0.01f) {
        dst[idx3 + 0] = src[idx3 + 0];
        dst[idx3 + 1] = src[idx3 + 1];
        dst[idx3 + 2] = src[idx3 + 2];
        return;
    }

    for (int c = 0; c < 3; c++) {
        float orig = (float)src[idx3 + c];
        float smooth = (float)smoothed[idx3 + c];
        float detail = orig - smooth;
        float result = smooth + detail * detail_preserve;
        float final_val = orig * (1.0f - mask_val) + result * mask_val;
        dst[idx3 + c] = (unsigned char)fminf(fmaxf(final_val, 0.0f), 255.0f);
    }
}
'''


# ==============================================================================
# [KERNEL 31] Dual-Pass Smooth Kernel (Wide/Fine 합성)
# ==============================================================================
DUAL_PASS_SMOOTH_KERNEL_CODE = r'''
extern "C" __global__
void dual_pass_smooth_kernel(
    const unsigned char* src,          // Original image (BGR)
    const unsigned char* wide_smooth,  // Wide Pass result (radius 15) - 저주파
    const unsigned char* fine_smooth,  // Fine Pass result (radius 5) - 미세 디테일
    const unsigned char* mask,         // Skin mask (0-255)
    unsigned char* dst,                // Output image (BGR)
    int width, int height,
    float skin_strength,               // 피부 보정 강도 (0.0~1.0)
    float blend_strength               // 전체 블렌드 강도 (0.0~1.0)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx3 = idx * 3;

    // 마스크 값 (0~1 범위로 정규화)
    float mask_val = (float)mask[idx] / 255.0f * blend_strength;

    // 마스크 외부: 원본 복사
    if (mask_val < 0.01f) {
        dst[idx3 + 0] = src[idx3 + 0];
        dst[idx3 + 1] = src[idx3 + 1];
        dst[idx3 + 2] = src[idx3 + 2];
        return;
    }

    // === 원본 BGR 읽기 ===
    float orig_b = (float)src[idx3 + 0];
    float orig_g = (float)src[idx3 + 1];
    float orig_r = (float)src[idx3 + 2];

    // === Wide Pass (저주파 - 피부 전체 균일화) ===
    float wide_b = (float)wide_smooth[idx3 + 0];
    float wide_g = (float)wide_smooth[idx3 + 1];
    float wide_r = (float)wide_smooth[idx3 + 2];

    // === Fine Pass (미세 디테일 보존) ===
    float fine_b = (float)fine_smooth[idx3 + 0];
    float fine_g = (float)fine_smooth[idx3 + 1];
    float fine_r = (float)fine_smooth[idx3 + 2];

    // === 채널별 고주파 디테일 추출 ===
    float detail_fine_b = orig_b - fine_b;
    float detail_fine_g = orig_g - fine_g;
    float detail_fine_r = orig_r - fine_r;

    // === 휘도 기반 엣지 검출 ===
    float detail_mag = fabsf(detail_fine_r * 0.299f + detail_fine_g * 0.587f + detail_fine_b * 0.114f);
    float edge_threshold = 12.0f;

    // === [V31] 비선형 디테일 곡선: powf(edge_factor, 0.3f) ===
    float edge_factor = fminf(detail_mag / edge_threshold, 1.0f);
    float nonlinear_edge = powf(edge_factor, 0.3f);

    // === Dual-Pass 합성 전략 ===
    float wide_weight = 1.0f - nonlinear_edge * 0.7f;
    float fine_weight = nonlinear_edge * 0.7f;

    float base_b = wide_b * wide_weight + fine_b * fine_weight;
    float base_g = wide_g * wide_weight + fine_g * fine_weight;
    float base_r = wide_r * wide_weight + fine_r * fine_weight;

    // 디테일 복원량
    float detail_preserve = 0.3f + (1.0f - skin_strength) * 0.5f + nonlinear_edge * 0.2f;
    detail_preserve = fminf(detail_preserve, 1.0f);

    // 최종 스무딩 결과
    float result_b = base_b + detail_fine_b * detail_preserve;
    float result_g = base_g + detail_fine_g * detail_preserve;
    float result_r = base_r + detail_fine_r * detail_preserve;

    // === [V31] Contrast 부스팅 1.05f ===
    float contrast = 1.05f;
    float mid = 128.0f;

    result_b = (result_b - mid) * contrast + mid;
    result_g = (result_g - mid) * contrast + mid;
    result_r = (result_r - mid) * contrast + mid;

    // 클리핑
    result_b = fmaxf(0.0f, fminf(255.0f, result_b));
    result_g = fmaxf(0.0f, fminf(255.0f, result_g));
    result_r = fmaxf(0.0f, fminf(255.0f, result_r));

    // === 마스크 기반 최종 블렌드 ===
    float final_b = orig_b * (1.0f - mask_val) + result_b * mask_val;
    float final_g = orig_g * (1.0f - mask_val) + result_g * mask_val;
    float final_r = orig_r * (1.0f - mask_val) + result_r * mask_val;

    // 반올림하여 저장
    dst[idx3 + 0] = (unsigned char)(final_b + 0.5f);
    dst[idx3 + 1] = (unsigned char)(final_g + 0.5f);
    dst[idx3 + 2] = (unsigned char)(final_r + 0.5f);
}
'''
