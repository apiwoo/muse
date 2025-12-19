# Project MUSE - cuda_kernels.py
# High-Fidelity Kernels (Alpha Blending & TPS Warping)
# V25.0: High-Precision Pipeline & Frequency Separation
# - Added: Polygon Mask Generation (Scanline Algorithm)
# - Added: Fast Guided Filter (Edge-Aware Smoothing)
# - Added: Tone Uniformity (Flat-fielding)
# - Added: Color Grading (HSL Temperature/Tint)
# (C) 2025 MUSE Corp. All rights reserved.

# ==============================================================================
# [KERNEL 1] Grid Generation (TPS Logic) - 기존 100% 유지
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
# - 인물 마스크(Alpha)를 dx, dy 그리드에 곱하여 배경 영역 변위를 0으로 고정
# - 배경 왜곡(Bending) 원천 차단
# - 마스크 경계면 블러링 필요 (호출 전 적용)
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
    // alpha = 1.0 (사람 영역): 변위 유지
    // alpha = 0.0 (배경 영역): 변위 = 0 (워핑 차단)
    dx[idx] *= alpha;
    dy[idx] *= alpha;
}
'''

# ==============================================================================
# [KERNEL 2] Smart Composite (Intrusion Handling) - 기존 100% 유지
# ==============================================================================
COMPOSITE_KERNEL_CODE = r'''
extern "C" __global__
void composite_kernel(
    const unsigned char* src,
    const unsigned char* mask, // Alpha Matte (0~255)
    const unsigned char* bg,   // Static Clean Plate
    unsigned char* dst,
    const float* dx_small,
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

    // --- 1. Original Alpha (Before) ---
    float original_alpha = 0.0f;
    if (use_bg) {
        original_alpha = (float)mask[idx] / 255.0f;
    }

    // --- 2. Warping & Displacement ---
    int sx = x / scale;
    int sy = y / scale;
    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;
    int s_idx = sy * small_width + sx;

    float shift_x = dx_small[s_idx] * (float)scale;
    float shift_y = dy_small[s_idx] * (float)scale;

    float shift_sq = shift_x * shift_x + shift_y * shift_y;
    bool is_significant_warp = (shift_sq > 1.0f);

    int u = (int)(x + shift_x);
    int v = (int)(y + shift_y);

    float warped_alpha = 0.0f;
    float fg_b = 0.0f, fg_g = 0.0f, fg_r = 0.0f;

    if (u >= 0 && u < width && v >= 0 && v < height) {
        int warped_idx = v * width + u;
        int warped_idx_rgb = warped_idx * 3;

        if (use_bg) {
            warped_alpha = (float)mask[warped_idx] / 255.0f;
        } else {
            warped_alpha = 1.0f;
        }

        if (warped_alpha > 0.0f) {
            fg_b = (float)src[warped_idx_rgb+0];
            fg_g = (float)src[warped_idx_rgb+1];
            fg_r = (float)src[warped_idx_rgb+2];
        }
    }

    // --- 3. Final Decision (Context-Aware Composite) - V2 Interior Protection ---

    // [V2] 1. Protect deep interior of mask
    // High original_alpha = definite person region -> block background
    // 임계값 0.7 이상이면 "내부"로 간주
    bool is_interior = (original_alpha > 0.7f);

    // [V2] 2. 마스크 경계 검출 (3x3 이웃 검사)
    // 주변에 알파가 낮은 픽셀이 있으면 "경계"로 간주
    float min_neighbor_alpha = original_alpha;
    float max_neighbor_alpha = original_alpha;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float neighbor_alpha = (float)mask[ny * width + nx] / 255.0f;
                min_neighbor_alpha = fminf(min_neighbor_alpha, neighbor_alpha);
                max_neighbor_alpha = fmaxf(max_neighbor_alpha, neighbor_alpha);
            }
        }
    }

    // 경계 판정: 주변에 알파 차이가 0.3 이상인 픽셀이 있으면 경계
    bool is_near_edge = (max_neighbor_alpha - min_neighbor_alpha > 0.3f);

    // [V2] 3. 워핑 좌표 유효성 검사 강화
    // 워핑 좌표가 이미지 범위 내인지 확인
    bool warped_in_bounds = (u >= 0 && u < width && v >= 0 && v < height);

    // [V2] 4. 배경 합성 조건 - 훨씬 엄격하게
    // 기존: is_alpha_loss && is_significant_warp
    // 변경: 경계 근처이고 + 워핑 좌표가 유효하고 + 내부가 아닐 때만 배경 합성
    bool is_alpha_loss = (original_alpha > warped_alpha + 0.1f);  // threshold up (0.05 -> 0.1)

    // 배경 합성 조건 (모두 만족해야 함):
    // 1) is_near_edge: 마스크 경계 근처일 것
    // 2) !is_interior: 마스크 내부 깊숙한 곳이 아닐 것
    // 3) is_alpha_loss: 알파 손실이 감지될 것
    // 4) is_significant_warp: 워핑 변위가 충분할 것
    // 5) warped_in_bounds: 워핑 좌표가 이미지 범위 내일 것
    bool should_use_bg = is_near_edge &&
                         !is_interior &&
                         is_alpha_loss &&
                         is_significant_warp &&
                         warped_in_bounds;

    float base_b, base_g, base_r;

    if (should_use_bg) {
        // [Case A] Edge region with slimming effect -> use background
        base_b = (float)bg[idx_rgb+0];
        base_g = (float)bg[idx_rgb+1];
        base_r = (float)bg[idx_rgb+2];
    } else {
        // [Case B] Interior or non-edge -> keep original (preserve correction)
        base_b = (float)src[idx_rgb+0];
        base_g = (float)src[idx_rgb+1];
        base_r = (float)src[idx_rgb+2];
    }

    // --- 4. Final Composite ---
    dst[idx_rgb+0] = (unsigned char)(fg_b * warped_alpha + base_b * (1.0f - warped_alpha));
    dst[idx_rgb+1] = (unsigned char)(fg_g * warped_alpha + base_g * (1.0f - warped_alpha));
    dst[idx_rgb+2] = (unsigned char)(fg_r * warped_alpha + base_r * (1.0f - warped_alpha));
}
'''

# ==============================================================================
# [KERNEL 3] Skin Smooth (Tone Blending) - 기존 100% 유지 (레거시 호환)
# ==============================================================================
SKIN_SMOOTH_KERNEL_CODE = r'''
extern "C" __global__
void skin_smooth_kernel(
    const unsigned char* src,
    unsigned char* dst,
    int width, int height,
    float strength,
    float face_cx, float face_cy, float face_rad,
    float target_r, float target_g, float target_b,
    float tone_val, // -1.0(White) ~ 1.0(Pink)
    const float* exclusion_params // Array: [x,y,r] * 5 zones
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    float center_b = (float)src[idx+0]; // BGR
    float center_g = (float)src[idx+1];
    float center_r = (float)src[idx+2];

    // 1. Face Region Check
    float dist_face_sq = (x - face_cx)*(x - face_cx) + (y - face_cy)*(y - face_cy);
    bool is_inside_face = (dist_face_sq < (face_rad * face_rad));

    // 2. Exclusion Zone Check (Tight Protection)
    bool is_protected = false;
    if (is_inside_face) {
        for(int i=0; i<5; i++) {
            int base = i * 3;
            float ex = exclusion_params[base + 0];
            float ey = exclusion_params[base + 1];
            float er = exclusion_params[base + 2];

            float d_sq = (x - ex)*(x - ex) + (y - ey)*(y - ey);
            if (d_sq < (er * er)) {
                is_protected = true;
                break;
            }
        }
    }

    // 3. Color Similarity
    float color_dist = sqrtf(
        powf(center_r - target_r, 2) +
        powf(center_g - target_g, 2) +
        powf(center_b - target_b, 2)
    );

    // 4. Determine Sigma (Strength)
    float sigma_color;
    if (is_protected) {
        sigma_color = 0.01f; // Preserve Details
    } else if (is_inside_face && color_dist < 80.0f) {
        sigma_color = 60.0f; // Skin: Strong Smooth
    } else {
        sigma_color = 10.0f; // Background
    }

    // Bilateral-like Filter Logic
    int radius = 6;
    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    float total_w = 0.0f;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = (ny * width + nx) * 3;
                float nb = (float)src[n_idx+0];
                float ng = (float)src[n_idx+1];
                float nr = (float)src[n_idx+2];

                float diff = fabsf(nr - center_r) + fabsf(ng - center_g) + fabsf(nb - center_b);
                float weight = expf(-diff / sigma_color);

                sum_b += nb * weight;
                sum_g += ng * weight;
                sum_r += nr * weight;
                total_w += weight;
            }
        }
    }

    float smooth_b = sum_b / total_w;
    float smooth_g = sum_g / total_w;
    float smooth_r = sum_r / total_w;

    // 5. Tone Correction (Bipolar Mixing)
    if (is_inside_face && !is_protected && color_dist < 80.0f && fabsf(tone_val) > 0.05f) {
        float mix_factor;
        float tr, tg, tb;

        if (tone_val > 0.0f) {
            // Rosy (Pinkish) Target: (255, 215, 225)
            tr = 255.0f; tg = 215.0f; tb = 225.0f;
            mix_factor = tone_val * 0.4f; // Max 40% mix
        } else {
            // Whitening (Pale) Target: (255, 255, 255) -> Increase Brightness
            tr = 255.0f; tg = 255.0f; tb = 255.0f;
            mix_factor = -tone_val * 0.3f; // Max 30% mix
        }

        smooth_r = smooth_r * (1.0f - mix_factor) + tr * mix_factor;
        smooth_g = smooth_g * (1.0f - mix_factor) + tg * mix_factor;
        smooth_b = smooth_b * (1.0f - mix_factor) + tb * mix_factor;
    }

    // Only apply if strong enough
    float effective_strength = is_protected ? 0.0f : strength;

    dst[idx+0] = (unsigned char)(center_b * (1.0f - effective_strength) + smooth_b * effective_strength);
    dst[idx+1] = (unsigned char)(center_g * (1.0f - effective_strength) + smooth_g * effective_strength);
    dst[idx+2] = (unsigned char)(center_r * (1.0f - effective_strength) + smooth_r * effective_strength);
}
'''

# ==============================================================================
# [V25.0 NEW KERNELS START HERE]
# ==============================================================================

# ==============================================================================
# [KERNEL 6] Fast Guided Filter (Edge-Preserving Smoothing)
# - Box Filter 기반 O(1) 복잡도 근사
# - 주파수 분리의 핵심: Low-frequency 추출
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
    // For each channel: mean_I, mean_p, mean_Ip, var_I

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

    // Guided filter coefficients: a = cov / (var + eps), b = mean_p - a * mean_I
    // Scale epsilon by 255^2 for 8-bit images
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
# [KERNEL 12-B] High-Pass Detail Preserve (V29 - 선명한 매끈함)
# - 핵심 원리: 스무딩된 이미지 + 원본 고주파 디테일
# - 피부 텍스처(모공, 잔주름)는 스무딩하되 윤곽선(눈, 코, 입)은 선명 유지
# - LAB 변환 대신 간단한 휘도 기반 디테일 추출 → 안정적이고 빠름
# ==============================================================================
LAB_SKIN_SMOOTH_KERNEL_CODE = r'''
extern "C" __global__
void lab_skin_smooth_kernel(
    const unsigned char* src,          // Original image (BGR)
    const unsigned char* smoothed,     // Guided Filter result (BGR) - 저주파(피부색)
    const unsigned char* mask,         // Skin mask (0-255)
    unsigned char* dst,                // Output image (BGR)
    int width, int height,
    float detail_strength,             // 디테일 복원 강도 (0.0~1.0) - 낮을수록 더 스무딩
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

    // === 스무딩된 BGR 읽기 (Guided Filter 결과 = 저주파/피부색) ===
    float smooth_b = (float)smoothed[idx3 + 0];
    float smooth_g = (float)smoothed[idx3 + 1];
    float smooth_r = (float)smoothed[idx3 + 2];

    // === 채널별 고주파 디테일 추출 ===
    // detail = original - smoothed (양수=밝은점, 음수=어두운점)
    float detail_b = orig_b - smooth_b;
    float detail_g = orig_g - smooth_g;
    float detail_r = orig_r - smooth_r;

    // === [V30] 적응형 디테일 보존 ===
    // 휘도(Luminance) 기반 디테일 크기 계산
    float detail_mag = fabsf(detail_r * 0.299f + detail_g * 0.587f + detail_b * 0.114f);

    // 엣지 임계값: 이 값 이상의 디테일은 윤곽선(눈코입)으로 간주
    float edge_threshold = 15.0f;

    // 적응형 디테일 강도: 엣지 영역은 더 많이 보존
    // detail_mag가 edge_threshold 이상이면 디테일을 100% 유지
    // detail_mag가 0이면 detail_strength 값 그대로 사용
    float edge_factor = fminf(detail_mag / edge_threshold, 1.0f);
    float adaptive_detail = detail_strength + (1.0f - detail_strength) * edge_factor;

    // === Skin texture smoothing + edge detail preservation ===
    // Low detail_mag (skin): apply detail_strength -> smoothing
    // High detail_mag (edges): adaptive_detail -> preserve original
    float result_b = smooth_b + detail_b * adaptive_detail;
    float result_g = smooth_g + detail_g * adaptive_detail;
    float result_r = smooth_r + detail_r * adaptive_detail;

    // 클리핑 (0~255 범위 유지)
    result_b = fmaxf(0.0f, fminf(255.0f, result_b));
    result_g = fmaxf(0.0f, fminf(255.0f, result_g));
    result_r = fmaxf(0.0f, fminf(255.0f, result_r));

    // === 마스크 기반 최종 블렌드 ===
    // 마스크 영역은 스무딩 결과, 마스크 외부는 원본
    float final_b = orig_b * (1.0f - mask_val) + result_b * mask_val;
    float final_g = orig_g * (1.0f - mask_val) + result_g * mask_val;
    float final_r = orig_r * (1.0f - mask_val) + result_r * mask_val;

    // 반올림하여 저장
    dst[idx3 + 0] = (unsigned char)(final_b + 0.5f);
    dst[idx3 + 1] = (unsigned char)(final_g + 0.5f);
    dst[idx3 + 2] = (unsigned char)(final_r + 0.5f);
}
'''


# ==============================================================================
# [KERNEL 12-1] V31: Dual-Pass Smooth (Wide/Fine 합성 + 비선형 디테일 곡선)
# - Wide Pass(넓은 스무딩)와 Fine Pass(좁은 스무딩) 결과를 합성
# - powf(edge_factor, 0.3f) 비선형 곡선으로 엣지 복원력 기하급수적 강화
# - Contrast 부스팅 1.05f로 투명감 있는 피부 표현
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
    // Wide에서의 디테일: 원본 - Wide (큰 디테일)
    float detail_wide_b = orig_b - wide_b;
    float detail_wide_g = orig_g - wide_g;
    float detail_wide_r = orig_r - wide_r;

    // Fine에서의 디테일: 원본 - Fine (미세 디테일)
    float detail_fine_b = orig_b - fine_b;
    float detail_fine_g = orig_g - fine_g;
    float detail_fine_r = orig_r - fine_r;

    // === 휘도 기반 엣지 검출 ===
    // Fine에서 추출한 디테일 크기 (미세 엣지일수록 큼)
    float detail_mag = fabsf(detail_fine_r * 0.299f + detail_fine_g * 0.587f + detail_fine_b * 0.114f);

    // 엣지 임계값
    float edge_threshold = 12.0f;

    // === [V31] 비선형 디테일 곡선: powf(edge_factor, 0.3f) ===
    // 기존: edge_factor 그대로 사용 (선형)
    // V31: pow(0.3)로 작은 엣지도 기하급수적으로 증폭
    // - edge_factor 0.1 -> pow(0.1, 0.3) ~= 0.50 (5x boost)
    // - edge_factor 0.5 -> pow(0.5, 0.3) ~= 0.81 (1.6x boost)
    // - edge_factor 1.0 -> pow(1.0, 0.3) = 1.0 (unchanged)
    float edge_factor = fminf(detail_mag / edge_threshold, 1.0f);
    float nonlinear_edge = powf(edge_factor, 0.3f);

    // === Dual-Pass 합성 전략 ===
    // - 피부 영역(낮은 edge_factor): Wide 결과 주로 사용 (도자기 효과)
    // - 엣지 영역(높은 edge_factor): Fine 결과 + 원본 디테일 (선명도 유지)

    // Wide와 Fine의 블렌딩 비율 (엣지에서는 Fine 위주)
    float wide_weight = 1.0f - nonlinear_edge * 0.7f;  // 피부: 1.0, 엣지: 0.3
    float fine_weight = nonlinear_edge * 0.7f;          // 피부: 0.0, 엣지: 0.7

    // 기본 스무딩 결과 (Wide + Fine 혼합)
    float base_b = wide_b * wide_weight + fine_b * fine_weight;
    float base_g = wide_g * wide_weight + fine_g * fine_weight;
    float base_r = wide_r * wide_weight + fine_r * fine_weight;

    // 디테일 복원량 (비선형 곡선 적용)
    // skin_strength가 높을수록 디테일 복원 줄임 (스무딩 강화)
    float detail_preserve = 0.3f + (1.0f - skin_strength) * 0.5f + nonlinear_edge * 0.2f;
    detail_preserve = fminf(detail_preserve, 1.0f);

    // 최종 스무딩 결과 = 기본 + 디테일 복원
    float result_b = base_b + detail_fine_b * detail_preserve;
    float result_g = base_g + detail_fine_g * detail_preserve;
    float result_r = base_r + detail_fine_r * detail_preserve;

    // === [V31] Contrast 부스팅 1.05f ===
    // 피부에 투명감과 입체감 부여
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


# ==============================================================================
# [KERNEL 13] YY-Style Bilateral Filter (Surface Smoothing)
# - 색상 유사한 픽셀만 스무딩 → 윤곽선 자동 보존
# - GPU 최적화: 다운스케일된 이미지에서 처리
# - 마스크 기반 처리로 피부 영역만 스무딩
# ==============================================================================
BILATERAL_SMOOTH_KERNEL_CODE = r'''
extern "C" __global__
void bilateral_smooth_kernel(
    const unsigned char* src,          // Input image (BGR)
    unsigned char* dst,                // Output image (BGR)
    const unsigned char* mask,         // Skin mask (0-255, gradation supported)
    int width, int height,
    float sigma_spatial,               // Spatial sigma (smoothing range, 10-20)
    float sigma_color                  // Color sigma (edge preservation, 25-40)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx3 = idx * 3;

    // Get mask value (0-255 gradation)
    float mask_val = (float)mask[idx] / 255.0f;

    // If mask is very low, just copy original
    if (mask_val < 0.01f) {
        dst[idx3 + 0] = src[idx3 + 0];
        dst[idx3 + 1] = src[idx3 + 1];
        dst[idx3 + 2] = src[idx3 + 2];
        return;
    }

    // Center pixel color
    float center_b = (float)src[idx3 + 0];
    float center_g = (float)src[idx3 + 1];
    float center_r = (float)src[idx3 + 2];

    // Calculate filter radius from sigma_spatial
    int radius = (int)(sigma_spatial * 2.0f);
    if (radius < 1) radius = 1;
    if (radius > 15) radius = 15;  // Cap for performance

    // Pre-calculate spatial Gaussian denominator
    float spatial_denom = 2.0f * sigma_spatial * sigma_spatial;
    float color_denom = 2.0f * sigma_color * sigma_color;

    // Bilateral filter accumulation
    float sum_b = 0.0f, sum_g = 0.0f, sum_r = 0.0f;
    float total_weight = 0.0f;

    for (int dy = -radius; dy <= radius; dy++) {
        int ny = y + dy;
        if (ny < 0 || ny >= height) continue;

        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            if (nx < 0 || nx >= width) continue;

            int n_idx = ny * width + nx;
            int n_idx3 = n_idx * 3;

            // Neighbor pixel color
            float nb = (float)src[n_idx3 + 0];
            float ng = (float)src[n_idx3 + 1];
            float nr = (float)src[n_idx3 + 2];

            // Spatial weight (Gaussian based on distance)
            float spatial_dist_sq = (float)(dx * dx + dy * dy);
            float spatial_weight = expf(-spatial_dist_sq / spatial_denom);

            // Color weight (Gaussian based on color difference)
            // This is the KEY: similar colors get high weight, different colors get low weight
            float color_diff_sq = (nb - center_b) * (nb - center_b)
                                + (ng - center_g) * (ng - center_g)
                                + (nr - center_r) * (nr - center_r);
            float color_weight = expf(-color_diff_sq / color_denom);

            // Combined bilateral weight
            float weight = spatial_weight * color_weight;

            sum_b += nb * weight;
            sum_g += ng * weight;
            sum_r += nr * weight;
            total_weight += weight;
        }
    }

    // Normalize
    float smooth_b, smooth_g, smooth_r;
    if (total_weight > 0.0001f) {
        smooth_b = sum_b / total_weight;
        smooth_g = sum_g / total_weight;
        smooth_r = sum_r / total_weight;
    } else {
        smooth_b = center_b;
        smooth_g = center_g;
        smooth_r = center_r;
    }

    // Blend with original based on mask value (gradation blending)
    float out_b = center_b * (1.0f - mask_val) + smooth_b * mask_val;
    float out_g = center_g * (1.0f - mask_val) + smooth_g * mask_val;
    float out_r = center_r * (1.0f - mask_val) + smooth_r * mask_val;

    // Output
    dst[idx3 + 0] = (unsigned char)fminf(fmaxf(out_b, 0.0f), 255.0f);
    dst[idx3 + 1] = (unsigned char)fminf(fmaxf(out_g, 0.0f), 255.0f);
    dst[idx3 + 2] = (unsigned char)fminf(fmaxf(out_r, 0.0f), 255.0f);
}
'''


# ==============================================================================
# [KERNEL 14] GPU Resize (Bilinear Interpolation)
# - GPU에서 다운스케일/업스케일 처리
# - CPU-GPU 전송 없이 빠른 리사이즈
# ==============================================================================
GPU_RESIZE_KERNEL_CODE = r'''
extern "C" __global__
void gpu_resize_kernel(
    const unsigned char* src,          // Input image (BGR)
    unsigned char* dst,                // Output image (BGR)
    int src_width, int src_height,
    int dst_width, int dst_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height) return;

    // Calculate source position (bilinear interpolation)
    float scale_x = (float)src_width / (float)dst_width;
    float scale_y = (float)src_height / (float)dst_height;

    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    // Clamp to valid range
    if (src_x < 0) src_x = 0;
    if (src_y < 0) src_y = 0;
    if (src_x > src_width - 1) src_x = src_width - 1;
    if (src_y > src_height - 1) src_y = src_height - 1;

    // Get integer and fractional parts
    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    float fx = src_x - x0;
    float fy = src_y - y0;

    // Bilinear interpolation for each channel
    int dst_idx = (y * dst_width + x) * 3;

    for (int c = 0; c < 3; c++) {
        float v00 = (float)src[(y0 * src_width + x0) * 3 + c];
        float v01 = (float)src[(y0 * src_width + x1) * 3 + c];
        float v10 = (float)src[(y1 * src_width + x0) * 3 + c];
        float v11 = (float)src[(y1 * src_width + x1) * 3 + c];

        float v0 = v00 * (1.0f - fx) + v01 * fx;
        float v1 = v10 * (1.0f - fx) + v11 * fx;
        float v = v0 * (1.0f - fy) + v1 * fy;

        dst[dst_idx + c] = (unsigned char)fminf(fmaxf(v, 0.0f), 255.0f);
    }
}
'''


# ==============================================================================
# [KERNEL 15] GPU Mask Resize (Single Channel)
# - 마스크 전용 리사이즈 (1채널)
# ==============================================================================
GPU_MASK_RESIZE_KERNEL_CODE = r'''
extern "C" __global__
void gpu_mask_resize_kernel(
    const unsigned char* src,          // Input mask
    unsigned char* dst,                // Output mask
    int src_width, int src_height,
    int dst_width, int dst_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height) return;

    float scale_x = (float)src_width / (float)dst_width;
    float scale_y = (float)src_height / (float)dst_height;

    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    if (src_x < 0) src_x = 0;
    if (src_y < 0) src_y = 0;
    if (src_x > src_width - 1) src_x = src_width - 1;
    if (src_y > src_height - 1) src_y = src_height - 1;

    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    float fx = src_x - x0;
    float fy = src_y - y0;

    float v00 = (float)src[y0 * src_width + x0];
    float v01 = (float)src[y0 * src_width + x1];
    float v10 = (float)src[y1 * src_width + x0];
    float v11 = (float)src[y1 * src_width + x1];

    float v0 = v00 * (1.0f - fx) + v01 * fx;
    float v1 = v10 * (1.0f - fx) + v11 * fx;
    float v = v0 * (1.0f - fy) + v1 * fy;

    dst[y * dst_width + x] = (unsigned char)fminf(fmaxf(v, 0.0f), 255.0f);
}
'''


# ==============================================================================
# [KERNEL 16] Final Blend with Color Grading
# - 스무딩된 결과와 원본을 강도에 따라 블렌딩
# - 색온도/틴트 보정 통합
# ==============================================================================
FINAL_BLEND_KERNEL_CODE = r'''
extern "C" __global__
void final_blend_kernel(
    const unsigned char* original,     // Original frame (BGR)
    const unsigned char* smoothed,     // Smoothed frame (BGR)
    const unsigned char* mask,         // Blend mask (0-255)
    unsigned char* dst,                // Output (BGR)
    int width, int height,
    float blend_strength,              // Overall smoothing strength (0-1)
    float color_temp,                  // Color temperature adjustment (-1 to 1)
    float color_tint                   // Color tint adjustment (-1 to 1)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx3 = idx * 3;

    // Get mask value and apply overall strength
    float mask_val = (float)mask[idx] / 255.0f * blend_strength;

    // Get original and smoothed values
    float orig_b = (float)original[idx3 + 0];
    float orig_g = (float)original[idx3 + 1];
    float orig_r = (float)original[idx3 + 2];

    float smooth_b = (float)smoothed[idx3 + 0];
    float smooth_g = (float)smoothed[idx3 + 1];
    float smooth_r = (float)smoothed[idx3 + 2];

    // Blend smoothed with original based on mask
    float blend_b = orig_b * (1.0f - mask_val) + smooth_b * mask_val;
    float blend_g = orig_g * (1.0f - mask_val) + smooth_g * mask_val;
    float blend_r = orig_r * (1.0f - mask_val) + smooth_r * mask_val;

    // Apply color temperature (warm = more red/yellow, cool = more blue)
    if (fabsf(color_temp) > 0.01f) {
        if (color_temp > 0.0f) {
            // Warm: increase red, decrease blue
            blend_r = fminf(blend_r + color_temp * 30.0f, 255.0f);
            blend_b = fmaxf(blend_b - color_temp * 20.0f, 0.0f);
        } else {
            // Cool: increase blue, decrease red
            blend_b = fminf(blend_b - color_temp * 30.0f, 255.0f);
            blend_r = fmaxf(blend_r + color_temp * 20.0f, 0.0f);
        }
    }

    // Apply color tint (positive = magenta, negative = green)
    if (fabsf(color_tint) > 0.01f) {
        if (color_tint > 0.0f) {
            // Magenta: increase red and blue, decrease green
            blend_r = fminf(blend_r + color_tint * 15.0f, 255.0f);
            blend_b = fminf(blend_b + color_tint * 15.0f, 255.0f);
            blend_g = fmaxf(blend_g - color_tint * 20.0f, 0.0f);
        } else {
            // Green: decrease red and blue, increase green
            blend_g = fminf(blend_g - color_tint * 20.0f, 255.0f);
            blend_r = fmaxf(blend_r + color_tint * 10.0f, 0.0f);
            blend_b = fmaxf(blend_b + color_tint * 10.0f, 0.0f);
        }
    }

    // Output
    dst[idx3 + 0] = (unsigned char)fminf(fmaxf(blend_b, 0.0f), 255.0f);
    dst[idx3 + 1] = (unsigned char)fminf(fmaxf(blend_g, 0.0f), 255.0f);
    dst[idx3 + 2] = (unsigned char)fminf(fmaxf(blend_r, 0.0f), 255.0f);
}
'''


# ==============================================================================
# [KERNEL 17] Forward Warp Mask (순방향 마스크 워핑) - V4 근본 해결
# - 입력 마스크의 각 픽셀을 출력 위치로 "이동"
# - 슬리밍 시 마스크가 정확히 수축됨
# - Custom Atomic Max (unsigned char용 CAS 기반)
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
    // (역방향 dx/dy를 순방향으로 변환: 부호 반전)
    float shift_x = -dx[s_idx] * (float)scale;
    float shift_y = -dy[s_idx] * (float)scale;

    int out_x = (int)roundf((float)u + shift_x);
    int out_y = (int)roundf((float)v + shift_y);

    // 범위 체크
    if (out_x < 0 || out_x >= width || out_y < 0 || out_y >= height) return;

    int out_idx = out_y * width + out_x;

    // Custom Atomic Max: 여러 입력 픽셀이 같은 출력에 쓸 수 있음
    // 가장 큰 값(가장 불투명)을 유지
    atomicMaxUChar(&mask_out[out_idx], mask_val);
}
'''


# ==============================================================================
# [KERNEL 18] Mask Dilate (마스크 확장으로 홀 채우기)
# - 순방향 매핑에서 생기는 홀(구멍) 문제 해결
# - radius 내 최대값으로 확장
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
# [KERNEL 19] Simple Composite (순방향 마스크 기반 합성) - V4 근본 해결
# - is_alpha_loss 휴리스틱 완전 제거!
# - 순방향 마스크가 정확한 사람/배경 영역을 결정
# - 단순하고 예측 가능한 동작
# ==============================================================================
SIMPLE_COMPOSITE_KERNEL_CODE = r'''
extern "C" __global__
void simple_composite_kernel(
    const unsigned char* src,              // 원본 이미지 (보정된 이미지)
    const unsigned char* bg,               // 배경 이미지
    const unsigned char* forward_mask,     // 순방향 워핑된 마스크
    unsigned char* dst,                    // 출력
    const float* dx_small,                 // 역방향 워핑 그리드
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

    // 순방향 마스크에서 알파 값 (정확한 사람 영역)
    float alpha = (float)forward_mask[idx] / 255.0f;

    // --- 역방향 워핑으로 전경 이미지 가져오기 ---
    int sx = x / scale;
    int sy = y / scale;
    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;
    int s_idx = sy * small_width + sx;

    float shift_x = dx_small[s_idx] * (float)scale;
    float shift_y = dy_small[s_idx] * (float)scale;

    int u = (int)(x + shift_x);
    int v = (int)(y + shift_y);

    float fg_b, fg_g, fg_r;

    if (u >= 0 && u < width && v >= 0 && v < height) {
        int warped_idx_rgb = (v * width + u) * 3;
        fg_b = (float)src[warped_idx_rgb + 0];
        fg_g = (float)src[warped_idx_rgb + 1];
        fg_r = (float)src[warped_idx_rgb + 2];
    } else {
        // 범위 밖: 원본 사용
        fg_b = (float)src[idx_rgb + 0];
        fg_g = (float)src[idx_rgb + 1];
        fg_r = (float)src[idx_rgb + 2];
    }

    // --- 배경 ---
    float bg_b = (float)bg[idx_rgb + 0];
    float bg_g = (float)bg[idx_rgb + 1];
    float bg_r = (float)bg[idx_rgb + 2];

    // --- 최종 합성: 순방향 마스크 기준 ---
    // alpha = 1: 전경 (워핑된 사람)
    // alpha = 0: 배경 (슬리밍으로 비워진 영역)

    if (!use_bg) {
        // 배경 미사용: 워핑된 이미지 그대로
        dst[idx_rgb + 0] = (unsigned char)fg_b;
        dst[idx_rgb + 1] = (unsigned char)fg_g;
        dst[idx_rgb + 2] = (unsigned char)fg_r;
    } else {
        // 알파 블렌딩
        dst[idx_rgb + 0] = (unsigned char)(fg_b * alpha + bg_b * (1.0f - alpha));
        dst[idx_rgb + 1] = (unsigned char)(fg_g * alpha + bg_g * (1.0f - alpha));
        dst[idx_rgb + 2] = (unsigned char)(fg_r * alpha + bg_r * (1.0f - alpha));
    }
}
'''


# ==============================================================================
# [KERNEL 20] Void Fill Composite (V35 - Clean Triple-Layer)
# - Layer 1 (Bottom): 동기화된 원본 프레임 (자연스러운 노이즈/조명 유지)
# - Layer 2 (Middle): 정적 배경 - 오직 "슬리밍으로 비워진 공간(Void)"에만 노출
# - Layer 3 (Top): 워핑된 인물 영역
# - 핵심: 배경 전체를 교체하지 않고 Void만 정적 배경으로 채움 → Floating 해결
# ==============================================================================
VOID_FILL_COMPOSITE_KERNEL_CODE = r'''
extern "C" __global__
void void_fill_composite_kernel(
    const unsigned char* src,        // 동기화된 원본 프레임 (Layer 1 - Bottom)
    const unsigned char* bg,         // 정적 배경 Clean Plate (Layer 2 - Void 전용)
    const unsigned char* mask_orig,  // 동기화된 원본 마스크 (사람이 "있었던" 영역)
    const unsigned char* mask_fwd,   // 순방향 워핑된 마스크 (사람이 "현재 있는" 영역)
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

    // === 마스크 값 읽기 ===
    float m_fwd = (float)mask_fwd[idx] / 255.0f;   // 현재 사람 위치
    float m_orig = (float)mask_orig[idx] / 255.0f; // 원래 사람 위치

    // === 역방향 워핑으로 전경(인물) 샘플링 ===
    // 중요: 동기화된 원본 프레임(src)에서 샘플링해야 잔상이 안 생김
    int sx = x / scale;
    int sy = y / scale;
    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;
    int s_idx = sy * small_width + sx;

    float shift_x = dx_small[s_idx] * (float)scale;
    float shift_y = dy_small[s_idx] * (float)scale;

    int u = (int)(x + shift_x);
    int v = (int)(y + shift_y);

    // 워핑된 전경 픽셀 (동기화된 src에서 샘플링)
    float fg_b, fg_g, fg_r;
    if (u >= 0 && u < width && v >= 0 && v < height) {
        int warped_idx_rgb = (v * width + u) * 3;
        fg_b = (float)src[warped_idx_rgb + 0];
        fg_g = (float)src[warped_idx_rgb + 1];
        fg_r = (float)src[warped_idx_rgb + 2];
    } else {
        // 범위 밖: 원본 위치 사용
        fg_b = (float)src[idx_rgb + 0];
        fg_g = (float)src[idx_rgb + 1];
        fg_r = (float)src[idx_rgb + 2];
    }

    // 정적 좌표 기반 원본/배경
    float src_b = (float)src[idx_rgb + 0];
    float src_g = (float)src[idx_rgb + 1];
    float src_r = (float)src[idx_rgb + 2];

    float bg_b = (float)bg[idx_rgb + 0];
    float bg_g = (float)bg[idx_rgb + 1];
    float bg_r = (float)bg[idx_rgb + 2];

    // === [V35] 클린 트리플 레이어 합성 ===
    float out_b, out_g, out_r;

    // 임계값 정의
    const float PERSON_THRESHOLD = 0.15f;  // 사람 영역 판정 임계값
    const float VOID_THRESHOLD = 0.10f;    // Void 판정 임계값

    if (!use_bg) {
        // 배경 미사용: 단순 워핑만 적용
        out_b = fg_b;
        out_g = fg_g;
        out_r = fg_r;
    }
    else if (m_fwd > PERSON_THRESHOLD) {
        // === [Layer 3 - Top] 사람 영역 ===
        // mask_fwd가 있는 곳 = 현재 사람이 있는 위치
        // → 워핑된 전경(fg) 출력
        out_b = fg_b;
        out_g = fg_g;
        out_r = fg_r;
    }
    else if (m_orig > VOID_THRESHOLD && m_fwd <= PERSON_THRESHOLD) {
        // === [Layer 2 - Middle] 슬리밍 Void 영역 ===
        // mask_orig > 0 이지만 mask_fwd == 0
        // = "사람이 있었으나 슬리밍으로 사라진 공간"
        // → 정적 배경(bg)으로 채움 (Bending 해결)

        // Void 강도 계산: 원래 마스크값이 높을수록 확실한 Void
        float void_strength = m_orig;

        // 부드러운 블렌딩 (경계면 자연스럽게)
        // smoothstep으로 부드러운 전환
        float t = (void_strength - VOID_THRESHOLD) / (0.5f - VOID_THRESHOLD);
        t = fmaxf(0.0f, fminf(1.0f, t));
        float blend = t * t * (3.0f - 2.0f * t);  // smoothstep

        out_b = bg_b * blend + src_b * (1.0f - blend);
        out_g = bg_g * blend + src_g * (1.0f - blend);
        out_r = bg_r * blend + src_r * (1.0f - blend);
    }
    else {
        // === [Layer 1 - Bottom] 일반 배경 영역 ===
        // 사람도 없고 Void도 아닌 순수 배경
        // → 동기화된 원본 프레임(src) 그대로 출력
        // (자연스러운 카메라 노이즈/조명 변화 유지 → Floating 해결)
        out_b = src_b;
        out_g = src_g;
        out_r = src_r;
    }

    dst[idx_rgb + 0] = (unsigned char)out_b;
    dst[idx_rgb + 1] = (unsigned char)out_g;
    dst[idx_rgb + 2] = (unsigned char)out_r;
}
'''

# ==============================================================================
# [KERNEL 20] Layered Composite (V34 - Clean 2-Layer Architecture)
# - 정적 배경(Base) + 워핑된 인물(Top)의 단순 2레이어 구조
# - 배경은 절대 워핑 좌표 참조 없이 정적 좌표(x, y)만 사용
# - 인물은 역방향 워핑된 좌표(u, v)에서 샘플링
# - Chromatic Guard 및 복잡한 에지 보정 로직 제거
# ==============================================================================
LAYERED_COMPOSITE_KERNEL_CODE = r'''
extern "C" __global__
void layered_composite_kernel(
    const unsigned char* src,        // 워핑 대상 원본 (피부 보정 적용됨)
    const unsigned char* bg,         // 정적 배경 (절대 워핑 안 됨)
    const unsigned char* mask,       // 인물 마스크 (동기화됨, 0-255)
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

    // === [BASE LAYER] 정적 배경 (항상 정적 좌표 사용) ===
    float bg_b = (float)bg[idx_rgb + 0];
    float bg_g = (float)bg[idx_rgb + 1];
    float bg_r = (float)bg[idx_rgb + 2];

    if (!use_bg) {
        // 배경 미사용 시 원본 그대로 출력
        dst[idx_rgb + 0] = src[idx_rgb + 0];
        dst[idx_rgb + 1] = src[idx_rgb + 1];
        dst[idx_rgb + 2] = src[idx_rgb + 2];
        return;
    }

    // === 역방향 워핑 좌표 계산 ===
    int sx = x / scale;
    int sy = y / scale;
    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;
    int s_idx = sy * small_width + sx;

    float shift_x = dx_small[s_idx] * (float)scale;
    float shift_y = dy_small[s_idx] * (float)scale;

    int u = (int)(x + shift_x);
    int v = (int)(y + shift_y);

    // === [TOP LAYER] 워핑된 인물 샘플링 ===
    float fg_b, fg_g, fg_r;
    float fg_alpha = 0.0f;

    if (u >= 0 && u < width && v >= 0 && v < height) {
        int warped_idx = v * width + u;
        int warped_idx_rgb = warped_idx * 3;

        // 워핑된 위치의 마스크 값 (인물 영역 판정)
        fg_alpha = (float)mask[warped_idx] / 255.0f;

        fg_b = (float)src[warped_idx_rgb + 0];
        fg_g = (float)src[warped_idx_rgb + 1];
        fg_r = (float)src[warped_idx_rgb + 2];
    } else {
        // 범위 밖: 배경 사용
        fg_b = bg_b;
        fg_g = bg_g;
        fg_r = bg_r;
        fg_alpha = 0.0f;
    }

    // === 단순 2레이어 합성 ===
    // fg_alpha = 1.0: 완전 인물 (워핑된 src 사용)
    // fg_alpha = 0.0: 완전 배경 (정적 bg 사용)
    // 중간값: 블렌딩

    float out_b = fg_b * fg_alpha + bg_b * (1.0f - fg_alpha);
    float out_g = fg_g * fg_alpha + bg_g * (1.0f - fg_alpha);
    float out_r = fg_r * fg_alpha + bg_r * (1.0f - fg_alpha);

    dst[idx_rgb + 0] = (unsigned char)out_b;
    dst[idx_rgb + 1] = (unsigned char)out_g;
    dst[idx_rgb + 2] = (unsigned char)out_r;
}
'''

# ==============================================================================
# [KERNEL V36] Warp Mask from Grid (워핑 그리드 기반 마스크 생성)
# - 워핑 그리드(dx, dy)에서 "워핑 영역"을 마스크로 추출
# - 워핑 강도(magnitude)가 임계값 이상인 영역 = 사람 영역
# - 시간 동기화 문제 근본 해결 (워핑 그리드와 마스크가 같은 소스에서 생성)
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
    // - 부드러운 falloff를 위해 smoothstep 적용
    float edge0 = threshold * 0.5f;
    float edge1 = threshold * 2.0f;

    float t = (magnitude - edge0) / (edge1 - edge0);
    t = fmaxf(0.0f, fminf(1.0f, t));
    float alpha = t * t * (3.0f - 2.0f * t);  // smoothstep

    int idx = y * width + x;
    warp_mask[idx] = (unsigned char)(alpha * 255.0f);
}
'''

# ==============================================================================
# [KERNEL V36] Void Only Fill Composite (Void 영역만 배경으로 채움)
# - 기본: 워핑된 원본 프레임
# - Void 영역만: 저장된 배경으로 패치 (슬리밍으로 비어진 공간만)
# - 배경 전체 교체 X, Void만 채움 O
# ==============================================================================
CLEAN_COMPOSITE_KERNEL_CODE = r'''
extern "C" __global__
void clean_composite_kernel(
    const unsigned char* src,        // 피부보정된 현재 프레임
    const unsigned char* bg,         // 정적 배경
    const unsigned char* mask_orig,  // 원본 마스크 (사람이 "있었던" 영역)
    const unsigned char* mask_fwd,   // 순방향 워핑된 마스크 (사람이 "현재 있는" 영역)
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

    // === 마스크 값 읽기 ===
    float m_orig = (float)mask_orig[idx] / 255.0f;  // 원래 사람 위치
    float m_fwd = (float)mask_fwd[idx] / 255.0f;    // 현재 사람 위치 (슬리밍 후)

    // === 역방향 워핑으로 src에서 픽셀 가져오기 ===
    int sx = x / scale;
    int sy = y / scale;
    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;
    int s_idx = sy * small_width + sx;

    float shift_x = dx_small[s_idx] * (float)scale;
    float shift_y = dy_small[s_idx] * (float)scale;

    int u = (int)(x + shift_x);
    int v = (int)(y + shift_y);

    // 워핑된 픽셀
    float warped_b, warped_g, warped_r;
    if (u >= 0 && u < width && v >= 0 && v < height) {
        int warped_idx_rgb = (v * width + u) * 3;
        warped_b = (float)src[warped_idx_rgb + 0];
        warped_g = (float)src[warped_idx_rgb + 1];
        warped_r = (float)src[warped_idx_rgb + 2];
    } else {
        // 범위 밖: 원본 위치 사용
        warped_b = (float)src[idx_rgb + 0];
        warped_g = (float)src[idx_rgb + 1];
        warped_r = (float)src[idx_rgb + 2];
    }

    // 정적 배경 픽셀
    float bg_b = (float)bg[idx_rgb + 0];
    float bg_g = (float)bg[idx_rgb + 1];
    float bg_r = (float)bg[idx_rgb + 2];

    // === [V37] Void Only Fill 합성 (Smooth Weight) ===
    float out_b, out_g, out_r;

    if (!use_bg) {
        // 배경 미사용: 워핑된 이미지 그대로
        out_b = warped_b;
        out_g = warped_g;
        out_r = warped_r;
    } else {
        // [V37] Smooth Void Weight - 이진 판정 대신 연속 가중치
        float void_weight = m_orig * (1.0f - m_fwd);

        // [V37] Noise Floor: 아주 작은 void는 무시
        const float VOID_NOISE_FLOOR = 0.03f;
        if (void_weight < VOID_NOISE_FLOOR) {
            void_weight = 0.0f;
        }

        // [V37] Smoothstep for gradual transition
        float edge0 = 0.05f;
        float edge1 = 0.25f;
        float t = (void_weight - edge0) / (edge1 - edge0);
        t = fmaxf(0.0f, fminf(1.0f, t));
        float smooth_void = t * t * (3.0f - 2.0f * t);

        // [V37] 연속 블렌딩 (경계 토글링 방지)
        out_b = bg_b * smooth_void + warped_b * (1.0f - smooth_void);
        out_g = bg_g * smooth_void + warped_g * (1.0f - smooth_void);
        out_r = bg_r * smooth_void + warped_r * (1.0f - smooth_void);
    }

    dst[idx_rgb + 0] = (unsigned char)out_b;
    dst[idx_rgb + 1] = (unsigned char)out_g;
    dst[idx_rgb + 2] = (unsigned char)out_r;
}
'''
