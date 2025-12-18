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

    // --- 3. Final Decision (Context-Aware Composite) ---
    bool is_alpha_loss = (original_alpha > warped_alpha + 0.05f);
    float base_b, base_g, base_r;

    if (is_alpha_loss && is_significant_warp) {
        // [Case A] Valid Slimming -> 배경(Clean Plate) 합성
        base_b = (float)bg[idx_rgb+0];
        base_g = (float)bg[idx_rgb+1];
        base_r = (float)bg[idx_rgb+2];
    } else {
        // [Case B] No Change or Internal Noise -> 원본(Live Feed) 유지
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
# [KERNEL 4] Polygon Mask Generation (Scanline Algorithm)
# - GPU 기반 고속 폴리곤 래스터화
# - 얼굴 윤곽/눈/입술 등 정밀 마스크 생성
# ==============================================================================
POLYGON_MASK_KERNEL_CODE = r'''
extern "C" __global__
void polygon_mask_kernel(
    unsigned char* mask,           // Output: Binary mask (single channel)
    const float* vertices,         // Input: [x0,y0, x1,y1, ...] polygon vertices (flattened)
    int num_vertices,              // Number of vertices
    int width, int height,         // Image dimensions
    unsigned char fill_value       // Value to fill (255 for inclusion, 0 for exclusion)
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    // Edge intersection calculation for this scanline
    // Using ray casting (point-in-polygon) algorithm

    for (int x = 0; x < width; x++) {
        int intersections = 0;

        for (int i = 0; i < num_vertices; i++) {
            int j = (i + 1) % num_vertices;

            float y1 = vertices[i * 2 + 1];
            float y2 = vertices[j * 2 + 1];
            float x1 = vertices[i * 2 + 0];
            float x2 = vertices[j * 2 + 0];

            // Check if edge crosses the scanline
            if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
                // Calculate x intersection
                float x_intersect = x1 + (y - y1) / (y2 - y1) * (x2 - x1);
                if (x < x_intersect) {
                    intersections++;
                }
            }
        }

        // Odd number of intersections = inside polygon
        if (intersections % 2 == 1) {
            mask[y * width + x] = fill_value;
        }
    }
}
'''

# ==============================================================================
# [KERNEL 5] Multi-Polygon Skin Mask (Face - Exclusions)
# - 얼굴 영역에서 눈/눈썹/입술 영역을 제외한 피부 마스크 생성
# - Single-pass 처리로 효율적
# ==============================================================================
SKIN_MASK_KERNEL_CODE = r'''
extern "C" __global__
void skin_mask_kernel(
    unsigned char* mask,              // Output: Skin mask (255=skin, 0=excluded)
    const float* face_vertices,       // Face oval polygon
    int face_num_vertices,
    const float* eye_l_vertices,      // Left eye polygon
    int eye_l_num_vertices,
    const float* eye_r_vertices,      // Right eye polygon
    int eye_r_num_vertices,
    const float* brow_l_vertices,     // Left brow polygon
    int brow_l_num_vertices,
    const float* brow_r_vertices,     // Right brow polygon
    int brow_r_num_vertices,
    const float* lips_vertices,       // Lips polygon
    int lips_num_vertices,
    int width, int height,
    float exclusion_padding           // Padding multiplier for exclusion zones (1.0 = exact, 1.2 = 20% larger)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Helper function to check point-in-polygon (ray casting)
    // Inline implementation for each polygon

    // 1. Check if inside face oval
    int face_intersections = 0;
    for (int i = 0; i < face_num_vertices; i++) {
        int j = (i + 1) % face_num_vertices;
        float y1 = face_vertices[i * 2 + 1];
        float y2 = face_vertices[j * 2 + 1];
        float x1 = face_vertices[i * 2 + 0];
        float x2 = face_vertices[j * 2 + 0];

        if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
            float x_int = x1 + (y - y1) / (y2 - y1) * (x2 - x1);
            if (x < x_int) face_intersections++;
        }
    }
    bool inside_face = (face_intersections % 2 == 1);

    if (!inside_face) {
        mask[idx] = 0;
        return;
    }

    // 2. Check exclusion zones (with padding for soft edges)
    // We use scaled distance check for padded exclusions

    // Left Eye
    int eye_l_int = 0;
    for (int i = 0; i < eye_l_num_vertices; i++) {
        int j = (i + 1) % eye_l_num_vertices;
        float y1 = eye_l_vertices[i * 2 + 1];
        float y2 = eye_l_vertices[j * 2 + 1];
        float x1 = eye_l_vertices[i * 2 + 0];
        float x2 = eye_l_vertices[j * 2 + 0];
        if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
            float x_int = x1 + (y - y1) / (y2 - y1) * (x2 - x1);
            if (x < x_int) eye_l_int++;
        }
    }
    if (eye_l_int % 2 == 1) { mask[idx] = 0; return; }

    // Right Eye
    int eye_r_int = 0;
    for (int i = 0; i < eye_r_num_vertices; i++) {
        int j = (i + 1) % eye_r_num_vertices;
        float y1 = eye_r_vertices[i * 2 + 1];
        float y2 = eye_r_vertices[j * 2 + 1];
        float x1 = eye_r_vertices[i * 2 + 0];
        float x2 = eye_r_vertices[j * 2 + 0];
        if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
            float x_int = x1 + (y - y1) / (y2 - y1) * (x2 - x1);
            if (x < x_int) eye_r_int++;
        }
    }
    if (eye_r_int % 2 == 1) { mask[idx] = 0; return; }

    // Left Brow
    int brow_l_int = 0;
    for (int i = 0; i < brow_l_num_vertices; i++) {
        int j = (i + 1) % brow_l_num_vertices;
        float y1 = brow_l_vertices[i * 2 + 1];
        float y2 = brow_l_vertices[j * 2 + 1];
        float x1 = brow_l_vertices[i * 2 + 0];
        float x2 = brow_l_vertices[j * 2 + 0];
        if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
            float x_int = x1 + (y - y1) / (y2 - y1) * (x2 - x1);
            if (x < x_int) brow_l_int++;
        }
    }
    if (brow_l_int % 2 == 1) { mask[idx] = 0; return; }

    // Right Brow
    int brow_r_int = 0;
    for (int i = 0; i < brow_r_num_vertices; i++) {
        int j = (i + 1) % brow_r_num_vertices;
        float y1 = brow_r_vertices[i * 2 + 1];
        float y2 = brow_r_vertices[j * 2 + 1];
        float x1 = brow_r_vertices[i * 2 + 0];
        float x2 = brow_r_vertices[j * 2 + 0];
        if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
            float x_int = x1 + (y - y1) / (y2 - y1) * (x2 - x1);
            if (x < x_int) brow_r_int++;
        }
    }
    if (brow_r_int % 2 == 1) { mask[idx] = 0; return; }

    // Lips
    int lips_int = 0;
    for (int i = 0; i < lips_num_vertices; i++) {
        int j = (i + 1) % lips_num_vertices;
        float y1 = lips_vertices[i * 2 + 1];
        float y2 = lips_vertices[j * 2 + 1];
        float x1 = lips_vertices[i * 2 + 0];
        float x2 = lips_vertices[j * 2 + 0];
        if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
            float x_int = x1 + (y - y1) / (y2 - y1) * (x2 - x1);
            if (x < x_int) lips_int++;
        }
    }
    if (lips_int % 2 == 1) { mask[idx] = 0; return; }

    // Passed all exclusions - this is skin!
    mask[idx] = 255;
}
'''

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
# [KERNEL 7] Tone Uniformity (Flat-fielding)
# - 피부 얼룩 제거 (Redness, Dark spots)
# - Low-frequency를 평균색 방향으로 보정
# ==============================================================================
TONE_UNIFORMITY_KERNEL_CODE = r'''
extern "C" __global__
void tone_uniformity_kernel(
    const unsigned char* src,        // Original image (BGR)
    const unsigned char* low_freq,   // Guided filter result (BGR)
    unsigned char* dst,              // Output (BGR)
    const unsigned char* mask,       // Skin mask
    int width, int height,
    float mean_b, float mean_g, float mean_r,  // Target skin color (mask average)
    float flatten_strength,          // 0.0-1.0: how much to push towards mean
    float detail_preserve            // 0.0-1.0: high-freq preservation amount
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx3 = idx * 3;

    // Non-skin: pass through original
    if (mask[idx] == 0) {
        dst[idx3 + 0] = src[idx3 + 0];
        dst[idx3 + 1] = src[idx3 + 1];
        dst[idx3 + 2] = src[idx3 + 2];
        return;
    }

    // Get original and low-frequency values
    float orig_b = (float)src[idx3 + 0];
    float orig_g = (float)src[idx3 + 1];
    float orig_r = (float)src[idx3 + 2];

    float lf_b = (float)low_freq[idx3 + 0];
    float lf_g = (float)low_freq[idx3 + 1];
    float lf_r = (float)low_freq[idx3 + 2];

    // High-frequency component (detail)
    float hf_b = orig_b - lf_b;
    float hf_g = orig_g - lf_g;
    float hf_r = orig_r - lf_r;

    // Flatten low-frequency towards mean color
    // This removes uneven tones while preserving overall skin color
    float flat_b = lf_b * (1.0f - flatten_strength) + mean_b * flatten_strength;
    float flat_g = lf_g * (1.0f - flatten_strength) + mean_g * flatten_strength;
    float flat_r = lf_r * (1.0f - flatten_strength) + mean_r * flatten_strength;

    // Reconstruct with preserved high-frequency detail
    float out_b = flat_b + hf_b * detail_preserve;
    float out_g = flat_g + hf_g * detail_preserve;
    float out_r = flat_r + hf_r * detail_preserve;

    // Clamp and output
    dst[idx3 + 0] = (unsigned char)fminf(fmaxf(out_b, 0.0f), 255.0f);
    dst[idx3 + 1] = (unsigned char)fminf(fmaxf(out_g, 0.0f), 255.0f);
    dst[idx3 + 2] = (unsigned char)fminf(fmaxf(out_r, 0.0f), 255.0f);
}
'''

# ==============================================================================
# [KERNEL 8] Color Grading (HSL-based Temperature & Tint)
# - 색온도: Blue ↔ Yellow (Warm/Cool)
# - 틴트: Green ↔ Magenta
# - 효율적인 HSL 변환 기반
# ==============================================================================
COLOR_GRADING_KERNEL_CODE = r'''
extern "C" __global__
void color_grading_kernel(
    const unsigned char* src,   // Input BGR
    unsigned char* dst,         // Output BGR
    const unsigned char* mask,  // Skin mask (optional, can be NULL for full image)
    int width, int height,
    float temperature,          // -1.0(Cool/Blue) ~ 1.0(Warm/Yellow)
    float tint,                 // -1.0(Green) ~ 1.0(Magenta)
    int use_mask                // 1=apply only to masked area, 0=full image
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx3 = idx * 3;

    // Check mask
    if (use_mask && mask != NULL && mask[idx] == 0) {
        dst[idx3 + 0] = src[idx3 + 0];
        dst[idx3 + 1] = src[idx3 + 1];
        dst[idx3 + 2] = src[idx3 + 2];
        return;
    }

    // Read BGR
    float b = (float)src[idx3 + 0] / 255.0f;
    float g = (float)src[idx3 + 1] / 255.0f;
    float r = (float)src[idx3 + 2] / 255.0f;

    // === RGB to HSL ===
    float cmax = fmaxf(r, fmaxf(g, b));
    float cmin = fminf(r, fminf(g, b));
    float delta = cmax - cmin;

    float h = 0.0f, s = 0.0f, l = (cmax + cmin) * 0.5f;

    if (delta > 0.0001f) {
        s = (l > 0.5f) ? delta / (2.0f - cmax - cmin) : delta / (cmax + cmin);

        if (cmax == r) {
            h = fmodf((g - b) / delta + 6.0f, 6.0f) / 6.0f;
        } else if (cmax == g) {
            h = ((b - r) / delta + 2.0f) / 6.0f;
        } else {
            h = ((r - g) / delta + 4.0f) / 6.0f;
        }
    }

    // === Apply Temperature & Tint ===
    // Temperature: Shift hue towards yellow (0.16) or blue (0.66)
    // Scale: ±0.05 hue shift max
    float temp_shift = temperature * 0.03f;

    // Tint: Shift hue towards green (0.33) or magenta (0.83)
    // Scale: ±0.03 hue shift max
    float tint_shift = tint * 0.02f;

    // Combined hue adjustment
    h = h + temp_shift + tint_shift;
    h = fmodf(h + 1.0f, 1.0f); // Wrap to [0, 1]

    // Temperature also affects saturation slightly
    s = s * (1.0f + temperature * 0.1f);
    s = fminf(fmaxf(s, 0.0f), 1.0f);

    // === HSL to RGB ===
    float c = (1.0f - fabsf(2.0f * l - 1.0f)) * s;
    float x_val = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
    float m = l - c * 0.5f;

    float r_out, g_out, b_out;

    int h_sector = (int)(h * 6.0f);
    switch (h_sector % 6) {
        case 0: r_out = c; g_out = x_val; b_out = 0; break;
        case 1: r_out = x_val; g_out = c; b_out = 0; break;
        case 2: r_out = 0; g_out = c; b_out = x_val; break;
        case 3: r_out = 0; g_out = x_val; b_out = c; break;
        case 4: r_out = x_val; g_out = 0; b_out = c; break;
        default: r_out = c; g_out = 0; b_out = x_val; break;
    }

    r_out += m;
    g_out += m;
    b_out += m;

    // Output BGR
    dst[idx3 + 0] = (unsigned char)(fminf(fmaxf(b_out * 255.0f, 0.0f), 255.0f));
    dst[idx3 + 1] = (unsigned char)(fminf(fmaxf(g_out * 255.0f, 0.0f), 255.0f));
    dst[idx3 + 2] = (unsigned char)(fminf(fmaxf(r_out * 255.0f, 0.0f), 255.0f));
}
'''

# ==============================================================================
# [KERNEL 9] Advanced Skin Smooth V2 (Polygon Mask + Guided Filter)
# - 폴리곤 마스크 기반 정밀 피부 영역 처리
# - Guided Filter 통합
# - 기존 SKIN_SMOOTH_KERNEL의 상위 호환 버전
# ==============================================================================
SKIN_SMOOTH_V2_KERNEL_CODE = r'''
extern "C" __global__
void skin_smooth_v2_kernel(
    const unsigned char* src,        // Input BGR
    const unsigned char* guide,      // Guide for edge preservation (can be same as src)
    unsigned char* dst,              // Output BGR
    const unsigned char* skin_mask,  // Polygon-based skin mask
    int width, int height,
    float strength,                  // Overall smoothing strength 0-1
    int radius,                      // Filter radius
    float epsilon,                   // Edge preservation (lower = more edge preservation)
    float target_r, float target_g, float target_b,  // Target skin color
    float tone_val                   // -1.0(pale) ~ 1.0(rosy)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx3 = idx * 3;

    // Original pixel
    float orig_b = (float)src[idx3 + 0];
    float orig_g = (float)src[idx3 + 1];
    float orig_r = (float)src[idx3 + 2];

    // Non-skin: pass through
    if (skin_mask[idx] == 0) {
        dst[idx3 + 0] = (unsigned char)orig_b;
        dst[idx3 + 1] = (unsigned char)orig_g;
        dst[idx3 + 2] = (unsigned char)orig_r;
        return;
    }

    // Guided filter statistics (local box)
    float sum_I_b = 0.0f, sum_I_g = 0.0f, sum_I_r = 0.0f;
    float sum_p_b = 0.0f, sum_p_g = 0.0f, sum_p_r = 0.0f;
    float sum_Ip_b = 0.0f, sum_Ip_g = 0.0f, sum_Ip_r = 0.0f;
    float sum_II_b = 0.0f, sum_II_g = 0.0f, sum_II_r = 0.0f;
    int count = 0;

    int x_min = max(0, x - radius);
    int x_max = min(width - 1, x + radius);
    int y_min = max(0, y - radius);
    int y_max = min(height - 1, y + radius);

    for (int ny = y_min; ny <= y_max; ny++) {
        for (int nx = x_min; nx <= x_max; nx++) {
            int n_idx = ny * width + nx;
            int n_idx3 = n_idx * 3;

            // Only include skin pixels in statistics
            if (skin_mask[n_idx] > 0) {
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
    }

    if (count < 1) {
        dst[idx3 + 0] = (unsigned char)orig_b;
        dst[idx3 + 1] = (unsigned char)orig_g;
        dst[idx3 + 2] = (unsigned char)orig_r;
        return;
    }

    float inv_count = 1.0f / (float)count;

    float mean_I_b = sum_I_b * inv_count;
    float mean_I_g = sum_I_g * inv_count;
    float mean_I_r = sum_I_r * inv_count;

    float mean_p_b = sum_p_b * inv_count;
    float mean_p_g = sum_p_g * inv_count;
    float mean_p_r = sum_p_r * inv_count;

    float cov_Ip_b = sum_Ip_b * inv_count - mean_I_b * mean_p_b;
    float cov_Ip_g = sum_Ip_g * inv_count - mean_I_g * mean_p_g;
    float cov_Ip_r = sum_Ip_r * inv_count - mean_I_r * mean_p_r;

    float var_I_b = sum_II_b * inv_count - mean_I_b * mean_I_b;
    float var_I_g = sum_II_g * inv_count - mean_I_g * mean_I_g;
    float var_I_r = sum_II_r * inv_count - mean_I_r * mean_I_r;

    float eps_scaled = epsilon * 255.0f * 255.0f;

    float a_b = cov_Ip_b / (var_I_b + eps_scaled);
    float a_g = cov_Ip_g / (var_I_g + eps_scaled);
    float a_r = cov_Ip_r / (var_I_r + eps_scaled);

    float b_b = mean_p_b - a_b * mean_I_b;
    float b_g = mean_p_g - a_g * mean_I_g;
    float b_r = mean_p_r - a_r * mean_I_r;

    float I_b = (float)guide[idx3 + 0];
    float I_g = (float)guide[idx3 + 1];
    float I_r = (float)guide[idx3 + 2];

    float smooth_b = a_b * I_b + b_b;
    float smooth_g = a_g * I_g + b_g;
    float smooth_r = a_r * I_r + b_r;

    // Tone correction
    if (fabsf(tone_val) > 0.05f) {
        float mix_factor;
        float tr, tg, tb;

        if (tone_val > 0.0f) {
            tr = 255.0f; tg = 215.0f; tb = 225.0f;
            mix_factor = tone_val * 0.4f;
        } else {
            tr = 255.0f; tg = 255.0f; tb = 255.0f;
            mix_factor = -tone_val * 0.3f;
        }

        smooth_r = smooth_r * (1.0f - mix_factor) + tr * mix_factor;
        smooth_g = smooth_g * (1.0f - mix_factor) + tg * mix_factor;
        smooth_b = smooth_b * (1.0f - mix_factor) + tb * mix_factor;
    }

    // Blend with original based on strength
    float out_b = orig_b * (1.0f - strength) + smooth_b * strength;
    float out_g = orig_g * (1.0f - strength) + smooth_g * strength;
    float out_r = orig_r * (1.0f - strength) + smooth_r * strength;

    dst[idx3 + 0] = (unsigned char)fminf(fmaxf(out_b, 0.0f), 255.0f);
    dst[idx3 + 1] = (unsigned char)fminf(fmaxf(out_g, 0.0f), 255.0f);
    dst[idx3 + 2] = (unsigned char)fminf(fmaxf(out_r, 0.0f), 255.0f);
}
'''

# ==============================================================================
# [KERNEL 10] Soft Edge Mask Blur (Feathering)
# - 폴리곤 마스크 경계를 부드럽게 처리
# - 스무딩 경계의 자연스러운 블렌딩
# ==============================================================================
MASK_BLUR_KERNEL_CODE = r'''
extern "C" __global__
void mask_blur_kernel(
    const unsigned char* mask_in,   // Input hard mask
    unsigned char* mask_out,        // Output soft mask
    int width, int height,
    int blur_radius                 // Feather radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Box blur for mask feathering
    float sum = 0.0f;
    int count = 0;

    for (int dy = -blur_radius; dy <= blur_radius; dy++) {
        for (int dx = -blur_radius; dx <= blur_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += (float)mask_in[ny * width + nx];
                count++;
            }
        }
    }

    mask_out[idx] = (unsigned char)(sum / (float)count);
}
'''

# ==============================================================================
# [KERNEL 11] Skin Color Expansion (Region Growing)
# - FaceMesh 경계 밖 인접 피부색 영역으로 마스크 확장
# - YCrCb 색공간 기반 피부색 검출
# - 기존 마스크에서 바깥으로 확장
# ==============================================================================
SKIN_COLOR_EXPAND_KERNEL_CODE = r'''
extern "C" __global__
void skin_color_expand_kernel(
    const unsigned char* src_bgr,      // Input image (BGR)
    const unsigned char* mask_in,      // Input: polygon-based mask
    unsigned char* mask_out,           // Output: expanded mask
    int width, int height,
    float ref_y, float ref_cr, float ref_cb,  // Reference skin color in YCrCb
    float y_thresh,                    // Y (luminance) tolerance
    float cr_thresh,                   // Cr tolerance
    float cb_thresh,                   // Cb tolerance
    int expand_radius                  // How many pixels to check for expansion
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx3 = idx * 3;

    // If already in mask, keep it
    if (mask_in[idx] > 0) {
        mask_out[idx] = mask_in[idx];
        return;
    }

    // Check if near existing mask boundary
    bool near_mask = false;
    for (int dy = -expand_radius; dy <= expand_radius && !near_mask; dy++) {
        for (int dx = -expand_radius; dx <= expand_radius && !near_mask; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (mask_in[ny * width + nx] > 0) {
                    near_mask = true;
                }
            }
        }
    }

    if (!near_mask) {
        mask_out[idx] = 0;
        return;
    }

    // Convert BGR to YCrCb for skin detection
    float b = (float)src_bgr[idx3 + 0];
    float g = (float)src_bgr[idx3 + 1];
    float r = (float)src_bgr[idx3 + 2];

    // RGB to YCrCb conversion
    float Y  = 0.299f * r + 0.587f * g + 0.114f * b;
    float Cr = (r - Y) * 0.713f + 128.0f;
    float Cb = (b - Y) * 0.564f + 128.0f;

    // Check if pixel is skin-colored (similar to reference)
    float y_diff = fabsf(Y - ref_y);
    float cr_diff = fabsf(Cr - ref_cr);
    float cb_diff = fabsf(Cb - ref_cb);

    bool is_skin_color = (y_diff < y_thresh) && (cr_diff < cr_thresh) && (cb_diff < cb_thresh);

    // Also check general skin color range in YCrCb
    // Typical skin: Cr: 133-173, Cb: 77-127
    bool in_skin_range = (Cr >= 133.0f && Cr <= 173.0f) && (Cb >= 77.0f && Cb <= 127.0f);

    if (is_skin_color || in_skin_range) {
        mask_out[idx] = 255;
    } else {
        mask_out[idx] = 0;
    }
}
'''


# ==============================================================================
# [KERNEL 12] Fast Skin Smooth Blend
# - Combines smoothing and blending in one pass
# - Uses simple box average for smoothing (fast)
# - Applies frequency separation for natural look
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
        // Outside mask - copy original
        dst[idx3 + 0] = src[idx3 + 0];
        dst[idx3 + 1] = src[idx3 + 1];
        dst[idx3 + 2] = src[idx3 + 2];
        return;
    }

    // Frequency separation: result = smoothed + (original - smoothed) * detail_preserve
    for (int c = 0; c < 3; c++) {
        float orig = (float)src[idx3 + c];
        float smooth = (float)smoothed[idx3 + c];
        float detail = orig - smooth;
        float result = smooth + detail * detail_preserve;

        // Blend with original based on mask
        float final_val = orig * (1.0f - mask_val) + result * mask_val;
        dst[idx3 + c] = (unsigned char)fminf(fmaxf(final_val, 0.0f), 255.0f);
    }
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
