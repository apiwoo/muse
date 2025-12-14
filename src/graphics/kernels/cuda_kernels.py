# Project MUSE - cuda_kernels.py
# High-Fidelity Kernels (Alpha Blending & TPS Warping)
# Updated: Ghosting Fix V13 (Revert RGB Check, Optimize Loss Margin)
# (C) 2025 MUSE Corp. All rights reserved.

# [Kernel 1] Grid Generation (TPS Logic) - 유지
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

# [Kernel 2] Smart Composite (Ghosting Fix V13 - Logic Restore)
# 색상 검증 로직(RGB Diff)을 제거하여 보정(Slimming) 기능을 복구합니다.
# 대신 is_alpha_loss의 민감도를 조절하여 미세한 노이즈를 무시합니다.
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

    // --- 2. Warped Alpha (After) ---
    int sx = x / scale;
    int sy = y / scale;
    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;
    int s_idx = sy * small_width + sx;

    float shift_x = dx_small[s_idx] * (float)scale;
    float shift_y = dy_small[s_idx] * (float)scale;

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

    // --- 3. Base Layer Construction (Smart Loss Logic Restored) ---
    
    // [Check 1] 현재 확실한 몸체인가? (이 픽셀은 보정 결과 몸으로 남아야 함)
    bool is_solid_body = (warped_alpha > 0.9f);

    // [Check 2] 알파값이 줄어들었는가? (Loss Detection)
    // 보정으로 인해 공간이 비었는지 확인.
    // Margin을 0.02 -> 0.05로 상향하여 미세한 연산 오차로 인한 깜빡임 방지
    bool is_alpha_loss = (original_alpha > warped_alpha + 0.05f);

    float base_b, base_g, base_r;

    // [Final Decision]
    // 1. 몸이 아니거나 반투명한 상태인데 (!is_solid_body)
    // 2. 원래보다 알파값이 확실히 줄어들었다면 (is_alpha_loss)
    // -> "보정으로 인해 깎여나간 빈 공간"으로 간주하고 배경(BG)을 채웁니다.
    // -> 이 로직이 작동해야 'Slimming' 효과가 나타납니다.
    if (!is_solid_body && is_alpha_loss) {
        base_b = (float)bg[idx_rgb+0];
        base_g = (float)bg[idx_rgb+1];
        base_r = (float)bg[idx_rgb+2];
    } else {
        // 그 외 (확실한 몸통, 늘어난 부위 등)
        // -> 원본(Live Feed)을 유지합니다.
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