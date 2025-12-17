# Project MUSE - cuda_kernels.py
# High-Fidelity Kernels (Alpha Blending & TPS Warping)
# Updated: Topology Protection V21 (Intrusion Logic Fix)
# Added: Skin Smoothing Kernel (Bilateral-like)
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

# [Kernel 2] Smart Composite (Intrusion Handling)
# "워핑에 의해 침범된 영역(Slimming Zone)"과 "단순 오류(Ghosting)"를 구분.
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

# [Kernel 3] Skin Smooth (Bilateral-like)
# 피부 톤 보정: 엣지(색상 차이가 큰 곳)는 살리고, 평탄한 부분(잡티)만 블러링
SKIN_SMOOTH_KERNEL_CODE = r'''
extern "C" __global__
void skin_smooth_kernel(
    const unsigned char* src, 
    unsigned char* dst, 
    int width, int height, 
    float strength
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    
    float center_r = (float)src[idx+0];
    float center_g = (float)src[idx+1];
    float center_b = (float)src[idx+2];
    
    // Config
    int radius = 4; // 9x9 Window (Performance tradeoff)
    float sigma_color = 25.0f; // 색상 차이 허용치 (작을수록 엣지 보존 강함)
    
    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
    float total_w = 0.0f;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = (ny * width + nx) * 3;
                float nr = (float)src[n_idx+0];
                float ng = (float)src[n_idx+1];
                float nb = (float)src[n_idx+2];
                
                // Color Distance (Manhattan)
                float diff = fabsf(nr - center_r) + fabsf(ng - center_g) + fabsf(nb - center_b);
                
                // Weight Calculation (Similarity)
                float weight = expf(-diff / sigma_color);
                
                sum_r += nr * weight;
                sum_g += ng * weight;
                sum_b += nb * weight;
                total_w += weight;
            }
        }
    }
    
    float smooth_r = sum_r / total_w;
    float smooth_g = sum_g / total_w;
    float smooth_b = sum_b / total_w;
    
    // Mix based on user strength parameter (0.0 ~ 1.0)
    dst[idx+0] = (unsigned char)(center_r * (1.0f - strength) + smooth_r * strength);
    dst[idx+1] = (unsigned char)(center_g * (1.0f - strength) + smooth_g * strength);
    dst[idx+2] = (unsigned char)(center_b * (1.0f - strength) + smooth_b * strength);
}
'''