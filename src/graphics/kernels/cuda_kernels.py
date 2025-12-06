# Project MUSE - cuda_kernels.py
# High-Fidelity Kernels (Alpha Blending & TPS Warping)
# (C) 2025 MUSE Corp. All rights reserved.

# [Kernel 1] Grid Generation (TPS Logic) - 기존 유지
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

# [Kernel 2] Alpha Compositing (Soft Edge Support)
# MODNet의 Soft Alpha를 지원하기 위해 로직을 완전히 변경했습니다.
COMPOSITE_KERNEL_CODE = r'''
extern "C" __global__
void composite_kernel(
    const unsigned char* src,  
    const unsigned char* mask, // Alpha Matte (0~255)
    const unsigned char* bg,   
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

    // 1. Warping Field Interpolation
    int sx = x / scale;
    int sy = y / scale;
    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;
    int s_idx = sy * small_width + sx;

    float shift_x = dx_small[s_idx] * (float)scale;
    float shift_y = dy_small[s_idx] * (float)scale;

    // Source Coordinates with Deformation
    int u = (int)(x + shift_x);
    int v = (int)(y + shift_y);

    int dst_idx = (y * width + x) * 3;
    int bg_idx = dst_idx; // Background matches destination coords

    // Boundary Check (If warped pixel is outside, show BG)
    if (u < 0 || u >= width || v < 0 || v >= height) {
        if (use_bg) {
            dst[dst_idx+0] = bg[bg_idx+0];
            dst[dst_idx+1] = bg[bg_idx+1];
            dst[dst_idx+2] = bg[bg_idx+2];
        } else {
            dst[dst_idx+0] = 0; dst[dst_idx+1] = 0; dst[dst_idx+2] = 0;
        }
        return;
    }

    int src_idx = (v * width + u) * 3;
    int mask_idx = v * width + u;

    // 2. Alpha Blending Logic
    float alpha = 0.0f;
    
    // Mask Value (0~255) -> Alpha (0.0~1.0)
    if (use_bg) {
        alpha = (float)mask[mask_idx] / 255.0f;
    } else {
        // No background mode: just copy source
        alpha = 1.0f; 
    }

    // [Optimization] Branchless Mixing
    // Out = Src * Alpha + BG * (1 - Alpha)
    float src_b = (float)src[src_idx+0];
    float src_g = (float)src[src_idx+1];
    float src_r = (float)src[src_idx+2];

    float bg_b = (float)bg[bg_idx+0];
    float bg_g = (float)bg[bg_idx+1];
    float bg_r = (float)bg[bg_idx+2];

    dst[dst_idx+0] = (unsigned char)(src_b * alpha + bg_b * (1.0f - alpha));
    dst[dst_idx+1] = (unsigned char)(src_g * alpha + bg_g * (1.0f - alpha));
    dst[dst_idx+2] = (unsigned char)(src_r * alpha + bg_r * (1.0f - alpha));
}
'''