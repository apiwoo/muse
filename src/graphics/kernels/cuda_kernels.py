# Project MUSE - cuda_kernels.py
# CUDA Kernel Strings for Beauty Engine
# (C) 2025 MUSE Corp. All rights reserved.

# [Kernel 1] Grid Generation (TPS Logic)
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
            if (mode == 0) { // Expand/Shrink (Radial)
                acc_dx -= diff_x * factor;
                acc_dy -= diff_y * factor;
            } else if (mode == 1) { // Shift (Vector)
                acc_dx -= vx * factor * r * 0.5f;
                acc_dy -= vy * factor * r * 0.5f;
            } else if (mode == 2) { // Shrink (Inverse Radial)
                acc_dx += diff_x * factor;
                acc_dy += diff_y * factor;
            }
        }
    }
    dx[idx] += acc_dx;
    dy[idx] += acc_dy;
}
'''

# [Kernel 2] Smart Composite (The Clean Plate Logic)
COMPOSITE_KERNEL_CODE = r'''
extern "C" __global__
void composite_kernel(
    const unsigned char* src,  
    const unsigned char* mask, 
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

    // 1. Get Displacement (Bilinear Interpolation from small grid)
    int sx = x / scale;
    int sy = y / scale;
    if (sx >= small_width) sx = small_width - 1;
    if (sy >= small_height) sy = small_height - 1;
    int s_idx = sy * small_width + sx;

    float shift_x = dx_small[s_idx] * (float)scale;
    float shift_y = dy_small[s_idx] * (float)scale;

    // Source Coordinates (u, v)
    float u = (float)x + shift_x;
    float v = (float)y + shift_y;

    int u_i = (int)u;
    int v_i = (int)v;

    // Boundary Check
    if (u_i < 0 || u_i >= width || v_i < 0 || v_i >= height) {
        if (use_bg) {
            int idx = (y * width + x) * 3;
            dst[idx+0] = bg[idx+0];
            dst[idx+1] = bg[idx+1];
            dst[idx+2] = bg[idx+2];
        } else {
            int idx = (y * width + x) * 3;
            dst[idx] = 0; dst[idx+1] = 0; dst[idx+2] = 0;
        }
        return;
    }

    // 2. Check Mask at Source (u, v)
    int mask_idx = v_i * width + u_i;
    unsigned char m_val = mask[mask_idx];

    int dst_idx = (y * width + x) * 3;
    int src_idx = (v_i * width + u_i) * 3;
    int bg_idx = (y * width + x) * 3; // BG uses original coords

    if (m_val > 10) { 
        // Person -> Warp
        dst[dst_idx+0] = src[src_idx+0];
        dst[dst_idx+1] = src[src_idx+1];
        dst[dst_idx+2] = src[src_idx+2];
    } else {
        // Background -> Static Clean Plate
        if (use_bg) {
            dst[dst_idx+0] = bg[bg_idx+0];
            dst[dst_idx+1] = bg[bg_idx+1];
            dst[dst_idx+2] = bg[bg_idx+2];
        } else {
            dst[dst_idx+0] = src[src_idx+0]; // Fallback
            dst[dst_idx+1] = src[src_idx+1];
            dst[dst_idx+2] = src[src_idx+2];
        }
    }
}
'''