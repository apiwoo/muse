# Project MUSE - cuda_kernels.py
# High-Fidelity Kernels (Alpha Blending & TPS Warping)
# Updated: Tri-Masking with Bilinear Interpolation (Precision Fix)
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

# [Kernel 2] Smart Composite (Tri-Masking & Bilinear Interpolation)
# Fixed: "Grid Misalignment" by adding Bilinear Interpolation.
# Logic:
# 1. Bilinear Interpolation: Smoothly fetch dx/dy from the small grid.
# 2. Tri-Masking: Composes (Warped FG) + (Hole-Filled BG) + (Live Source).
# 3. Precision: Uses float coordinates to prevent blocky artifacts.
COMPOSITE_KERNEL_CODE = r'''
// Helper: Bilinear Interpolation for smooth grid sampling
__device__ float get_interpolated_value(const float* data, int w, int h, float x, float y) {
    int x0 = (int)floorf(x);
    int x1 = x0 + 1;
    int y0 = (int)floorf(y);
    int y1 = y0 + 1;

    // Clamp coordinates
    if (x0 < 0) x0 = 0; if (x1 >= w) x1 = w - 1;
    if (y0 < 0) y0 = 0; if (y1 >= h) y1 = h - 1;

    // Weights
    float wx = x - (float)x0;
    float wy = y - (float)y0;

    // Fetch values
    float v00 = data[y0 * w + x0];
    float v10 = data[y0 * w + x1];
    float v01 = data[y1 * w + x0];
    float v11 = data[y1 * w + x1];

    // Interpolate
    float top = v00 * (1.0f - wx) + v10 * wx;
    float bottom = v01 * (1.0f - wx) + v11 * wx;
    
    return top * (1.0f - wy) + bottom * wy;
}

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

    // --- 1. Get Original Alpha (Where the person WAS) ---
    float original_alpha = 0.0f;
    if (use_bg) {
        original_alpha = (float)mask[idx] / 255.0f;
    }

    // --- 2. Calculate Precise Warped Coordinates (Bilinear) ---
    // Map current pixel (x,y) to small grid coordinates (float)
    float map_scale = 1.0f / (float)scale;
    float sx = (float)x * map_scale;
    float sy = (float)y * map_scale;

    // Fetch smooth dx, dy from small grid
    float shift_x = get_interpolated_value(dx_small, small_width, small_height, sx, sy);
    float shift_y = get_interpolated_value(dy_small, small_width, small_height, sx, sy);

    // Apply scaling to shift (since dx/dy are in small grid units? No, typically stored as-is or scaled. 
    // Assuming dx/dy from warp_kernel are in 'pixel units' of the small grid, we need to scale them up.)
    // *Correction*: In warp_kernel, dx/dy are calculated in 'small grid' pixel units.
    // To apply to 'large grid', we must multiply by 'scale'.
    float u_float = (float)x + shift_x * (float)scale;
    float v_float = (float)y + shift_y * (float)scale;

    int u = (int)(u_float + 0.5f); // Round to nearest pixel
    int v = (int)(v_float + 0.5f);

    // --- 3. Fetch Warped Data ---
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

    // --- 4. Calculate Tri-Mask Weights (with Epsilon) ---
    // Epsilon to prevent noise/flickering
    float epsilon = 0.01f;

    // Weight 1: Hole (Original - Warped)
    float hole_weight = 0.0f;
    if (original_alpha > (warped_alpha + epsilon)) {
        hole_weight = original_alpha - warped_alpha;
    }

    // Weight 2: Body (Warped)
    float body_weight = warped_alpha;

    // Weight 3: Source (Rest)
    // Ensure weights sum to 1.0
    float src_weight = 1.0f - body_weight - hole_weight;
    if (src_weight < 0.0f) src_weight = 0.0f;


    // --- 5. Final Composite ---
    float final_b = 0.0f, final_g = 0.0f, final_r = 0.0f;

    // A. Add Body (Foreground)
    if (body_weight > 0.0f) {
        final_b += fg_b * body_weight;
        final_g += fg_g * body_weight;
        final_r += fg_r * body_weight;
    }

    // B. Add Hole (Clean Plate)
    if (hole_weight > 0.0f) {
        float bg_b = (float)bg[idx_rgb+0];
        float bg_g = (float)bg[idx_rgb+1];
        float bg_r = (float)bg[idx_rgb+2];
        
        final_b += bg_b * hole_weight;
        final_g += bg_g * hole_weight;
        final_r += bg_r * hole_weight;
    }

    // C. Add Source (Live Video)
    if (src_weight > 0.0f) {
        float src_b = (float)src[idx_rgb+0];
        float src_g = (float)src[idx_rgb+1];
        float src_r = (float)src[idx_rgb+2];

        final_b += src_b * src_weight;
        final_g += src_g * src_weight;
        final_r += src_r * src_weight;
    }

    dst[idx_rgb+0] = (unsigned char)final_b;
    dst[idx_rgb+1] = (unsigned char)final_g;
    dst[idx_rgb+2] = (unsigned char)final_r;
}
'''