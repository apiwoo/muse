# Project MUSE - cuda_kernels.py
# High-Fidelity Kernels (Alpha Blending & TPS Warping)
# Updated: "No-Distortion" Logic (Differential Blending)
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

# [Kernel 2] Smart Composite (Differential Logic)
# 변경점: OriginalAlpha와 WarpedAlpha를 비교하여, 
# "알파값이 줄어든 만큼(Hole)"만 배경을 보여줍니다. 
# 변화가 없으면(No Morph) 배경을 전혀 섞지 않고 100% 라이브 영상을 사용합니다.
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

    // --- 1. Get Alpha Values ---
    
    // A. Original Alpha (Current Position)
    float original_alpha = 0.0f;
    if (use_bg) {
        original_alpha = (float)mask[idx] / 255.0f;
    }

    // B. Warped Alpha (Source Position)
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
        
        // Fetch Warped Foreground
        if (warped_alpha > 0.0f) {
            fg_b = (float)src[warped_idx_rgb+0];
            fg_g = (float)src[warped_idx_rgb+1];
            fg_r = (float)src[warped_idx_rgb+2];
        }
    }

    // --- 2. Calculate "Hole" Visibility (Differential) ---
    // 구멍(Hole)이란? "원래 사람이었는데(Original), 워핑 후 사람이 없어진(Warped) 정도"
    // 예: Original=1.0, Warped=0.0 -> Hole=1.0 (배경 100% 필요)
    // 예: Original=0.5, Warped=0.5 -> Hole=0.0 (배경 필요 없음, 라이브 영상 유지)
    
    float hole_alpha = 0.0f;
    if (original_alpha > warped_alpha) {
        hole_alpha = original_alpha - warped_alpha;
    }

    // --- 3. Base Layer Construction ---
    // Hole만큼만 Static BG를 보여주고, 나머지는 Live Src를 보여줍니다.
    // 이렇게 하면 성형 안 한 부분(Hole=0)은 100% Live Src가 되어 왜곡이 사라집니다.
    
    float src_live_b = (float)src[idx_rgb+0];
    float src_live_g = (float)src[idx_rgb+1];
    float src_live_r = (float)src[idx_rgb+2];
    
    float base_b = src_live_b;
    float base_g = src_live_g;
    float base_r = src_live_r;

    // Only read BG if we actually have a hole
    if (hole_alpha > 0.0f) {
        float bg_b = (float)bg[idx_rgb+0];
        float bg_g = (float)bg[idx_rgb+1];
        float bg_r = (float)bg[idx_rgb+2];
        
        // Base = BG * Hole + Src * (1 - Hole)
        // (여기서 Src는 Live Src입니다. 즉, 구멍 난 곳만 BG로 메꿈)
        base_b = bg_b * hole_alpha + src_live_b * (1.0f - hole_alpha);
        base_g = bg_g * hole_alpha + src_live_g * (1.0f - hole_alpha);
        base_r = bg_r * hole_alpha + src_live_r * (1.0f - hole_alpha);
    }

    // --- 4. Final Composite ---
    // Final = Foreground + Base(Behind)
    // Foreground는 WarpedAlpha만큼, Base는 나머지(1-WarpedAlpha)만큼 보입니다.
    
    dst[idx_rgb+0] = (unsigned char)(fg_b * warped_alpha + base_b * (1.0f - warped_alpha));
    dst[idx_rgb+1] = (unsigned char)(fg_g * warped_alpha + base_g * (1.0f - warped_alpha));
    dst[idx_rgb+2] = (unsigned char)(fg_r * warped_alpha + base_r * (1.0f - warped_alpha));
}
'''