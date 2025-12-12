# Project MUSE - cuda_kernels.py
# High-Fidelity Kernels (Alpha Blending & TPS Warping)
# Updated: Ghosting Fix V8 (Absolute Zero Threshold)
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

# [Kernel 2] Smart Composite (Ghosting Fix V8 - Absolute Zero Logic)
# 변경점: '몸' 판단 기준을 0.05(5%)에서 0.0(0%)으로 변경했습니다.
# 이제 알파값이 1(약 0.0039)이라도 있으면 무조건 몸으로 인식합니다.
# "0이 아니면 안 되는 이유" -> 0.05로 설정하면 1~12값(희미한 잔상)이 무시되어 잔상이 남기 때문입니다.
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

    // --- 1. Original Alpha (Start State) ---
    float original_alpha = 0.0f;
    if (use_bg) {
        original_alpha = (float)mask[idx] / 255.0f;
    }

    // --- 2. Warped Alpha (End State) ---
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

    // --- 3. Base Layer Construction (Absolute Zero Decision) ---
    // [Threshold: 0.0f]
    // 8bit 이미지에서 1/255(약 0.004) 값이라도 존재하면 "몸"으로 판단합니다.
    // 아주 미세한 테두리 잔상(1~5%)까지 확실하게 잡기 위함입니다.
    
    bool was_body = (original_alpha > 0.0f); // 0이 아니면 무조건 Yes
    bool is_now_empty = (warped_alpha < 0.5f); // 지금 비어있는가?
    
    // [Logic] Gap Detection
    // 원래 몸이었는데(Was Body) && 지금 비어있다(Is Empty) = 잔상 발생 지점(Gap)
    // -> 이 경우에만 배경(BG)을 사용하여 덮습니다.
    bool is_gap = (was_body && is_now_empty);

    float base_b, base_g, base_r;

    if (is_gap) {
        // [CASE 1] 잔상 부위 -> 미리 찍어둔 배경(Clean Plate)으로 강제 치환
        base_b = (float)bg[idx_rgb+0];
        base_g = (float)bg[idx_rgb+1];
        base_r = (float)bg[idx_rgb+2];
    } else {
        // [CASE 2] 그 외 (몸 안쪽이거나, 원래 배경이던 곳) -> 원본 영상(Live Feed) 유지
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