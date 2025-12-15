# Project MUSE - cuda_kernels.py
# High-Fidelity Kernels (Alpha Blending & TPS Warping)
# Updated: Topology Protection V21 (Intrusion Logic Fix)
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
# 워핑 벡터의 크기(Magnitude)를 핵심 판별 기준으로 사용.
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

    // [Core Logic] 변위량(Displacement) 계산
    // 워핑이 강하게 일어난 곳(=살을 깎은 곳)은 shift_sq가 큽니다.
    // 반면, 몸통 안쪽이나 배경은 shift_sq가 0에 가깝습니다.
    float shift_sq = shift_x * shift_x + shift_y * shift_y;
    
    // Threshold: 1.0 픽셀 이상 움직인 곳만 '유효한 보정'으로 인정
    // 이 값이 너무 크면 보정이 씹히고, 너무 작으면 노이즈가 생깁니다.
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
    
    // [Loss Check] 원래는 살이었는데, 워핑 후 빈 공간(배경)이 되었는가?
    // Margin을 두어 미세한 오차는 무시 (0.05)
    bool is_alpha_loss = (original_alpha > warped_alpha + 0.05f);

    float base_b, base_g, base_r;

    // [Final Rule]
    // Slimming(배경 합성)이 적용되려면 두 가지 조건이 모두 충족되어야 합니다.
    // 1. is_alpha_loss: 살이 깎여나갔음.
    // 2. is_significant_warp: 실제로 픽셀을 이동시킨 결과임. (단순 떨림 아님)
    //
    // 이 2번 조건이 "몸 안쪽 보호" 역할을 수행합니다.
    // 몸 안쪽은 워핑 벡터가 0에 가까우므로, 설령 알파값이 튀어서 is_alpha_loss가 True가 되더라도
    // is_significant_warp가 False가 되어 배경 합성을 막습니다.
    
    if (is_alpha_loss && is_significant_warp) {
        // [Case A] Valid Slimming -> 배경(Clean Plate) 합성
        base_b = (float)bg[idx_rgb+0];
        base_g = (float)bg[idx_rgb+1];
        base_r = (float)bg[idx_rgb+2];
    } else {
        // [Case B] No Change or Internal Noise -> 원본(Live Feed) 유지
        // 워핑이 없거나(몸 안쪽), 살이 늘어난 경우(Warped > Original) 등
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