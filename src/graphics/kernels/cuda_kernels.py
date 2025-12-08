# Project MUSE - cuda_kernels.py
# High-Fidelity Kernels (Alpha Blending & TPS Warping)
# Updated: Ghosting Fix (Partition of Unity Blending)
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

# [Kernel 2] Smart Composite (Ghosting Fix)
# 변경점: 배경 레이어(Base) 생성 로직을 'Hole(차이)' 기반에서 'Original Alpha(원본)' 기반으로 변경
# 이를 통해 인물이 조금이라도 있었던 픽셀은 Live 영상(인물 잔상 포함) 대신 Clean BG를 강제로 사용합니다.
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
    // 이 픽셀에 '원래 인물'이 얼마나 있었는지 확인합니다.
    float original_alpha = 0.0f;
    if (use_bg) {
        original_alpha = (float)mask[idx] / 255.0f;
    }

    // B. Warped Alpha (Source Position)
    // 성형 후 이 픽셀에 '변형된 인물'이 얼마나 오게 되는지 확인합니다.
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

    // --- 2. Base Layer Construction (Fixed Logic) ---
    // 잔상 문제 해결의 핵심입니다.
    // 기존 로직은 'Hole(차이)'만큼만 배경을 썼기 때문에, 차이가 적은 경계선에서
    // 원본 인물(Live Src)이 배경 레이어에 섞여 들어가는 문제가 있었습니다.
    
    // 수정된 로직: 
    // "원래 인물이 있던 자리(original_alpha > 0)"라면 -> 무조건 Clean BG를 사용.
    // "원래 인물이 없던 자리(original_alpha == 0)"라면 -> Live Src(배경)를 사용.
    
    float src_live_b = (float)src[idx_rgb+0];
    float src_live_g = (float)src[idx_rgb+1];
    float src_live_r = (float)src[idx_rgb+2];
    
    float base_b = src_live_b;
    float base_g = src_live_g;
    float base_r = src_live_r;

    // 만약 이 픽셀이 조금이라도 '원본 인물' 영역이었다면, 
    // 그 부분은 Live 화면(인물 포함) 대신 깨끗한 배경(BG)으로 덮어써야 합니다.
    if (original_alpha > 0.0f) {
        float bg_b = (float)bg[idx_rgb+0];
        float bg_g = (float)bg[idx_rgb+1];
        float bg_r = (float)bg[idx_rgb+2];
        
        // Linear Interpolation: Alpha만큼 BG를 섞습니다.
        // Alpha가 1.0이면 100% BG가 되어 인물이 완전히 지워집니다.
        base_b = bg_b * original_alpha + src_live_b * (1.0f - original_alpha);
        base_g = bg_g * original_alpha + src_live_g * (1.0f - original_alpha);
        base_r = bg_r * original_alpha + src_live_r * (1.0f - original_alpha);
    }

    // --- 3. Final Composite ---
    // 최종 결과 = (변형된 인물) + (배경 레이어 * 인물 투명도 반전)
    // Standard Over Operator: result = FG * alpha + BG * (1 - alpha)
    
    dst[idx_rgb+0] = (unsigned char)(fg_b * warped_alpha + base_b * (1.0f - warped_alpha));
    dst[idx_rgb+1] = (unsigned char)(fg_g * warped_alpha + base_g * (1.0f - warped_alpha));
    dst[idx_rgb+2] = (unsigned char)(fg_r * warped_alpha + base_r * (1.0f - warped_alpha));
}
'''