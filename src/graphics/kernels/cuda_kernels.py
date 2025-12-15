# Project MUSE - cuda_kernels.py
# High-Fidelity Kernels (Alpha Blending & TPS Warping)
# Updated: Topology Protection V20 (Deep Interior Protection using Neighbor Check)
# (C) 2025 MUSE Corp. All rights reserved.

# [Kernel 1] Grid Generation (TPS Logic) - 변경 없음
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

# [Kernel 2] Smart Composite (Deep Interior Protection)
# "사람 덩어리 안쪽은 배경이 절대 개입할 수 없다"는 논리를 물리적으로 구현.
# 상하좌우 주변 픽셀을 검사하여, '완전한 내부(Deep Interior)'로 판단되면 배경 합성을 원천 차단함.
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

    // --- 2. Deep Interior Check (Topology Protection) ---
    // 핵심 논리: "배경이 들어올 경로가 없다면(안쪽이라면) 잘못된 상황이다."
    // 현재 픽셀을 기준으로 상하좌우 일정 거리(margin)를 확인합니다.
    // 4방향 모두 '살(Body)'이라면, 이곳은 '깊은 내부(Deep Interior)'입니다.
    // 깊은 내부에서는 워핑 맵이 꼬이든 말든 절대 배경이 튀어나오면 안 됩니다.
    
    bool is_deep_interior = false;
    
    if (use_bg && original_alpha > 0.95f) { // 일단 현재 위치가 확실한 살이어야 함
        int margin = 5; // 5픽셀 두께의 보호막 (이 값보다 안쪽은 절대 보호됨)
        bool safe_left = false, safe_right = false, safe_top = false, safe_bottom = false;

        // 경계 검사 포함하여 주변 마스크 확인
        // (255에 가까운 값이면 살, 0에 가까우면 배경)
        if (x - margin >= 0 && mask[idx - margin] > 200) safe_left = true;
        if (x + margin < width && mask[idx + margin] > 200) safe_right = true;
        if (y - margin >= 0 && mask[(y - margin) * width + x] > 200) safe_top = true;
        if (y + margin < height && mask[(y + margin) * width + x] > 200) safe_bottom = true;

        // 4방향 모두 살로 막혀있다면 -> 여기는 뚫려선 안 되는 내부다.
        if (safe_left && safe_right && safe_top && safe_bottom) {
            is_deep_interior = true;
        }
    }

    // --- 3. Warped Alpha Calculation ---
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
        
        // 워핑된 위치의 원본 색상 가져오기
        fg_b = (float)src[warped_idx_rgb+0];
        fg_g = (float)src[warped_idx_rgb+1];
        fg_r = (float)src[warped_idx_rgb+2];
    }

    // --- 4. Final Decision (Ghosting Prevention) ---
    
    // [Loss Check] 원래는 살이었는데, 워핑 후 빈 공간(배경)이 되었는가?
    bool is_alpha_loss = (original_alpha > warped_alpha + 0.05f);

    float base_b, base_g, base_r;

    // [Final Rule]
    // 1. 살이 깎여나간 상황이고 (is_alpha_loss)
    // 2. [New] 여기가 '깊은 내부'가 아니라면 (!is_deep_interior)
    // -> 즉, "피부의 가장자리"라면 배경을 채웁니다. (Slimming 적용)
    //
    // 반대로, '깊은 내부'라면 설령 is_alpha_loss가 True라도 (빠른 움직임으로 인한 오류),
    // 배경을 채우지 않고 원본(src)을 유지하여 번쩍임을 막습니다.
    
    if (is_alpha_loss && !is_deep_interior) {
        // 배경(Clean Plate) 합성
        base_b = (float)bg[idx_rgb+0];
        base_g = (float)bg[idx_rgb+1];
        base_r = (float)bg[idx_rgb+2];
    } else {
        // 원본(Live Feed) 유지
        // (내부에서 찢어짐이 발생해도 배경 대신 원본이 나옴 -> Ghosting 해결)
        base_b = (float)src[idx_rgb+0];
        base_g = (float)src[idx_rgb+1];
        base_r = (float)src[idx_rgb+2];
    }

    // --- 5. Composite ---
    dst[idx_rgb+0] = (unsigned char)(fg_b * warped_alpha + base_b * (1.0f - warped_alpha));
    dst[idx_rgb+1] = (unsigned char)(fg_g * warped_alpha + base_g * (1.0f - warped_alpha));
    dst[idx_rgb+2] = (unsigned char)(fg_r * warped_alpha + base_r * (1.0f - warped_alpha));
}
'''