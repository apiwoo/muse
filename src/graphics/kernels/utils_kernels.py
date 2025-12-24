# Project MUSE - utils_kernels.py
# 유틸리티 CUDA 커널
# (C) 2025 MUSE Corp. All rights reserved.

# ==============================================================================
# [KERNEL 14] GPU Resize (Bilinear Interpolation)
# ==============================================================================
GPU_RESIZE_KERNEL_CODE = r'''
extern "C" __global__
void gpu_resize_kernel(
    const unsigned char* src,          // Input image (BGR)
    unsigned char* dst,                // Output image (BGR)
    int src_width, int src_height,
    int dst_width, int dst_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height) return;

    // Calculate source position (bilinear interpolation)
    float scale_x = (float)src_width / (float)dst_width;
    float scale_y = (float)src_height / (float)dst_height;

    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    // Clamp to valid range
    if (src_x < 0) src_x = 0;
    if (src_y < 0) src_y = 0;
    if (src_x > src_width - 1) src_x = src_width - 1;
    if (src_y > src_height - 1) src_y = src_height - 1;

    // Get integer and fractional parts
    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    float fx = src_x - x0;
    float fy = src_y - y0;

    // Bilinear interpolation for each channel
    int dst_idx = (y * dst_width + x) * 3;

    for (int c = 0; c < 3; c++) {
        float v00 = (float)src[(y0 * src_width + x0) * 3 + c];
        float v01 = (float)src[(y0 * src_width + x1) * 3 + c];
        float v10 = (float)src[(y1 * src_width + x0) * 3 + c];
        float v11 = (float)src[(y1 * src_width + x1) * 3 + c];

        float v0 = v00 * (1.0f - fx) + v01 * fx;
        float v1 = v10 * (1.0f - fx) + v11 * fx;
        float v = v0 * (1.0f - fy) + v1 * fy;

        dst[dst_idx + c] = (unsigned char)fminf(fmaxf(v, 0.0f), 255.0f);
    }
}
'''


# ==============================================================================
# [KERNEL 15] GPU Mask Resize (Single Channel)
# ==============================================================================
GPU_MASK_RESIZE_KERNEL_CODE = r'''
extern "C" __global__
void gpu_mask_resize_kernel(
    const unsigned char* src,          // Input mask
    unsigned char* dst,                // Output mask
    int src_width, int src_height,
    int dst_width, int dst_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height) return;

    float scale_x = (float)src_width / (float)dst_width;
    float scale_y = (float)src_height / (float)dst_height;

    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    if (src_x < 0) src_x = 0;
    if (src_y < 0) src_y = 0;
    if (src_x > src_width - 1) src_x = src_width - 1;
    if (src_y > src_height - 1) src_y = src_height - 1;

    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    float fx = src_x - x0;
    float fy = src_y - y0;

    float v00 = (float)src[y0 * src_width + x0];
    float v01 = (float)src[y0 * src_width + x1];
    float v10 = (float)src[y1 * src_width + x0];
    float v11 = (float)src[y1 * src_width + x1];

    float v0 = v00 * (1.0f - fx) + v01 * fx;
    float v1 = v10 * (1.0f - fx) + v11 * fx;
    float v = v0 * (1.0f - fy) + v1 * fy;

    dst[y * dst_width + x] = (unsigned char)fminf(fmaxf(v, 0.0f), 255.0f);
}
'''


# ==============================================================================
# [KERNEL 16] Final Blend with Color Grading
# ==============================================================================
FINAL_BLEND_KERNEL_CODE = r'''
extern "C" __global__
void final_blend_kernel(
    const unsigned char* original,     // Original frame (BGR)
    const unsigned char* smoothed,     // Smoothed frame (BGR)
    const unsigned char* mask,         // Blend mask (0-255)
    unsigned char* dst,                // Output (BGR)
    int width, int height,
    float blend_strength,              // Overall smoothing strength (0-1)
    float color_temp,                  // Color temperature adjustment (-1 to 1)
    float color_tint                   // Color tint adjustment (-1 to 1)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idx3 = idx * 3;

    // Get mask value and apply overall strength
    float mask_val = (float)mask[idx] / 255.0f * blend_strength;

    // Get original and smoothed values
    float orig_b = (float)original[idx3 + 0];
    float orig_g = (float)original[idx3 + 1];
    float orig_r = (float)original[idx3 + 2];

    float smooth_b = (float)smoothed[idx3 + 0];
    float smooth_g = (float)smoothed[idx3 + 1];
    float smooth_r = (float)smoothed[idx3 + 2];

    // Blend smoothed with original based on mask
    float blend_b = orig_b * (1.0f - mask_val) + smooth_b * mask_val;
    float blend_g = orig_g * (1.0f - mask_val) + smooth_g * mask_val;
    float blend_r = orig_r * (1.0f - mask_val) + smooth_r * mask_val;

    // Apply color temperature (warm = more red/yellow, cool = more blue)
    if (fabsf(color_temp) > 0.01f) {
        if (color_temp > 0.0f) {
            // Warm: increase red, decrease blue
            blend_r = fminf(blend_r + color_temp * 30.0f, 255.0f);
            blend_b = fmaxf(blend_b - color_temp * 20.0f, 0.0f);
        } else {
            // Cool: increase blue, decrease red
            blend_b = fminf(blend_b - color_temp * 30.0f, 255.0f);
            blend_r = fmaxf(blend_r + color_temp * 20.0f, 0.0f);
        }
    }

    // Apply color tint (positive = magenta, negative = green)
    if (fabsf(color_tint) > 0.01f) {
        if (color_tint > 0.0f) {
            // Magenta: increase red and blue, decrease green
            blend_r = fminf(blend_r + color_tint * 15.0f, 255.0f);
            blend_b = fminf(blend_b + color_tint * 15.0f, 255.0f);
            blend_g = fmaxf(blend_g - color_tint * 20.0f, 0.0f);
        } else {
            // Green: decrease red and blue, increase green
            blend_g = fminf(blend_g - color_tint * 20.0f, 255.0f);
            blend_r = fmaxf(blend_r + color_tint * 10.0f, 0.0f);
            blend_b = fmaxf(blend_b + color_tint * 10.0f, 0.0f);
        }
    }

    // Output
    dst[idx3 + 0] = (unsigned char)fminf(fmaxf(blend_b, 0.0f), 255.0f);
    dst[idx3 + 1] = (unsigned char)fminf(fmaxf(blend_g, 0.0f), 255.0f);
    dst[idx3 + 2] = (unsigned char)fminf(fmaxf(blend_r, 0.0f), 255.0f);
}
'''
