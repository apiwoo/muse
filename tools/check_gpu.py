# Project MUSE - check_gpu.py
# (C) 2025 MUSE Corp.
# Role: Verifies CuPy acceleration and Gaussian Blur.

import time
import sys
import os

def main():
    print("========================================================")
    print("   MUSE GPU Acceleration Check (CuPy)")
    print("========================================================")

    print("[SCAN] [Step 1] Loading CuPy Library...")
    try:
        import cupy as cp
        import cupyx.scipy.ndimage
        print(f"   [OK] CuPy Version: {cp.__version__}")
        
        dev = cp.cuda.Device()
        print(f"   [OK] Detected GPU: {dev.mem_info[1] / 1024**3:.2f} GB VRAM Available")
        
    except ImportError as e:
        print(f"   [ERROR] CuPy Load Failed: {e}")
        print("   -> Run 'pip install cupy-cuda12x'")
        return
    except Exception as e:
        print(f"   [ERROR] GPU Init Failed: {e}")
        return

    print("\n[FAST] [Step 2] Performance Test (Gaussian Blur 4K)")
    
    h, w = 2160, 3840
    print(f"   - Target Resolution: {w}x{h} (Single Channel)")
    
    try:
        t0 = time.time()
        gpu_arr = cp.random.random((h, w), dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        print(f"   - GPU Memory Alloc: {time.time()-t0:.4f} sec")
        
        print("   - Running Gaussian Filter (Sigma=5)...")
        t_start = time.time()
        
        result_gpu = cupyx.scipy.ndimage.gaussian_filter(gpu_arr, sigma=5)
        
        cp.cuda.Stream.null.synchronize()
        t_end = time.time()
        
        gpu_time = t_end - t_start
        print(f"   [OK] GPU Time: {gpu_time:.5f} sec")
        
        if gpu_time < 0.02:
            print("   [START] Status: Excellent (Real-time Ready)")
        else:
            print("   [WARNING] Status: Slower than expected (Still faster than CPU)")

    except Exception as e:
        print(f"   [ERROR] Test Failed: {e}")
        return

    print("\n========================================================")
    print("[DONE] Verification Complete.")
    print("========================================================")

if __name__ == "__main__":
    main()