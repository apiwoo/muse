# Project MUSE - check_gpu.py
# (C) 2025 MUSE Corp.
# 역할: CuPy 가속 및 Gaussian Blur 기능이 정상 작동하는지 검증합니다.

import time
import sys
import os

def main():
    print("========================================================")
    print("   MUSE GPU Acceleration Check (CuPy)")
    print("========================================================")

    # 1. CuPy 임포트 확인
    print("🔍 [Step 1] CuPy 라이브러리 로딩 중...")
    try:
        import cupy as cp
        import cupyx.scipy.ndimage
        print(f"   ✅ CuPy Version: {cp.__version__}")
        
        # GPU 정보 출력
        dev = cp.cuda.Device()
        print(f"   ✅ Detected GPU: {dev.mem_info[1] / 1024**3:.2f} GB VRAM Available")
        
    except ImportError as e:
        print(f"   ❌ CuPy 로딩 실패: {e}")
        print("   👉 'pip install cupy-cuda12x'를 실행해주세요.")
        return
    except Exception as e:
        print(f"   ❌ GPU 초기화 실패: {e}")
        return

    # 2. 성능 테스트 (CPU vs GPU)
    print("\n⚡ [Step 2] 성능 비교 테스트 (Gaussian Blur 4K)")
    
    # 4K 해상도 더미 데이터 (Float32)
    h, w = 2160, 3840
    print(f"   - Target Resolution: {w}x{h} (Single Channel)")
    
    try:
        # 데이터 생성 (GPU)
        t0 = time.time()
        gpu_arr = cp.random.random((h, w), dtype=cp.float32)
        cp.cuda.Stream.null.synchronize() # 대기
        print(f"   - GPU Memory Alloc: {time.time()-t0:.4f} sec")
        
        # Gaussian Blur 실행
        print("   - Running Gaussian Filter (Sigma=5)...")
        t_start = time.time()
        
        # [핵심] 뷰티 엔진에서 사용할 함수
        result_gpu = cupyx.scipy.ndimage.gaussian_filter(gpu_arr, sigma=5)
        
        cp.cuda.Stream.null.synchronize() # 연산 완료 대기
        t_end = time.time()
        
        gpu_time = t_end - t_start
        print(f"   ✅ GPU 처리 시간: {gpu_time:.5f} sec")
        
        if gpu_time < 0.02: # 20ms 미만이면 합격 (30FPS 방어 가능)
            print("   🚀 상태: 아주 훌륭함 (Real-time Ready)")
        else:
            print("   ⚠️ 상태: 예상보다 느림 (그래도 CPU보단 빠를 것)")

    except Exception as e:
        print(f"   ❌ 테스트 중 오류 발생: {e}")
        return

    print("\n========================================================")
    print("🎉 검증 완료. 이제 'src/graphics/beauty_engine.py'를 수정해도 좋습니다.")
    print("========================================================")

if __name__ == "__main__":
    main()