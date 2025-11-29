# Project MUSE - main.py
# The Visual Singularity Engine Entry Point
# (C) 2025 MUSE Corp. All rights reserved.

import time
import sys

# High-Performance GPU Library
try:
    import cupy as cp
except ImportError:
    cp = None

# 모듈 경로 추가 (src 폴더 인식용)
sys.path.append('src')

from core.input_manager import InputManager
from core.virtual_cam import VirtualCamera

def main():
    print("========================================")
    print("   Project MUSE - Engine Start (v1.2)")
    print("   Target: RTX 3060 / Mode A")
    print("   Device: Logitech C920 (FPS Fix Applied)")
    print("========================================")

    # 1. 설정 (Configuration)
    DEVICE_ID = 1  
    WIDTH = 1920 
    HEIGHT = 1080
    FPS = 30       

    # 2. 모듈 초기화
    try:
        input_mgr = InputManager(device_id=DEVICE_ID, width=WIDTH, height=HEIGHT, fps=FPS)
        virtual_cam = VirtualCamera(width=WIDTH, height=HEIGHT, fps=FPS)
        
    except Exception as e:
        print(f"❌ 초기화 중 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🚀 파이프라인 가동 시작... (Press Ctrl+C to Stop)")
    
    prev_time = time.time()
    frame_count = 0
    
    try:
        while True:
            # [Step 1] Input
            frame_gpu, ret = input_mgr.read()
            
            if not ret:
                print("⚠️ 프레임 드랍 발생 (Camera Read Fail)")
                time.sleep(0.01)
                continue

            # [Step 2] AI Processing (Passthrough)
            # (테스트용) 작동 확인용 붉은 박스
            # if frame_gpu is not None:
            #     frame_gpu[0:50, 0:50, :] = cp.array([255, 0, 0], dtype=cp.uint8)

            # [Step 3] Output
            virtual_cam.send(frame_gpu)

            # [Step 4] FPS Calculation
            frame_count += 1
            curr_time = time.time()
            elapsed = curr_time - prev_time
            
            if elapsed >= 1.0:
                fps_val = frame_count / elapsed
                print(f"⚡ Pipeline FPS: {fps_val:.2f} (Target: {FPS})")
                
                # 조명이 너무 어두우면 경고 메시지 출력 (C920 특성 안내)
                if fps_val < 20:
                    print("   ⚠️ FPS가 낮습니다! 방의 조명을 더 밝게 하거나, input_manager.py의 노출(Exposure) 값을 조절하세요.")

                frame_count = 0
                prev_time = curr_time

    except KeyboardInterrupt:
        print("\n🛑 사용자 중단 요청 (KeyboardInterrupt)")
    
    finally:
        print("🧹 리소스 정리 중...")
        if 'input_mgr' in locals(): input_mgr.release()
        if 'virtual_cam' in locals(): virtual_cam.close()
        print("👋 MUSE Engine 종료.")

if __name__ == "__main__":
    main()