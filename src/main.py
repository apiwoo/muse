# Project MUSE - main.py
# The Visual Singularity Engine Entry Point
# (C) 2025 MUSE Corp. All rights reserved.

import time
import sys
import cv2
import os 

# High-Performance GPU Library
try:
    import cupy as cp
except ImportError:
    cp = None

# 모듈 경로 추가
sys.path.append('src')

# [CRITICAL FIX] CUDA/cuDNN DLL 경로 강제 주입
from utils.cuda_helper import setup_cuda_environment
setup_cuda_environment()

from core.input_manager import InputManager
from core.virtual_cam import VirtualCamera
from ai.tracking.facemesh import FaceMesh
# [New] BeautyEngine 추가
from graphics.beauty_engine import BeautyEngine

def main():
    print("========================================")
    print("   Project MUSE - Engine Start (v2.0)")
    print("   Target: RTX 3060 / Mode A")
    print("   Feature: Real-time Beauty (Eye+Jaw)")
    print("========================================")

    # 1. 설정
    DEVICE_ID = 1  
    WIDTH = 1920 
    HEIGHT = 1080
    FPS = 30       

    # 2. 모듈 초기화
    try:
        input_mgr = InputManager(device_id=DEVICE_ID, width=WIDTH, height=HEIGHT, fps=FPS)
        virtual_cam = VirtualCamera(width=WIDTH, height=HEIGHT, fps=FPS)
        tracker = FaceMesh(root_dir="assets")
        # [New] 성형 엔진 생성
        beauty_engine = BeautyEngine()
        
    except Exception as e:
        print(f"❌ 초기화 중 치명적 오류 발생: {e}")
        return

    print("\n🚀 파이프라인 가동 시작... (Press 'q' to Stop)")
    
    prev_time = time.time()
    frame_count = 0
    
    # [Test Params] 성형 강도 테스트 (GUI 연결 전 하드코딩)
    # eye_scale: 0.0 ~ 1.0 (클수록 왕눈이)
    # face_v: 0.0 ~ 1.0 (클수록 뾰족 턱)
    test_params = {'eye_scale': 0.3, 'face_v': 0.2}
    print(f"💅 적용된 성형값: {test_params}")

    try:
        while True:
            # [Step 1] Input
            frame_gpu, ret = input_mgr.read()
            if not ret:
                time.sleep(0.01)
                continue

            # [Step 2] AI Processing (Tracking)
            if cp and hasattr(frame_gpu, 'get'):
                frame_cpu = frame_gpu.get()
            else:
                frame_cpu = frame_gpu

            # 얼굴 분석
            faces = tracker.process(frame_cpu)
            
            # [Step 3] Beauty Processing (Warping)
            # 성형 엔진을 통과시켜 얼굴을 변형합니다.
            if faces:
                frame_cpu = beauty_engine.process(frame_cpu, faces, test_params)
            
            # (선택) 디버깅용 점은 이제 안 그려도 되지만, 확인용으로 켜둘 수 있음
            # tracker.draw_mesh_debug(frame_cpu, faces)
            
            output_frame = frame_cpu

            # [Step 4] Output
            virtual_cam.send(output_frame)
            cv2.imshow("MUSE Preview", output_frame)

            # [Step 5] FPS Calculation
            frame_count += 1
            curr_time = time.time()
            elapsed = curr_time - prev_time
            
            if elapsed >= 1.0:
                fps_val = frame_count / elapsed
                print(f"⚡ Pipeline FPS: {fps_val:.2f} (Faces: {len(faces)})")
                frame_count = 0
                prev_time = curr_time
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n🛑 사용자 중단 요청")
    
    finally:
        print("🧹 리소스 정리 중...")
        if 'input_mgr' in locals(): input_mgr.release()
        if 'virtual_cam' in locals(): virtual_cam.close()
        cv2.destroyAllWindows()
        print("👋 MUSE Engine 종료.")

if __name__ == "__main__":
    main()