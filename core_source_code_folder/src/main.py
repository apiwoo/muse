# Project MUSE - main.py
# The Visual Singularity Engine Entry Point
# (C) 2025 MUSE Corp. All rights reserved.

import time
import sys
import cv2
import os # 파일 저장을 위해 추가

# High-Performance GPU Library
try:
    import cupy as cp
except ImportError:
    cp = None

# 모듈 경로 추가
sys.path.append('src')

# [CRITICAL FIX] CUDA/cuDNN DLL 경로 강제 주입
# 이 코드가 없으면 onnxruntime-gpu가 설치되어 있어도 DLL을 못 찾아서 CPU로 돕니다.
from utils.cuda_helper import setup_cuda_environment
setup_cuda_environment()

from core.input_manager import InputManager
from core.virtual_cam import VirtualCamera
from ai.tracking.facemesh import FaceMesh

def main():
    print("========================================")
    print("   Project MUSE - Engine Start (v1.9)")
    print("   Target: RTX 3060 / Mode A")
    print("   Feature: CUDA Fix + Index Debugging")
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
        
    except Exception as e:
        print(f"❌ 초기화 중 치명적 오류 발생: {e}")
        return

    print("\n🚀 파이프라인 가동 시작... (Press 'q' to Stop)")
    print("📸 [Info] 얼굴이 감지되면 'debug_snapshot.jpg'를 자동 저장합니다.")
    
    prev_time = time.time()
    frame_count = 0
    snapshot_taken = False # 스냅샷 찍었는지 여부
    
    try:
        while True:
            # [Step 1] Input
            frame_gpu, ret = input_mgr.read()
            if not ret:
                time.sleep(0.01)
                continue

            # [Step 2] AI Processing
            if cp and hasattr(frame_gpu, 'get'):
                frame_cpu = frame_gpu.get()
            else:
                frame_cpu = frame_gpu

            # 얼굴 분석
            faces = tracker.process(frame_cpu)
            
            # [Debug] 인덱스 번호 시각화 & 스냅샷 저장
            # 얼굴이 있고, 아직 스냅샷을 안 찍었다면
            if faces and not snapshot_taken:
                # 1. 인덱스 그리기 (이 프레임은 무거워도 상관없음)
                debug_frame = frame_cpu.copy()
                tracker.draw_indices_debug(debug_frame, faces)
                
                # 2. 파일로 저장
                cv2.imwrite("debug_snapshot.jpg", debug_frame)
                print("✅ [Snapshot] 'debug_snapshot.jpg' 저장 완료! 확인해보세요.")
                snapshot_taken = True
            
            # 평소에는 가벼운 Mesh만 그리기 (FPS 확보)
            if snapshot_taken:
                tracker.draw_mesh_debug(frame_cpu, faces)
            else:
                # 스냅샷 찍기 전까진 인덱스 보여주기
                tracker.draw_indices_debug(frame_cpu, faces)
            
            output_frame = frame_cpu

            # [Step 3] Output
            virtual_cam.send(output_frame)
            cv2.imshow("MUSE Preview", output_frame)

            # [Step 4] FPS Calculation
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