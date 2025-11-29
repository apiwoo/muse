# Project MUSE - main.py
# The Visual Singularity Engine Entry Point
# (C) 2025 MUSE Corp. All rights reserved.

import time
import sys
import cv2

# High-Performance GPU Library
try:
    import cupy as cp
except ImportError:
    cp = None

# 모듈 경로 추가
sys.path.append('src')

from core.input_manager import InputManager
from core.virtual_cam import VirtualCamera

# [수정] cm.py 구조에 맞게 임포트 경로 변경
from ai.tracking.facemesh import FaceMesh

def main():
    print("========================================")
    print("   Project MUSE - Engine Start (v1.5)")
    print("   Target: RTX 3060 / Mode A")
    print("   Feature: Face Tracking + Preview Window")
    print("========================================")

    # 1. 설정
    DEVICE_ID = 1  
    WIDTH = 1920 
    HEIGHT = 1080
    FPS = 30       

    # 2. 모듈 초기화
    try:
        # Input/Output
        input_mgr = InputManager(device_id=DEVICE_ID, width=WIDTH, height=HEIGHT, fps=FPS)
        virtual_cam = VirtualCamera(width=WIDTH, height=HEIGHT, fps=FPS)
        
        # AI Engine (Face)
        # assets/models 경로 지정
        tracker = FaceMesh(root_dir="assets")
        
    except Exception as e:
        print(f"❌ 초기화 중 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🚀 파이프라인 가동 시작... (Press 'q' to Stop)")
    
    prev_time = time.time()
    frame_count = 0
    
    try:
        while True:
            # [Step 1] Input
            frame_gpu, ret = input_mgr.read()
            
            if not ret:
                time.sleep(0.01)
                continue

            # ==========================================
            # [Step 2] AI Processing
            # ==========================================
            
            # InsightFace는 CPU(Numpy) 입력을 받으므로 변환
            # (추후 렌더링 단계에서는 GPU 데이터를 그대로 쓸 것임)
            if cp and hasattr(frame_gpu, 'get'):
                frame_cpu = frame_gpu.get()
            else:
                frame_cpu = frame_gpu

            # 얼굴 분석
            faces = tracker.process(frame_cpu)
            
            # [Debug] 시각화 (얼굴에 점 찍기)
            # 원본 이미지에 그림 (화면 송출용 + 미리보기용)
            tracker.draw_debug(frame_cpu, faces)
            
            # 출력용 프레임 설정
            output_frame = frame_cpu

            # ==========================================

            # [Step 3] Output (Dual)
            
            # 1. OBS 가상 카메라 송출
            virtual_cam.send(output_frame)
            
            # 2. [NEW] PC 화면 미리보기 창 표시
            cv2.imshow("MUSE Preview", output_frame)

            # [Step 4] FPS Calculation & Key Control
            frame_count += 1
            curr_time = time.time()
            elapsed = curr_time - prev_time
            
            if elapsed >= 1.0:
                fps_val = frame_count / elapsed
                face_count = len(faces)
                print(f"⚡ Pipeline FPS: {fps_val:.2f} (Faces: {face_count})")
                
                if fps_val < 20:
                    print("   ⚠️ Low FPS detected. Check lighting or GPU load.")

                frame_count = 0
                prev_time = curr_time
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 종료 키(q) 입력됨.")
                break

    except KeyboardInterrupt:
        print("\n🛑 사용자 중단 요청 (KeyboardInterrupt)")
    
    finally:
        print("🧹 리소스 정리 중...")
        if 'input_mgr' in locals(): input_mgr.release()
        if 'virtual_cam' in locals(): virtual_cam.close()
        cv2.destroyAllWindows() # 윈도우 닫기
        print("👋 MUSE Engine 종료.")

if __name__ == "__main__":
    main()