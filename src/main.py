# Project MUSE - main.py
# The Visual Singularity Engine Entry Point (GUI Version)
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import time
import cv2
import numpy as np
import os
import signal
import threading

# [PySide6 GUI Framework]
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread, Signal, Slot, QMutex, QWaitCondition
import qdarktheme

# [System Path Setup]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# [MUSE Modules]
from utils.cuda_helper import setup_cuda_environment
setup_cuda_environment()

from core.input_manager import InputManager
from core.virtual_cam import VirtualCamera
from ai.tracking.facemesh import FaceMesh
from ai.tracking.body_tracker import BodyTracker 
from graphics.beauty_engine import BeautyEngine
from ui.main_window import MainWindow

# High-Performance GPU Library Check
try:
    import cupy as cp
except ImportError:
    cp = None

# ==============================================================================
# [Helper] Visualization Utils (Moved from BodyTracker)
# ==============================================================================
def draw_body_skeleton(frame, keypoints):
    """
    [Visual Check] COCO 포맷(17 Keypoints) 뼈대 그리기
    Main Thread에서 직접 그리기 위해 클래스에서 분리된 함수입니다.
    """
    if keypoints is None:
        return frame

    # [Prod] 신뢰도 임계값
    CONF_THRESH = 0.4

    # 1. 점 찍기
    for i in range(17):
        # keypoints shape: (17, 3) -> [x, y, conf]
        if i >= len(keypoints): break
        
        x, y, conf = keypoints[i]
        
        # 좌표가 화면 밖이면 스킵
        h, w = frame.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        if conf > CONF_THRESH:
            # 관절마다 색깔 다르게 (좌:파랑, 우:빨강)
            color = (255, 100, 0) if i % 2 == 1 else (0, 100, 255)
            
            # 얼굴 부위(0~4: 코,눈,귀)는 노란색 계열로 강조
            if i <= 4: 
                color = (0, 255, 255) # Yellow
                radius = 4
            else:
                radius = 6 # 몸통은 좀 더 크게
            
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)
            
            # [Visual] 테두리 추가 (가시성 확보)
            cv2.circle(frame, (int(x), int(y)), radius+1, (255, 255, 255), 1)

    # 2. 선 연결 (Skeleton)
    skeleton = [
        # 팔
        (5, 7), (7, 9),       # 왼팔
        (6, 8), (8, 10),      # 오른팔
        # 다리
        (11, 13), (13, 15),   # 왼다리
        (12, 14), (14, 16),   # 오른다리
        # 몸통
        (5, 6),               # 어깨선
        (11, 12),             # 골반선
        (5, 11), (6, 12),     # 옆구리
        # 얼굴 (갈매기 모양)
        (0, 1), (0, 2),       # 코-눈
        (1, 3), (2, 4)        # 눈-귀
    ]

    for p1, p2 in skeleton:
        if p1 >= len(keypoints) or p2 >= len(keypoints): continue
        
        x1, y1, c1 = keypoints[p1]
        x2, y2, c2 = keypoints[p2]
        
        if c1 > CONF_THRESH and c2 > CONF_THRESH:
            # 얼굴 연결선은 얇게, 몸통은 굵게
            thickness = 2
            color = (0, 255, 0) # Green
            
            if p1 <= 4 and p2 <= 4:
                thickness = 1
                color = (100, 255, 100) # Light Green

            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    return frame

# ==============================================================================
# [Parallel Worker 1] Face Tracking Thread
# ==============================================================================
class FaceWorker(QThread):
    def __init__(self, shared_data_lock):
        super().__init__()
        self.running = True
        self.latest_frame = None
        self.result = []
        self.lock = shared_data_lock
        self.new_frame_event = threading.Event()

    def update_frame(self, frame):
        """메인 스레드에서 최신 프레임을 밀어넣어줌"""
        self.latest_frame = frame
        self.new_frame_event.set()

    def run(self):
        print("🧠 [FaceWorker] 독립 스레드 시작 (InsightFace)")
        # [Critical] 스레드 내부에서 엔진 초기화 (GPU 컨텍스트 분리)
        tracker = FaceMesh(root_dir="assets")
        
        while self.running:
            # 새 프레임이 올 때까지 대기 (CPU 낭비 방지)
            if not self.new_frame_event.wait(timeout=0.1):
                continue
            
            self.new_frame_event.clear()
            frame = self.latest_frame
            
            if frame is None:
                continue

            # 추론 수행
            faces = tracker.process(frame)
            
            # 결과 업데이트 (Atomic에 가깝지만 안전하게 Lock 사용 가능, 여기선 속도위해 직접할당)
            # 리스트 교체는 파이썬에서 Atomic하므로 Lock 없이도 꽤 안전함
            self.result = faces

        print("🧠 [FaceWorker] 종료")

# ==============================================================================
# [Parallel Worker 2] Body Tracking Thread (Heavy)
# ==============================================================================
class BodyWorker(QThread):
    def __init__(self, shared_data_lock):
        super().__init__()
        self.running = True
        self.latest_frame = None
        self.result = None
        self.lock = shared_data_lock
        self.new_frame_event = threading.Event()

    def update_frame(self, frame):
        self.latest_frame = frame
        self.new_frame_event.set()

    def run(self):
        print("💪 [BodyWorker] 독립 스레드 시작 (ViTPose TensorRT)")
        # [Critical] TensorRT 엔진은 반드시 해당 스레드에서 로드해야 함
        tracker = BodyTracker()
        
        while self.running:
            if not self.new_frame_event.wait(timeout=0.1):
                continue
                
            self.new_frame_event.clear()
            frame = self.latest_frame
            
            if frame is None:
                continue

            # 추론 수행 (무거움)
            # TensorRT가 GPU를 쓰지만, Python 스레드는 여기서 대기함
            body_pts = tracker.process(frame)
            
            # 결과가 유효할 때만 업데이트
            if body_pts is not None:
                self.result = body_pts

        print("💪 [BodyWorker] 종료")

# ==============================================================================
# [Main Worker] Rendering & Orchestration
# ==============================================================================
class BeautyWorker(QThread):
    """
    [Main Orchestrator]
    - 역할: 카메라 캡처 -> Face/Body 스레드에 프레임 전달 -> 최신 결과 취합 -> 렌더링 -> 송출
    - 특징: AI 추론 속도와 관계없이 카메라 FPS(30~60)를 유지함 (비동기 렌더링)
    """
    frame_processed = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = True
        self.params = {
            'eye_scale': 0.0, 
            'face_v': 0.0,
            'waist_slim': 0.0,
            'show_body_debug': False
        }
        
        # Resources
        self.input_mgr = None
        self.virtual_cam = None
        self.beauty_engine = None
        
        # Parallel Workers
        self.lock = QMutex()
        self.face_worker = FaceWorker(self.lock)
        self.body_worker = BodyWorker(self.lock)

        # Settings
        self.DEVICE_ID = 0
        self.WIDTH = 1920
        self.HEIGHT = 1080
        self.FPS = 30

    def run(self):
        print("🧵 [MainWorker] 렌더링 스레드 시작")

        try:
            # 1. Main Resource Init
            self.input_mgr = InputManager(device_id=self.DEVICE_ID, width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)
            self.virtual_cam = VirtualCamera(width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)
            self.beauty_engine = BeautyEngine()
            
            # 2. Start Sub-Workers (AI Engines)
            self.face_worker.start()
            self.body_worker.start()
            
        except Exception as e:
            print(f"❌ [MainWorker] 초기화 실패: {e}")
            return

        prev_time = time.time()
        frame_count = 0

        while self.running:
            # [Step 1] Input Capture (Blocking but fast)
            if self.input_mgr:
                frame_gpu, ret = self.input_mgr.read()
            else:
                break
                
            if not ret:
                self.msleep(5)
                continue

            # GPU -> CPU (Rendering은 CPU/OpenCV로 진행)
            # 추후 전체 파이프라인 GPU화 가능하지만, 현재는 호환성 위주
            if cp and hasattr(frame_gpu, 'get'):
                frame_cpu = frame_gpu.get()
            else:
                frame_cpu = frame_gpu

            # [Step 2] Dispatch to AI Workers (Non-Blocking)
            # 현재 프레임을 AI 스레드들에게 던져주고, 메인은 기다리지 않고 넘어갑니다.
            # 복사본을 넘겨야 렌더링 중인 프레임이 오염되지 않음 (비용 발생하지만 안전)
            inference_frame = frame_cpu.copy() 
            self.face_worker.update_frame(inference_frame)
            self.body_worker.update_frame(inference_frame)

            # [Step 3] Fetch Latest Results (Instant)
            # "지금 당장 준비된" 가장 최신 데이터를 가져옵니다.
            latest_faces = self.face_worker.result
            latest_body = self.body_worker.result

            # [Step 4] Render (Beauty Engine)
            # 이전 프레임의 결과일 수 있지만, 30fps에서는 큰 차이 없음 (Ghosting 최소화)
            if self.beauty_engine:
                frame_cpu = self.beauty_engine.process(frame_cpu, latest_faces, latest_body, self.params)

            # [Debug] Body Skeleton (Enabled)
            # 메인 스레드에서 직접 그립니다.
            if self.params.get('show_body_debug', False):
                frame_cpu = draw_body_skeleton(frame_cpu, latest_body)

            # [Step 5] Output
            if self.virtual_cam:
                self.virtual_cam.send(frame_cpu)
            
            self.frame_processed.emit(frame_cpu)

            # [Step 6] FPS Log
            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 1.0:
                print(f"⚡ FPS: {frame_count} (Target: {self.FPS}) | Face: {'OK' if latest_faces else 'None'} | Body: {'OK' if latest_body is not None else 'None'}")
                frame_count = 0
                prev_time = curr_time

        self.cleanup()
        print("🧵 [MainWorker] 종료")

    def cleanup(self):
        print("🧹 [MainWorker] 정리 시작...")
        
        # Stop Workers
        if self.face_worker:
            self.face_worker.running = False
            self.face_worker.new_frame_event.set() # 깨우기
            self.face_worker.wait()
        
        if self.body_worker:
            self.body_worker.running = False
            self.body_worker.new_frame_event.set() # 깨우기
            self.body_worker.wait()

        if self.input_mgr:
            self.input_mgr.release()
            self.input_mgr = None
        
        if self.virtual_cam:
            self.virtual_cam.close()
            self.virtual_cam = None
            
        print("✨ [MainWorker] 정리 완료")

    @Slot(dict)
    def update_params(self, new_params):
        self.params = new_params.copy()

    def stop(self):
        self.running = False

def main():
    def signal_handler(sig, frame):
        print(f"\n🛑 [System] 종료 시그널 감지 ({sig})")
        if QApplication.instance():
            QApplication.instance().quit()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    app.setQuitOnLastWindowClosed(True)

    window = MainWindow()
    worker = BeautyWorker()
    worker.setTerminationEnabled(True)
    
    window.connect_worker(worker)
    worker.start()
    window.show()
    
    print("🚀 [System] MUSE GUI (Fully Parallel Mode) 가동.")
    
    app.exec()
    
    print("🛑 [System] 종료 시퀀스 진입...")
    if worker.isRunning():
        worker.stop()
        if not worker.wait(1000):
            print("⚠️ 강제 종료")
            worker.terminate()
    
    print("💀 프로세스 종료")
    os._exit(0)

if __name__ == "__main__":
    main()