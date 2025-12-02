# Project MUSE - main.py
# The Visual Singularity Engine Entry Point (Multi-Profile Edition)
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import time
import cv2
import numpy as np
import os
import signal

# [PySide6 GUI Framework]
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread, Signal, Slot, Qt
import qdarktheme

# [System Path Setup]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# [MUSE Modules]
from utils.cuda_helper import setup_cuda_environment
setup_cuda_environment()

from utils.config import ProfileManager
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

class BeautyWorker(QThread):
    """
    [Background Thread]
    영상 처리 및 프로파일 스위칭 로직 수행
    - Input -> AI -> Render -> Output 파이프라인
    - ProfileManager를 통한 설정값 관리 및 자동 동기화
    """
    frame_processed = Signal(np.ndarray)       # 처리된 영상 -> UI 표시용
    slider_sync_requested = Signal(dict)       # 설정 로드됨 -> UI 슬라이더 동기화 요청

    def __init__(self):
        super().__init__()
        self.running = True
        self.should_reset_bg = False
        
        # 1. Profile System 초기화
        self.profile_mgr = ProfileManager()
        self.profiles = self.profile_mgr.get_profile_list()
        
        # 초기 프로파일 설정 (없으면 default)
        self.current_profile_name = "default"
        if self.profiles:
            self.current_profile_name = self.profiles[0]
        
        self.active_profile_idx = 0

        # Config에서 초기 파라미터 로드
        initial_config = self.profile_mgr.get_config(self.current_profile_name)
        self.params = initial_config.get("params", {})
        
        # Resources (나중에 run에서 초기화)
        self.input_mgr = None
        self.virtual_cam = None
        self.tracker = None
        self.body_tracker = None
        self.beauty_engine = None
        
        # System Settings
        self.WIDTH = 1920
        self.HEIGHT = 1080
        self.FPS = 30

    def run(self):
        print(f"🧵 [Worker] MUSE Engine Start (Active: {self.current_profile_name})")

        try:
            # 1. Input Manager: 필요한 모든 카메라 연결
            # 각 프로파일별로 할당된 camera_id 수집
            required_cams = []
            for p in self.profiles:
                cfg = self.profile_mgr.get_config(p)
                cid = cfg.get("camera_id", 0)
                required_cams.append(cid)
            
            # 중복 제거 후 InputManager 초기화 (Warm-up 포함)
            self.input_mgr = InputManager(camera_indices=required_cams, width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)
            
            # 초기 활성 카메라 설정
            init_cid = self.profile_mgr.get_config(self.current_profile_name).get("camera_id", 0)
            self.input_mgr.select_camera(init_cid)

            # 2. AI & Engine: 멀티 프로파일 로드
            # FaceMesh는 공용 (설정값만 바뀜)
            self.tracker = FaceMesh(root_dir="assets")
            
            # BodyTracker & BeautyEngine은 프로파일별 리소스(모델/배경) 로드
            self.body_tracker = BodyTracker(profiles=self.profiles)
            self.beauty_engine = BeautyEngine(profiles=self.profiles)
            
            # 3. Output (Virtual Camera)
            self.virtual_cam = VirtualCamera(width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)

            # 4. 초기 상태 동기화 (Active Profile 설정)
            self.body_tracker.set_profile(self.current_profile_name)
            self.beauty_engine.set_profile(self.current_profile_name)
            
            # UI에 초기 슬라이더 값 전송
            self.slider_sync_requested.emit(self.params)
            
        except Exception as e:
            print(f"❌ [Worker] 초기화 중 치명적 오류: {e}")
            import traceback
            traceback.print_exc()
            return

        # Main Loop
        frame_count = 0
        prev_time = time.time()

        while self.running:
            # [Step 1] Input Capture
            if self.input_mgr:
                frame_gpu, ret = self.input_mgr.read()
            else:
                break
                
            if not ret:
                # 프레임이 안 들어오면 CPU 소모 방지를 위해 잠깐 대기
                self.msleep(5)
                continue
            
            # [Event] Background Reset
            if self.should_reset_bg:
                self.beauty_engine.reset_background(frame_gpu)
                self.should_reset_bg = False

            # [Step 2] AI Pre-processing
            # GPU -> CPU Copy for Inference (추후 전체 GPU 파이프라인화 가능)
            if cp and hasattr(frame_gpu, 'get'):
                frame_cpu_ai = frame_gpu.get()
            else:
                frame_cpu_ai = frame_gpu
            
            # Face Tracking (MediaPipe)
            faces = self.tracker.process(frame_cpu_ai) if self.tracker else []
            
            # Body Tracking (Student Model)
            body_landmarks = self.body_tracker.process(frame_cpu_ai) if self.body_tracker else None

            # [Step 3] Rendering (GPU)
            if self.beauty_engine:
                frame_out_gpu = self.beauty_engine.process(
                    frame_gpu, faces, body_landmarks, self.params, 
                    mask=self.body_tracker.get_mask()
                )
            else:
                frame_out_gpu = frame_gpu

            # [Step 4] Output & Display
            # OBS 전송
            if self.virtual_cam:
                self.virtual_cam.send(frame_out_gpu)

            # UI 미리보기 업데이트 (CPU 변환 필요)
            frame_out_cpu = frame_out_gpu.get() if hasattr(frame_out_gpu, 'get') else frame_out_gpu
            
            # 디버그: 뼈대 그리기 옵션이 켜져있으면 그리기
            if self.params.get('show_body_debug', False) and self.body_tracker:
                frame_out_cpu = self.body_tracker.draw_debug(frame_out_cpu, body_landmarks)
            
            self.frame_processed.emit(frame_out_cpu)

            # FPS Logging (1초 단위)
            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 1.0:
                # print(f"⚡ FPS: {frame_count} | Profile: {self.current_profile_name}")
                frame_count = 0
                prev_time = curr_time

        # 루프 종료 후 정리
        self.cleanup()

    def cleanup(self):
        """자원 해제 및 마지막 설정 저장"""
        print("🧹 [Worker] Cleaning up resources...")
        
        # 마지막으로 현재 설정 저장 (종료 시점의 슬라이더 값)
        self.save_current_config()
        
        self.tracker = None
        self.body_tracker = None
        self.beauty_engine = None
        
        # VRAM 정리
        if cp:
            try:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
            except: pass

        if self.input_mgr: self.input_mgr.release()
        if self.virtual_cam: self.virtual_cam.close()
        
        print("👋 [Worker] Shutdown complete.")

    def save_current_config(self):
        """현재 활성화된 프로파일의 설정을 JSON에 저장"""
        self.profile_mgr.update_params(self.current_profile_name, self.params)
        print(f"💾 [{self.current_profile_name}] 설정 저장됨.")

    @Slot(dict)
    def update_params(self, new_params):
        """UI 슬라이더가 움직일 때마다 호출되어 파라미터 갱신"""
        self.params = new_params.copy()

    @Slot()
    def reset_background(self):
        """배경 리셋 요청 플래그 설정"""
        self.should_reset_bg = True

    @Slot(int)
    def switch_profile(self, index):
        """
        [Key Logic] 프로파일 전환 (1, 2, 3 키 입력 시 호출)
        1. 현재 설정 저장
        2. 인덱스 변경 및 새 프로파일 로드
        3. 컴포넌트(카메라, 모델, 배경) 스위칭
        4. UI 슬라이더 동기화
        """
        if index < 0 or index >= len(self.profiles):
            return # Invalid index

        target_profile = self.profiles[index]
        if target_profile == self.current_profile_name:
            return # 이미 해당 프로파일임

        print(f"\n🔄 [Switch] {self.current_profile_name} -> {target_profile}")
        
        # 1. 현재 설정(슬라이더 값) 저장
        self.save_current_config()
        
        # 2. 상태 변경
        self.current_profile_name = target_profile
        self.active_profile_idx = index
        
        # 3. 새 설정 로드
        new_config = self.profile_mgr.get_config(target_profile)
        self.params = new_config.get("params", {}).copy()
        target_cam_id = new_config.get("camera_id", 0)
        
        # 4. 컴포넌트 스위칭 (Instant Switch)
        self.input_mgr.select_camera(target_cam_id)
        self.body_tracker.set_profile(target_profile)
        self.beauty_engine.set_profile(target_profile)
        
        # 5. UI 동기화 요청 (역방향 시그널) -> 슬라이더가 자동으로 움직임
        self.slider_sync_requested.emit(self.params)

    def stop(self):
        self.running = False

class MuseApp(MainWindow):
    """
    MainWindow를 상속받아 키보드 단축키 로직을 확장한 메인 앱 클래스
    """
    request_profile_switch = Signal(int)

    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        
        # Worker -> UI (슬라이더 동기화) 연결
        worker.slider_sync_requested.connect(self.beauty_panel.update_sliders_from_config)
        
        # UI -> Worker (프로파일 변경 요청) 연결
        self.request_profile_switch.connect(worker.switch_profile)

    def keyPressEvent(self, event):
        # 1~9 숫자키 감지 -> 프로파일 전환
        key = event.key()
        if Qt.Key_1 <= key <= Qt.Key_9:
            idx = key - Qt.Key_1 # 0-based index
            print(f"⌨️ [Key] Profile Switch Request: {idx + 1}")
            self.request_profile_switch.emit(idx)
            
            # 패널 제목 업데이트 (UI 반응성 향상)
            if idx < len(self.worker.profiles):
                self.beauty_panel.set_profile_info(self.worker.profiles[idx])
        
        elif key == Qt.Key_B:
            # B키는 MainWindow의 기본 동작(배경 리셋) 유지
            super().keyPressEvent(event)
        
        else:
            super().keyPressEvent(event)

def main():
    # Ctrl+C 종료 시그널 처리
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")

    # Worker 생성
    worker = BeautyWorker()
    
    # Window 생성 (확장된 MuseApp 사용)
    window = MuseApp(worker)
    
    # Worker와 Window 연결 (영상/파라미터 교환)
    window.connect_worker(worker)
    
    # 실행
    worker.start()
    window.show()
    
    # 앱 루프 실행
    app.exec()
    
    # 종료 처리
    worker.stop()
    worker.wait(2000)
    os._exit(0)

if __name__ == "__main__":
    main()