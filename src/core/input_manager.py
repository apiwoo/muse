# Project MUSE - input_manager.py
# (C) 2025 MUSE Corp. All rights reserved.
# Target: Multi-Camera Support & NVDEC Acceleration (FFmpeg Pipe)

import cv2
import numpy as np
import time
import sys
import threading
import subprocess
import os

# High-Performance GPU Library
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    print("[Critical] CuPy not found. GPU acceleration unavailable.")
    HAS_CUDA = False
    raise RuntimeError("CuPy library not found. Please run 'pip install cupy-cuda12x'.")

class NVDECCapture:
    """
    [Plan A] FFmpeg Pipe-based NVDEC Capture
    - ffmpeg.exe를 서브프로세스로 실행하여 GPU 가속 디코딩을 수행합니다.
    - stdout으로 Raw Video Bytes(BGR)를 받아옵니다.
    - cv2.VideoCapture와 동일한 인터페이스(read, release)를 제공합니다.
    """
    def __init__(self, source, width=1920, height=1080, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_len = width * height * 3 # BGR24 size
        
        # 프로젝트 루트 기준 libs/ffmpeg.exe 경로 찾기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.ffmpeg_path = os.path.join(project_root, "libs", "ffmpeg.exe")
        
        if not os.path.exists(self.ffmpeg_path):
            raise FileNotFoundError(f"❌ FFmpeg not found at {self.ffmpeg_path}. Run 'tools/download_models.py' first.")

        print(f"🚀 [NVDEC] Initializing FFmpeg Pipe for: {source}")
        
        # FFmpeg Command Construction
        # -hwaccel cuda: 디코딩에 CUDA 사용
        # -vf scale=...: 출력 크기 강제 조절 (MUSE 엔진 요구사항에 맞춤)
        self.cmd = [
            self.ffmpeg_path,
            '-hide_banner', '-loglevel', 'error',
            '-hwaccel', 'cuda',           # GPU Decoding
            '-i', str(source),
            '-vf', f'scale={width}:{height}', # Resize to target resolution
            '-an', '-sn',                 # Disable Audio/Subtitles
            '-f', 'image2pipe',           # Output format
            '-pix_fmt', 'bgr24',          # Pixel format for OpenCV/CuPy compatibility
            '-vcodec', 'rawvideo',
            '-'                           # Output to stdout
        ]
        
        # 프로세스 실행 (bufsize를 넉넉하게 잡아 끊김 방지)
        self.process = subprocess.Popen(
            self.cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            bufsize=10**7
        )

    def isOpened(self):
        return self.process.poll() is None

    def read(self):
        """
        Pipe에서 한 프레임 분량의 바이트를 읽어옵니다.
        """
        if self.process.poll() is not None:
            return False, None

        # 정확히 프레임 크기만큼 읽기 (Blocking)
        raw_frame = self.process.stdout.read(self.frame_len)

        if len(raw_frame) != self.frame_len:
            print("⚠️ [NVDEC] End of Stream or Incomplete Frame.")
            return False, None
        
        # Byte -> Numpy Array (No Copy, just view logic ideally, but frombuffer creates new array)
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame

    def release(self):
        if self.process:
            self.process.kill()
            self.process.wait()

    def grab(self):
        # Pipe에서는 grab(skip)을 하려면 읽어서 버려야 함
        # 성능상 비효율적이지만 인터페이스 호환성을 위해 구현
        _ = self.process.stdout.read(self.frame_len)
        return True

class CaptureWorker(threading.Thread):
    """
    [Plan D] Background Capture Thread
    - 메인 루프와 별개로 항상 최신 프레임을 가져옵니다.
    - 입력 지연(Input Lag)을 최소화하고 메인 스레드 병목을 방지합니다.
    """
    def __init__(self, caps):
        super().__init__()
        self.caps = caps # {id: (type, cap_obj)}
        self.active_id = None
        self.latest_frame = None
        self.new_frame_available = False
        self.running = True
        self.lock = threading.Lock()
        self.daemon = True # 메인 프로그램 종료 시 자동 종료

    def set_active_camera(self, cid):
        with self.lock:
            self.active_id = cid
            self.latest_frame = None # 리셋

    def run(self):
        print("🧵 [Input] Capture Thread Started.")
        while self.running:
            # 활성 카메라가 없으면 대기
            if self.active_id is None or self.active_id not in self.caps:
                time.sleep(0.01)
                continue
            
            # 1. Grab all (Hardware Sync Strategy)
            # 활성 카메라는 read(), 나머지는 grab()(버퍼 비우기)
            # 단, 파일(NVDEC)은 grab이 의미 없거나 비용이 크므로 활성 상태일 때만 읽음
            
            for cid, cap in self.caps.items():
                if cid == self.active_id:
                    # Active: Read full frame
                    ret, frame = cap.read()
                    if ret:
                        with self.lock:
                            self.latest_frame = frame
                            self.new_frame_available = True
                    else:
                        # 파일 재생 끝났거나 에러 -> 루프? 일단 유지
                        pass
                else:
                    # Inactive: Webcam인 경우 버퍼 플러시 (Latency 제거용)
                    if isinstance(cap, cv2.VideoCapture):
                        cap.grab()
            
            # 과도한 CPU 점유 방지 (Sleep removed for max performance, or very small sleep)
            # time.sleep(0.001) 

    def get_latest_frame(self):
        with self.lock:
            if self.new_frame_available and self.latest_frame is not None:
                self.new_frame_available = False
                return self.latest_frame, True
            else:
                return None, False

    def stop(self):
        self.running = False

class InputManager:
    def __init__(self, camera_indices=[0], width=1920, height=1080, fps=30):
        """
        [수정 v3.1] NVDEC & Webcam Hybrid Support
        - camera_indices: [0, 1] (Webcam) or ["video.mp4"] (File/NVDEC)
        - 정수형 입력 -> cv2.VideoCapture (CPU)
        - 문자열 입력 -> NVDECCapture (GPU Accelerated via FFmpeg)
        """
        self.caps = {}
        self.active_id = None
        self.width = width
        self.height = height
        self.fps = fps
        
        # 중복 제거 (순서 유지)
        unique_sources = []
        for src in camera_indices:
            if src not in unique_sources: unique_sources.append(src)
            
        print(f"📷 [InputManager] 입력 소스 초기화: {unique_sources}")
        
        for idx, source in enumerate(unique_sources):
            # ID는 리스트 내의 인덱스가 아니라, 소스 자체(값)를 키로 사용하거나
            # 관리 편의를 위해 내부적으로 매핑된 ID를 쓸 수 있음.
            # 여기서는 편의상 입력된 '값' 자체를 식별자로 씁니다.
            cid = source
            
            if isinstance(source, int):
                # [Case 1] Webcam (Legacy CPU)
                print(f"   -> Connecting to Webcam {source}...", end=" ")
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, fps)
                    # Warm-up
                    for _ in range(5): cap.read()
                    self.caps[cid] = cap
                    print("✅ OK")
                else:
                    print("❌ Failed")
            
            elif isinstance(source, str):
                # [Case 2] Video File / Stream (NVDEC GPU)
                print(f"   -> Opening NVDEC Stream for {os.path.basename(source)}...", end=" ")
                try:
                    cap = NVDECCapture(source, width, height, fps)
                    self.caps[cid] = cap
                    print("✅ OK (GPU Accelerated)")
                except Exception as e:
                    print(f"❌ Failed ({e})")

            if self.active_id is None and cid in self.caps:
                self.active_id = cid

        if not self.caps:
            raise RuntimeError("❌ 연결 가능한 입력 소스가 없습니다.")

        print(f"✨ [InputManager] 활성 소스: {self.active_id}")

        # [Plan D] Start Capture Thread
        self.worker = CaptureWorker(self.caps)
        self.worker.set_active_camera(self.active_id)
        self.worker.start()

    def select_camera(self, camera_id):
        """활성 카메라 변경 (Instant Switch)"""
        # camera_id는 int(웹캠 인덱스)일 수도 있고 str(파일명)일 수도 있음
        if camera_id in self.caps:
            if self.active_id != camera_id:
                self.active_id = camera_id
                print(f"🔄 [Input] Switched to Source: {camera_id}")
                self.worker.set_active_camera(camera_id)
            return True
        else:
            # 설정 파일에는 camera_id가 0, 1로 저장되어 있는데
            # 실제 소스가 파일 경로인 경우 매칭 실패 가능성 있음.
            # 이 부분은 main.py나 config 로직에서 매핑을 잘 해줘야 함.
            print(f"⚠️ [Input] Source '{camera_id}' not available.")
            return False

    def read(self):
        """
        [Plan D] Non-blocking Read
        - 스레드가 가져온 최신 프레임을 반환합니다.
        - CPU(Numpy) -> GPU(CuPy) 업로드는 여기서 수행합니다.
        """
        frame_cpu, ret = self.worker.get_latest_frame()
        
        frame_gpu = None
        if ret and frame_cpu is not None:
             # BGR 유지 + GPU 업로드 (Host -> Device)
             # NVDECCapture를 썼더라도 pipe 출력은 RAM에 있으므로 업로드 필요
             frame_gpu = cp.asarray(frame_cpu)
        
        return frame_gpu, ret

    def release(self):
        if self.worker:
            self.worker.stop()
            self.worker.join()
            
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()