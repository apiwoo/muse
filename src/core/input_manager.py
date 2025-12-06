# Project MUSE - input_manager.py
# (C) 2025 MUSE Corp. All rights reserved.
# Optimization: No-Lag Switching & NVDEC Support

import cv2
import numpy as np
import time
import sys
import threading
import subprocess
import os

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

class NVDECCapture:
    def __init__(self, source, width=1920, height=1080, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_len = width * height * 3
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.ffmpeg_path = os.path.join(project_root, "libs", "ffmpeg.exe")
        
        if not os.path.exists(self.ffmpeg_path):
            raise FileNotFoundError(f"[ERROR] FFmpeg missing.")

        self.cmd = [
            self.ffmpeg_path,
            '-hide_banner', '-loglevel', 'error',
            '-hwaccel', 'cuda',
            '-i', str(source),
            '-vf', f'scale={width}:{height}',
            '-an', '-sn',
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            '-'
        ]
        
        self.process = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7
        )

    def isOpened(self):
        return self.process.poll() is None

    def read(self):
        if self.process.poll() is not None: return False, None
        raw_frame = self.process.stdout.read(self.frame_len)
        if len(raw_frame) != self.frame_len: return False, None
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame

    def release(self):
        if self.process:
            self.process.kill()
            self.process.wait()

    def grab(self):
        _ = self.process.stdout.read(self.frame_len)
        return True

class CaptureWorker(threading.Thread):
    def __init__(self, caps, width, height):
        super().__init__()
        self.caps = caps
        self.width = width
        self.height = height
        self.active_id = None
        self.latest_frame = None
        self.new_frame_available = False
        self.running = True
        self.lock = threading.Lock()
        self.daemon = True 
        print(f"[DEBUG] [CaptureWorker] Initialized. Cameras: {list(caps.keys())}")

    def set_active_camera(self, cid):
        with self.lock:
            self.active_id = cid
            self.latest_frame = None
            print(f"[DEBUG] [CaptureWorker] Active camera set to: {cid}")

    def run(self):
        print("[DEBUG] [CaptureWorker] Thread loop started.")
        fail_count = 0
        
        while self.running:
            # 카메라가 선택되지 않았거나 연결된 카메라가 없을 때 더미 처리
            if self.active_id is None:
                time.sleep(0.033)
                continue
            
            processed = False
            for cid, cap in self.caps.items():
                if cid == self.active_id:
                    ret, frame = cap.read()
                    processed = True
                    if ret and frame is not None:
                        with self.lock:
                            self.latest_frame = frame
                            self.new_frame_available = True
                        if fail_count > 0:
                            print(f"[INFO] [CaptureWorker] Camera {cid} recovered after {fail_count} failures.")
                            fail_count = 0
                    else:
                        fail_count += 1
                        # 60프레임(약 2초)마다 한 번씩만 로그 출력
                        if fail_count % 60 == 0:
                            print(f"[WARNING] [CaptureWorker] Failed to read from Cam {cid} (Ret={ret})")
                            
                else:
                    # 비활성 카메라는 버퍼 비우기용 grab만 수행
                    if isinstance(cap, cv2.VideoCapture):
                        cap.grab()
            
            if not processed:
                time.sleep(0.01)
            
    def get_latest_frame(self):
        with self.lock:
            if self.new_frame_available and self.latest_frame is not None:
                self.new_frame_available = False
                return self.latest_frame, True
            else:
                return None, False

    def stop(self):
        self.running = False
        print("[DEBUG] [CaptureWorker] Stopping thread.")

class InputManager:
    def __init__(self, camera_indices=[0], width=1920, height=1080, fps=30):
        self.caps = {}
        self.active_id = None
        self.width = width
        self.height = height
        self.fps = fps
        
        unique_sources = list(dict.fromkeys(camera_indices))
        
        print(f"[CAM] [InputManager] Initializing sources: {unique_sources}")
        
        for source in unique_sources:
            cid = source
            if isinstance(source, int):
                # Webcam
                print(f"   -> Opening Webcam {source}...", end=" ")
                # [Recorder.py Style] 최대한 단순하게 오픈
                cap = cv2.VideoCapture(source)
                
                # [DEBUG] 백엔드 확인 (DSHOW, MSMF 등)
                backend = cap.getBackendName()
                print(f"[Backend: {backend}]", end=" ")

                if cap.isOpened():
                    # [Recorder.py Style] 해상도/FPS만 설정하고 FourCC 강제 설정 제거
                    res_w = cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    res_h = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    res_fps = cap.set(cv2.CAP_PROP_FPS, fps)
                    
                    # [DEBUG] 설정 적용 결과 확인
                    print(f"\n      [Settings] W:{width}({res_w}) H:{height}({res_h}) FPS:{fps}({res_fps})")

                    # [Critical] MJPG 강제 설정 제거됨 (recorder.py와 동일 환경)
                    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G')) 

                    # [Check] 즉시 프레임 읽기 시도 (Warm-up 루프 제거)
                    ret, _ = cap.read()
                    if ret:
                        self.caps[cid] = cap
                        print(f"      -> [OK] Camera Ready. (Resolution: {int(cap.get(3))}x{int(cap.get(4))})")
                    else:
                        print(f"      -> [FAILED] Camera opened but returned no frame.")
                        cap.release()
                else:
                    print("[ERROR] (Could not open)")
            
            elif isinstance(source, str):
                # File/Stream
                try:
                    cap = NVDECCapture(source, width, height, fps)
                    self.caps[cid] = cap
                    print(f"   -> NVDEC Source {source} [OK]")
                except: 
                    print(f"   -> NVDEC Source {source} [FAIL]")

            if self.active_id is None and cid in self.caps:
                self.active_id = cid

        if not self.caps:
            print("[WARNING] No cameras found. InputManager will run empty.")
        
        print(f"[DEBUG] [InputManager] Starting CaptureWorker with Active ID: {self.active_id}")
        self.worker = CaptureWorker(self.caps, width, height)
        self.worker.set_active_camera(self.active_id)
        self.worker.start()

    def select_camera(self, camera_id):
        if camera_id == self.active_id: return True
        
        if camera_id in self.caps:
            self.active_id = camera_id
            self.worker.set_active_camera(camera_id)
            print(f"[INPUT] Switched to Source: {camera_id}")
            return True
        else:
            print(f"[WARNING] Camera {camera_id} not initialized.")
            return False

    def read(self):
        if not self.worker: return None, False
        frame_cpu, ret = self.worker.get_latest_frame()
        frame_gpu = None
        if ret and frame_cpu is not None:
             frame_gpu = cp.asarray(frame_cpu)
        return frame_gpu, ret

    def release(self):
        if self.worker:
            self.worker.stop()
            self.worker.join()
        for cap in self.caps.values():
            cap.release()