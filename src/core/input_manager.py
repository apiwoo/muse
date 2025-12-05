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
    print("[ERROR] CuPy not found. GPU acceleration unavailable.")
    HAS_CUDA = False
    raise RuntimeError("CuPy library not found. Please run 'pip install cupy-cuda12x'.")

class NVDECCapture:
    """
    [Plan A] FFmpeg Pipe-based NVDEC Capture
    - Executes ffmpeg.exe as subprocess for GPU accelerated decoding.
    - Receives Raw Video Bytes(BGR) via stdout.
    - Provides same interface as cv2.VideoCapture (read, release).
    """
    def __init__(self, source, width=1920, height=1080, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_len = width * height * 3 # BGR24 size
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.ffmpeg_path = os.path.join(project_root, "libs", "ffmpeg.exe")
        
        if not os.path.exists(self.ffmpeg_path):
            raise FileNotFoundError(f"[ERROR] FFmpeg not found at {self.ffmpeg_path}. Run 'tools/download_models.py' first.")

        print(f"[START] [NVDEC] Initializing FFmpeg Pipe for: {source}")
        
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
        Reads one frame bytes from Pipe.
        """
        if self.process.poll() is not None:
            return False, None

        raw_frame = self.process.stdout.read(self.frame_len)

        if len(raw_frame) != self.frame_len:
            print("[WARNING] [NVDEC] End of Stream or Incomplete Frame.")
            return False, None
        
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
    """
    [Plan D] Background Capture Thread
    - Fetches latest frame independently from main loop.
    - Minimizes Input Lag and prevents main thread bottlenecks.
    """
    def __init__(self, caps):
        super().__init__()
        self.caps = caps # {id: (type, cap_obj)}
        self.active_id = None
        self.latest_frame = None
        self.new_frame_available = False
        self.running = True
        self.lock = threading.Lock()
        self.daemon = True 

    def set_active_camera(self, cid):
        with self.lock:
            self.active_id = cid
            self.latest_frame = None # Reset

    def run(self):
        print("[THREAD] [Input] Capture Thread Started.")
        while self.running:
            if self.active_id is None or self.active_id not in self.caps:
                time.sleep(0.01)
                continue
            
            for cid, cap in self.caps.items():
                if cid == self.active_id:
                    # Active: Read full frame
                    ret, frame = cap.read()
                    if ret:
                        with self.lock:
                            self.latest_frame = frame
                            self.new_frame_available = True
                    else:
                        pass
                else:
                    # Inactive: Flush buffer for webcam
                    if isinstance(cap, cv2.VideoCapture):
                        cap.grab()
            
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
        [Fix v3.1] NVDEC & Webcam Hybrid Support
        - camera_indices: [0, 1] (Webcam) or ["video.mp4"] (File/NVDEC)
        """
        self.caps = {}
        self.active_id = None
        self.width = width
        self.height = height
        self.fps = fps
        
        unique_sources = []
        for src in camera_indices:
            if src not in unique_sources: unique_sources.append(src)
            
        print(f"[CAM] [InputManager] Initializing sources: {unique_sources}")
        
        for idx, source in enumerate(unique_sources):
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
                    print("[OK]")
                else:
                    print("[ERROR] Failed")
            
            elif isinstance(source, str):
                # [Case 2] Video File / Stream (NVDEC GPU)
                print(f"   -> Opening NVDEC Stream for {os.path.basename(source)}...", end=" ")
                try:
                    cap = NVDECCapture(source, width, height, fps)
                    self.caps[cid] = cap
                    print("[OK] (GPU Accelerated)")
                except Exception as e:
                    print(f"[ERROR] Failed ({e})")

            if self.active_id is None and cid in self.caps:
                self.active_id = cid

        if not self.caps:
            raise RuntimeError("[ERROR] No available input sources.")

        print(f"[INFO] [InputManager] Active source: {self.active_id}")

        # [Plan D] Start Capture Thread
        self.worker = CaptureWorker(self.caps)
        self.worker.set_active_camera(self.active_id)
        self.worker.start()

    def select_camera(self, camera_id):
        """Change Active Camera"""
        if camera_id in self.caps:
            if self.active_id != camera_id:
                self.active_id = camera_id
                print(f"[LOOP] [Input] Switched to Source: {camera_id}")
                self.worker.set_active_camera(camera_id)
            return True
        else:
            print(f"[WARNING] [Input] Source '{camera_id}' not available.")
            return False

    def read(self):
        """
        [Plan D] Non-blocking Read
        - Returns latest frame fetched by thread.
        - CPU(Numpy) -> GPU(CuPy) upload happens here.
        """
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
        self.caps.clear()