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
    def __init__(self, caps):
        super().__init__()
        self.caps = caps
        self.active_id = None
        self.latest_frame = None
        self.new_frame_available = False
        self.running = True
        self.lock = threading.Lock()
        self.daemon = True 

    def set_active_camera(self, cid):
        with self.lock:
            self.active_id = cid
            self.latest_frame = None

    def run(self):
        # [Optimization] Thread sleep removed for max polling rate
        while self.running:
            if self.active_id is None:
                time.sleep(0.01)
                continue
            
            # [Performance] Only read active camera fully
            # Inactive cameras: skip or grab() only if buffer lagging is issue
            # For USB bandwidth, better to NOT read inactive at all if possible,
            # but OpenCV buffers might get stale.
            # Compromise: Read Active FAST, others SLOW.
            
            for cid, cap in self.caps.items():
                if cid == self.active_id:
                    ret, frame = cap.read()
                    if ret:
                        with self.lock:
                            self.latest_frame = frame
                            self.new_frame_available = True
                else:
                    # [Bandwidth Saver] 
                    # Don't grab every loop for inactive cams. 
                    # If using DirectShow, buffers might pile up (latency on switch).
                    # If we care about bandwidth, we assume user accepts 1sec lag on switch.
                    # Here we just 'grab' (header only usually) to keep alive.
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
        self.caps = {}
        self.active_id = None
        self.width = width
        self.height = height
        self.fps = fps
        
        unique_sources = list(dict.fromkeys(camera_indices)) # Remove dups
        
        print(f"[CAM] [InputManager] Initializing sources: {unique_sources}")
        
        for source in unique_sources:
            cid = source
            if isinstance(source, int):
                # Webcam
                print(f"   -> Opening Webcam {source}...", end=" ")
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    # [Performance] Set props ONLY ONCE here.
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, fps)
                    # MJPG is critical for >1 cams on USB 2.0
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
                    
                    self.caps[cid] = cap
                    print("[OK]")
                else:
                    print("[ERROR]")
            
            elif isinstance(source, str):
                # File/Stream
                try:
                    cap = NVDECCapture(source, width, height, fps)
                    self.caps[cid] = cap
                except: pass

            if self.active_id is None and cid in self.caps:
                self.active_id = cid

        if not self.caps:
            # Fallback
            print("[WARNING] No cameras found. Using dummy.")
            self.active_id = 0

        self.worker = CaptureWorker(self.caps)
        self.worker.set_active_camera(self.active_id)
        self.worker.start()

    def select_camera(self, camera_id):
        # [Optimization] Only switch if different
        if camera_id == self.active_id: return True
        
        if camera_id in self.caps:
            self.active_id = camera_id
            self.worker.set_active_camera(camera_id)
            print(f"[INPUT] Switched to Source: {camera_id}")
            return True
        else:
            # Try to open dynamically if not present?
            # For stability, we only allow pre-configured cams for now.
            print(f"[WARNING] Camera {camera_id} not initialized.")
            return False

    def read(self):
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