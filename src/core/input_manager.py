# Project MUSE - input_manager.py
# (C) 2025 MUSE Corp. All rights reserved.
# Optimization: No-Lag Switching & NVDEC Support
# [2025-05 Update] Enhanced Camera Init with Warm-up & Detailed Logging
# [Critical Fix] Reverted to 'recorder.py' style simple initialization.
# No forced backend, no forced codec. Let OS/Driver handle optimizations.

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
            self.ffmpeg_path = "ffmpeg"

        print(f"[NVDEC] Initializing FFmpeg pipe for: {source}")
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
        
        try:
            self.process = subprocess.Popen(
                self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7
            )
        except Exception as e:
            print(f"[NVDEC] FFmpeg Launch Failed: {e}")
            self.process = None

    def isOpened(self):
        if self.process is None: return False
        return self.process.poll() is None

    def read(self):
        if self.process is None or self.process.poll() is not None: return False, None
        try:
            raw_frame = self.process.stdout.read(self.frame_len)
            if len(raw_frame) != self.frame_len: return False, None
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3))
            return True, frame
        except Exception as e:
            print(f"[NVDEC] Read Error: {e}")
            return False, None

    def release(self):
        if self.process:
            self.process.kill()
            self.process.wait()

    def grab(self):
        if self.process:
            _ = self.process.stdout.read(self.frame_len)
        return True

class InputManager:
    def __init__(self, camera_indices=[0], width=1920, height=1080, fps=30):
        self.caps = {}
        self.active_id = None
        self.width = width
        self.height = height
        self.fps = fps
        
        unique_sources = list(dict.fromkeys(camera_indices))
        print(f"\n[CAM] [InputManager] === Initialization (Simple Mode) ===")
        print(f"[CAM] [InputManager] Target Resolution: {width}x{height} @ {fps}fps")
        
        for source in unique_sources:
            cid = source
            if isinstance(source, int):
                # Webcam Initialization Logic (Simplified like recorder.py)
                print(f"   -> [Init] Opening Webcam ID {source} (Auto Backend)...")
                
                # [FIX] Do NOT force CAP_DSHOW. Let OpenCV/OS decide (usually MSMF).
                cap = cv2.VideoCapture(source)
                
                if cap.isOpened():
                    # [FIX] Simple Setup Sequence (Same as recorder.py)
                    # 1. Width
                    # 2. Height
                    # 3. FPS
                    # No Codec forcing, No Auto-Exposure messing.
                    
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, fps)
                    
                    # Log actual settings
                    real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    real_fps = cap.get(cv2.CAP_PROP_FPS)
                    backend = cap.getBackendName()
                    
                    print(f"      [OK] Opened: {int(real_w)}x{int(real_h)} @ {real_fps:.1f}fps (Backend: {backend})")
                    
                    # Warm-up (Just a few frames to wake up sensor)
                    for _ in range(5):
                        cap.read()
                        
                    self.caps[cid] = cap
                else:
                    print(f"      [ERROR] Could not open device.")
            
            elif isinstance(source, str):
                # File/Stream (NVDEC)
                try:
                    cap = NVDECCapture(source, width, height, fps)
                    if cap.isOpened():
                        self.caps[cid] = cap
                        print(f"   -> [NVDEC] Source {source} [OK]")
                    else:
                        print(f"   -> [NVDEC] Source {source} [FAIL]")
                except Exception as e: 
                    print(f"   -> [NVDEC] Error: {e}")

            if self.active_id is None and cid in self.caps:
                self.active_id = cid

        print(f"[CAM] [InputManager] === Ready. Active Cams: {len(self.caps)} ===\n")

    def select_camera(self, camera_id):
        if camera_id == self.active_id: return True
        
        if camera_id in self.caps:
            self.active_id = camera_id
            print(f"[INPUT] Switched to Source: {camera_id}")
            return True
        else:
            print(f"[WARNING] Camera {camera_id} not available in initialized list.")
            return False

    def read(self):
        # [Direct Read Mode]
        if self.active_id is None or self.active_id not in self.caps:
            return None, False

        cap = self.caps[self.active_id]
        
        # Simple read
        ret, frame_cpu = cap.read()
        
        frame_gpu = None
        if ret and frame_cpu is not None:
             if HAS_CUDA:
                 frame_gpu = cp.asarray(frame_cpu)
             else:
                 frame_gpu = frame_cpu 
        
        return frame_gpu, ret

    def release(self):
        for cap in self.caps.values():
            cap.release()