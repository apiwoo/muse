# Project MUSE - input_manager.py
# (C) 2025 MUSE Corp. All rights reserved.
# Optimization: No-Lag Switching & NVDEC Support
# [2025-05 Update] Enhanced Camera Init with Warm-up & Detailed Logging
# [Critical Fix] Reverted to MSMF (Default) backend to fix 1 FPS issue
# [Critical Fix] Removed CaptureWorker for Thread Affinity (Direct Read Mode)
# [Fix] Added MJPG FourCC & Corrected Init Order (Bandwidth Fix)
# [Debug] Added Ultra-Verbose Logging for hang detection

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

# [Critical Change] CaptureWorker removed to fix DSHOW thread affinity issues.
# InputManager now handles reading directly in the calling thread.

class InputManager:
    def __init__(self, camera_indices=[0], width=1920, height=1080, fps=30):
        self.caps = {}
        self.active_id = None
        self.width = width
        self.height = height
        self.fps = fps
        
        unique_sources = list(dict.fromkeys(camera_indices))
        print(f"\n[CAM] [InputManager] === Debug Mode: Ultra Verbose Init ===")
        print(f"[CAM] [InputManager] Target Resolution: {width}x{height} @ {fps}fps")
        print(f"[CAM] [InputManager] Target Sources: {unique_sources}")
        
        for source in unique_sources:
            cid = source
            if isinstance(source, int):
                # Webcam Initialization Logic
                print(f"   -> [Init] Attempting to open Webcam ID {source}...")
                
                # Debug Log: Opening
                print(f"      [DEBUG] Calling cv2.VideoCapture({source})... ", end="", flush=True)
                t_start = time.time()
                cap = cv2.VideoCapture(source)
                print(f"Done in {time.time()-t_start:.4f}s")
                
                backend_name = cap.getBackendName()
                print(f"      [Backend] {backend_name}")

                if cap.isOpened():
                    # [CRITICAL FIX] Set MJPG Codec FIRST to reserve bandwidth before setting High-Res
                    # Debug Logs for Settings
                    
                    print(f"      [DEBUG] Setting MJPG Codec...", end="", flush=True)
                    t0 = time.time()
                    ret_cc = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
                    print(f" Result: {ret_cc} ({time.time()-t0:.4f}s)")
                    
                    print(f"      [DEBUG] Setting Width {width}...", end="", flush=True)
                    t0 = time.time()
                    ret_w = cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    print(f" Result: {ret_w} ({time.time()-t0:.4f}s)")

                    print(f"      [DEBUG] Setting Height {height}...", end="", flush=True)
                    t0 = time.time()
                    ret_h = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    print(f" Result: {ret_h} ({time.time()-t0:.4f}s)")
                    
                    print(f"      [DEBUG] Setting FPS {fps}...", end="", flush=True)
                    t0 = time.time()
                    ret_fps = cap.set(cv2.CAP_PROP_FPS, fps)
                    print(f" Result: {ret_fps} ({time.time()-t0:.4f}s)")
                    
                    # [Optimization] Low Latency Buffer (from pages.py)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # 3. Validate Settings
                    real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    real_fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    try:
                        fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                        fcc_str = "".join([chr((fcc >> 8 * i) & 0xFF) for i in range(4)])
                    except Exception as e:
                        fcc_str = f"Error({e})"
                    
                    print(f"      [Settings Check] {int(real_w)}x{int(real_h)} @ {real_fps:.1f}fps (Codec: {fcc_str})")

                    # 4. Warm-up Loop (Still needed for stability)
                    print("      [Warm-up] Starting frame stabilization loop...")
                    success = False
                    max_retries = 20 # Try for ~2 seconds
                    
                    for i in range(max_retries):
                        print(f"      [DEBUG] Attempt {i+1}: Calling read()... ", end="", flush=True)
                        t_read = time.time()
                        ret, frame = cap.read()
                        t_end = time.time()
                        print(f"Returned in {t_end - t_read:.4f}s | Ret: {ret}")
                        
                        if ret:
                            if frame is None:
                                print(f"      [DEBUG] !!! Ret is True but Frame is None !!!")
                            elif frame.size == 0:
                                print(f"      [DEBUG] !!! Frame size is 0 !!!")
                            else:
                                print(f"      [DEBUG] Frame Valid! Shape: {frame.shape}")
                                success = True
                                break
                        else:
                            print(f"      [DEBUG] Read Failed. Sleeping 0.1s...")
                            time.sleep(0.1)
                    
                    if success:
                        self.caps[cid] = cap
                        print(f"      -> [SUCCESS] Camera {source} Ready.")
                    else:
                        print(f"\n      -> [FAILED] Camera opened but returned NO frames after {max_retries} attempts.")
                        cap.release()
                else:
                    print(f"      -> [ERROR] Could not open device (Occupied or not found).")
            
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

        print(f"[CAM] [InputManager] === Initialization Complete. Active Cams: {len(self.caps)} ===\n")

        if not self.caps:
            print("[CRITICAL WARNING] No cameras found! InputManager running in EMPTY mode.")
        
        # [Thread Affinity Fix] Do NOT start a separate worker thread.
        # We will read directly in the read() method.

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
        # Reads directly from the OpenCV object in the calling thread (BeautyWorker).
        # This prevents DSHOW thread affinity issues.
        
        if self.active_id is None or self.active_id not in self.caps:
            return None, False

        cap = self.caps[self.active_id]
        ret, frame_cpu = cap.read()
        
        frame_gpu = None
        if ret and frame_cpu is not None:
             if HAS_CUDA:
                 frame_gpu = cp.asarray(frame_cpu)
             else:
                 frame_gpu = frame_cpu # Fallback for CPU-only
        
        return frame_gpu, ret

    def release(self):
        for cap in self.caps.values():
            cap.release()