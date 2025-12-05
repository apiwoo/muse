# Project MUSE - recorder.py
# Data Acquisition Tool for Multi-Camera Personalization (Profile Support)
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import os
import time
import sys
import shutil
import glob
import numpy as np

# [Optional] Windows Camera Name Detection
try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False
    print("\n[WARNING] 'pygrabber' module not installed.")
    print("   -> Cameras will be shown as numbers like 'Camera Device 0'.")
    print("   -> Install 'pip install pygrabber' to see real names.")

class DataRecorder:
    def __init__(self, output_dir="recorded_data"):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.root_dir, output_dir)
        
        self.root_data_path = os.path.join(self.output_dir, "personal_data")
        self.model_dir = os.path.join(self.root_dir, "assets", "models", "personal")

        self.cap = None
        self.clean_plate = None
        self.is_recording = False
        self.video_writer = None
        self.record_count = 0
        self.current_profile = "default" 
        self.profile_dir = ""            
        
        self.total_recorded_time = 0.0
        self.current_start_time = 0.0
        
        self._select_global_mode()
        self._setup_profile_session()

    def _select_global_mode(self):
        print("\n========================================================")
        print("   MUSE Multi-Cam Recorder - Global Mode")
        print("========================================================")
        print("   1. [RESET ALL] : Delete all profiles & reset")
        print("   2. [MANAGE]    : Add/Edit profiles (Keep data)")
        print("========================================================")
        
        while True:
            choice = input("-> Select (1 or 2): ").strip()
            if choice == '1':
                self._reset_all_data()
                break
            elif choice == '2':
                if not os.path.exists(self.root_data_path):
                    print("[WARNING] No existing data. Creating folder.")
                    os.makedirs(self.root_data_path, exist_ok=True)
                break
            else:
                print("[ERROR] Invalid input.")

    def _reset_all_data(self):
        print("\n[CLEAN] [RESET ALL] Backing up and resetting data...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_root = os.path.join(self.output_dir, "backup", f"backup_{timestamp}")
        
        if os.path.exists(self.root_data_path):
            try:
                os.makedirs(backup_root, exist_ok=True)
                shutil.move(self.root_data_path, os.path.join(backup_root, "personal_data"))
                print(f"   [OK] Data Backup: {backup_root}")
            except Exception as e:
                print(f"   [WARNING] Data Backup Failed: {e}")
        
        if os.path.exists(self.model_dir):
            try:
                model_backup = os.path.join(backup_root, "models")
                os.makedirs(model_backup, exist_ok=True)
                for ext in ["*.pth", "*.engine"]:
                    for f in glob.glob(os.path.join(self.model_dir, ext)):
                        shutil.move(f, model_backup)
                print(f"   [OK] Model Backup Complete")
            except Exception as e: pass

        os.makedirs(self.root_data_path, exist_ok=True)
        print("[INFO] Reset Complete.")

    def _get_camera_list(self):
        cameras = []
        if HAS_PYGRABBER:
            try:
                graph = FilterGraph()
                devices = graph.get_input_devices()
                for i, name in enumerate(devices):
                    cameras.append((i, name))
            except Exception as e:
                print(f"[WARNING] Camera Name Lookup Failed: {e}")
        
        if not cameras:
            if not HAS_PYGRABBER:
                print("\n[TIP] Install 'pygrabber' to see camera names.")
                
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cameras.append((i, f"Camera Device {i}"))
                    cap.release()
        
        return cameras

    def _setup_profile_session(self):
        print("\n========================================================")
        print("   Profile & Camera Setup")
        print("========================================================")
        
        existing_profiles = [d for d in os.listdir(self.root_data_path) if os.path.isdir(os.path.join(self.root_data_path, d))]
        if existing_profiles:
            print(f"[DIR] Existing Profiles: {', '.join(existing_profiles)}")
        
        while True:
            p_name = input("-> Enter Profile Name (e.g., front, top): ").strip()
            if p_name:
                self.current_profile = p_name
                self.profile_dir = os.path.join(self.root_data_path, p_name)
                os.makedirs(self.profile_dir, exist_ok=True)
                print(f"   -> Target Folder: {self.profile_dir}")
                break
        
        cameras = self._get_camera_list()
        print(f"\n[SCAN] Detected Cameras:")
        for idx, name in cameras:
            print(f"   [{idx}] {name}")
            
        while True:
            try:
                cam_id_str = input("-> Enter Camera ID: ").strip()
                cam_id = int(cam_id_str)
                
                valid_ids = [c[0] for c in cameras]
                if cam_id not in valid_ids:
                    print(f"[WARNING] Warning: ID {cam_id} not in detected list.")
                
                print(f"   [CAM] Connecting to Camera {cam_id}...")
                if self.cap: self.cap.release()
                self.cap = cv2.VideoCapture(cam_id)
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                if not self.cap.isOpened():
                    print("   [ERROR] Cannot open camera.")
                    continue
                
                ret, _ = self.cap.read()
                if ret:
                    print("   [OK] Connection Successful!")
                    break
                else:
                    print("   [WARNING] Camera opened but returned no frame.")
            except ValueError:
                print("   [ERROR] Enter a number.")

        self._load_existing_status()

    def _load_existing_status(self):
        bg_path = os.path.join(self.profile_dir, "background.jpg")
        if os.path.exists(bg_path):
            self.clean_plate = cv2.imread(bg_path)
            print("   [OK] Background loaded.")
        else:
            self.clean_plate = None
            print("   [WARNING] Need to capture background ('B').")

        files = glob.glob(os.path.join(self.profile_dir, "train_video_*.mp4"))
        max_idx = 0
        total_seconds = 0.0
        
        for f in files:
            try:
                name = os.path.splitext(os.path.basename(f))[0]
                idx = int(name.replace("train_video_", ""))
                if idx > max_idx: max_idx = idx
            except: pass
            
            cap = cv2.VideoCapture(f)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps > 0: total_seconds += (frames / fps)
            cap.release()
            
        self.record_count = max_idx
        self.total_recorded_time = total_seconds
        print(f"   [STAT] [{self.current_profile}] Existing Videos: {len(files)} ({self._fmt_time(total_seconds)})")

    def _fmt_time(self, seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def run(self):
        print("\n========================================================")
        print(f"   MUSE Recorder - [{self.current_profile}] Mode")
        print("========================================================")
        print("   [Key Controls]")
        print("   - 'B': Capture Background")
        print("   - 'R': Start/Stop Recording")
        print("   - 'N': Next Profile")
        print("   - 'Q': Quit")
        print("========================================================")

        while True:
            if not self.cap or not self.cap.isOpened():
                print("[ERROR] Camera Disconnected")
                break

            ret, frame = self.cap.read()
            if not ret: break

            display = frame.copy()
            h, w = display.shape[:2]

            current_clip_time = 0.0
            if self.is_recording:
                current_clip_time = time.time() - self.current_start_time
            total_display_time = self.total_recorded_time + current_clip_time

            cv2.putText(display, f"Profile: {self.current_profile}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if self.clean_plate is None:
                cv2.putText(display, "STEP 1: Press 'B' (Clean Plate)", (30, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif not self.is_recording:
                cv2.putText(display, "READY: Press 'R'", (30, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                small_bg = cv2.resize(self.clean_plate, (320, 180))
                display[0:180, w-320:w] = small_bg
                cv2.rectangle(display, (w-320, 0), (w, 180), (255, 255, 0), 2)
            else:
                cv2.putText(display, "[REC] RECORDING...", (30, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(display, (20, 90), 10, (0, 0, 255), -1)

            cv2.putText(display, f"Current: {self._fmt_time(current_clip_time)}", (w - 300, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(display, f"Total  : {self._fmt_time(total_display_time)}", (w - 300, 260), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("MUSE Recorder", display)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            
            elif key == ord('n'):
                if self.is_recording:
                    print("[WARNING] Cannot switch while recording.")
                else:
                    print("\n[LOOP] Switching profile...")
                    self._setup_profile_session() 
            
            elif key == ord('b'):
                self.clean_plate = frame.copy()
                path = os.path.join(self.profile_dir, "background.jpg")
                cv2.imwrite(path, self.clean_plate)
                print(f"[SNAP] [{self.current_profile}] Background Saved")
            
            elif key == ord('r'):
                if self.clean_plate is None:
                    print("[WARNING] Capture background first ('B')")
                    continue

                if not self.is_recording:
                    self.is_recording = True
                    self.current_start_time = time.time()
                    self.record_count += 1
                    
                    filename = f"train_video_{self.record_count:02d}.mp4"
                    video_path = os.path.join(self.profile_dir, filename)
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
                    print(f"[CAM] Recording Start: {filename}")
                else:
                    self.is_recording = False
                    elapsed = time.time() - self.current_start_time
                    self.total_recorded_time += elapsed
                    
                    if self.video_writer:
                        self.video_writer.release()
                        self.video_writer = None
                    print(f"[SAVE] Saved. Total: {self._fmt_time(self.total_recorded_time)}")

            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)

        self.cleanup()

    def cleanup(self):
        if self.cap: self.cap.release()
        if self.video_writer: self.video_writer.release()
        cv2.destroyAllWindows()
        print("[BYE] Recorder Exited.")

if __name__ == "__main__":
    recorder = DataRecorder()
    recorder.run()