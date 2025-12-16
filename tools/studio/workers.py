# Project MUSE - workers.py
# Updated for Smart Resume & Stop Logic + LoRA Track Support

import sys
import os
import cv2
import subprocess
import time
from PySide6.QtCore import QThread, Signal

class CameraLoader(QThread):
    finished = Signal(object, int) 
    error = Signal(str)

    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index

    def run(self):
        try:
            cap = cv2.VideoCapture(self.camera_index)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if cap.isOpened():
                self.finished.emit(cap, self.camera_index)
            else:
                self.error.emit("카메라를 열 수 없습니다.")
        except Exception as e:
            self.error.emit(f"연결 중 오류 발생: {e}")

class PipelineWorker(QThread):
    """
    [Updated V7] Smart Stop & Resume Logic + LoRA Track
    - mode="train": Standard Student Training
    - mode="train_lora": High-Precision LoRA Training
    """
    log_signal = Signal(str)
    progress_signal = Signal(int, str, str) # percent, status, time
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, root_dir, mode="train"):
        super().__init__()
        self.root_dir = root_dir
        self.tools_dir = os.path.join(root_dir, "tools")
        self.mode = mode
        self.current_process = None
        
        self.target_profile = self._detect_active_profile()
        
        # Flag Path
        self.stop_flag_path = os.path.join(self.root_dir, "recorded_data", "personal_data", "stop_training.flag")
        
        # Time Tracking
        self.start_time = 0.0
        self.step_start_time = 0.0

    def _detect_active_profile(self):
        data_root = os.path.join(self.root_dir, "recorded_data", "personal_data")
        if not os.path.exists(data_root): return "default"
        
        profiles = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        if not profiles: return "default"
        
        latest_profile = max(profiles, key=os.path.getmtime)
        return os.path.basename(latest_profile)

    def request_early_stop(self):
        try:
            with open(self.stop_flag_path, "w") as f: f.write("STOP")
            self.log_signal.emit("\n[CMD] 중단 요청 신호 전송 완료. 현재 작업이 끝나면 멈춥니다.")
        except Exception as e:
            self.log_signal.emit(f"\n[ERROR] 중단 신호 생성 실패: {e}")

    def _check_stop(self):
        """Returns True if stop is requested."""
        if os.path.exists(self.stop_flag_path):
            self.log_signal.emit("\n[STOP] 파이프라인이 사용자 요청에 의해 중단되었습니다.")
            return True
        return False

    def _fmt_duration(self, seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}분 {s:02d}초"

    def _get_time_info(self):
        now = time.time()
        total_elapsed = now - self.start_time
        step_elapsed = now - self.step_start_time
        return f"총 소요: {self._fmt_duration(total_elapsed)} | 현재 단계: {self._fmt_duration(step_elapsed)}"

    def run(self):
        try:
            # Clear previous stop flag on start
            if os.path.exists(self.stop_flag_path):
                os.remove(self.stop_flag_path)

            self.start_time = time.time()
            self.log_signal.emit(f"[INFO] Target Profile Detected: {self.target_profile}")
            
            if self.mode == "analyze":
                self._run_analysis()
            elif self.mode == "train":
                self._run_training_pipeline()
            elif self.mode == "train_lora":
                self._run_lora_pipeline()
            else:
                raise ValueError("Invalid Mode")
            
            if not self._check_stop():
                self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(str(e))

    def _run_analysis(self):
        if self._check_stop(): return
        self.step_start_time = time.time()
        self.progress_signal.emit(0, "영상 분석 중 (첫 프레임 추출)...", self._get_time_info())
        
        self.run_script(
            os.path.join(self.tools_dir, "auto_labeling", "run_labeling.py"), 
            ["personal_data", "--mode", "preview"],
            weight_range=(0, 100)
        )
        
        if not self._check_stop():
            self.progress_signal.emit(100, "분석 완료", self._get_time_info())

    def _run_training_pipeline(self):
        # 1. Full Labeling
        if self._check_stop(): return
        self.step_start_time = time.time()
        self.progress_signal.emit(0, "Step 1/5: 정밀 라벨링 (이어하기 가능)...", self._get_time_info())
        self.run_script(
            os.path.join(self.tools_dir, "auto_labeling", "run_labeling.py"), 
            ["personal_data", "--mode", "full"],
            weight_range=(0, 20)
        )
        
        # 2. Data Filtering
        if self._check_stop(): return
        self.step_start_time = time.time()
        self.progress_signal.emit(20, f"Step 2/5: 데이터 정제 ({self.target_profile})...", self._get_time_info())
        self.run_script(
            os.path.join(self.tools_dir, "auto_labeling", "filter_bad_data.py"), 
            [self.target_profile],
            weight_range=(20, 22)
        )

        # 3. Train Segmentation (Skip if exists)
        if self._check_stop(): return
        self.step_start_time = time.time()
        self.progress_signal.emit(22, f"Step 3/5: Seg 학습 ({self.target_profile})...", self._get_time_info())
        self.run_script(
            os.path.join(self.tools_dir, "train_student.py"), 
            ["personal_data", "--task", "seg", "--profile", self.target_profile],
            weight_range=(22, 59)
        )

        # 4. Train Pose (Skip if exists)
        if self._check_stop(): return
        self.step_start_time = time.time()
        self.progress_signal.emit(59, f"Step 4/5: Pose 학습 ({self.target_profile})...", self._get_time_info())
        self.run_script(
            os.path.join(self.tools_dir, "train_student.py"), 
            ["personal_data", "--task", "pose", "--profile", self.target_profile],
            weight_range=(59, 95)
        )
        
        # 5. Conversion (Skip if exists)
        if self._check_stop(): return
        self.step_start_time = time.time()
        self.progress_signal.emit(95, f"Step 5/5: 변환 및 최적화 ({self.target_profile})...", self._get_time_info())
        self.run_script(
            os.path.join(self.tools_dir, "convert_student_to_trt.py"), 
            ["--profile", self.target_profile],
            weight_range=(95, 100)
        )
        
        if not self._check_stop():
            self.progress_signal.emit(100, "선택된 프로파일 학습 완료!", self._get_time_info())

    def _run_lora_pipeline(self):
        # 1. Full Labeling (Shared)
        if self._check_stop(): return
        self.step_start_time = time.time()
        self.progress_signal.emit(0, "Step 1/3: 정밀 라벨링 (공통 과정)...", self._get_time_info())
        self.run_script(
            os.path.join(self.tools_dir, "auto_labeling", "run_labeling.py"), 
            ["personal_data", "--mode", "full"],
            weight_range=(0, 30)
        )
        
        # 2. Data Filtering (Shared)
        if self._check_stop(): return
        self.step_start_time = time.time()
        self.progress_signal.emit(30, f"Step 2/3: 데이터 정제...", self._get_time_info())
        self.run_script(
            os.path.join(self.tools_dir, "auto_labeling", "filter_bad_data.py"), 
            [self.target_profile],
            weight_range=(30, 35)
        )

        # 3. LoRA Training + Merge + Convert (All in One)
        if self._check_stop(): return
        self.step_start_time = time.time()
        self.progress_signal.emit(35, f"Step 3/3: LoRA 학습 및 엔진 병합...", self._get_time_info())
        self.run_script(
            os.path.join(self.tools_dir, "train_pose_lora.py"), 
            ["personal_data", "--profile", self.target_profile],
            weight_range=(35, 100)
        )
        
        if not self._check_stop():
            self.progress_signal.emit(100, "LoRA 고정밀 모델 학습 완료!", self._get_time_info())

    def run_script(self, script_path, args, weight_range=(0, 100)):
        cmd = [sys.executable, script_path] + args
        script_name = os.path.basename(script_path)
        self.log_signal.emit(f"\n[START] Executing: {script_name} {' '.join(args)}")
        
        w_start, w_end = weight_range
        w_span = w_end - w_start
        
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        self.current_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            startupinfo=startupinfo
        )
        
        last_update_time = 0
        
        for line in self.current_process.stdout:
            line = line.strip()
            if not line: continue
            
            self.log_signal.emit(line)
            
            if "[PROGRESS]" in line:
                try:
                    parts = line.split("[PROGRESS]")
                    if len(parts) > 1:
                        val_str = parts[1].strip().split()[0]
                        local_p = int(float(val_str))
                        global_p = int(w_start + (local_p / 100.0) * w_span)
                        
                        now = time.time()
                        if now - last_update_time > 0.1 or local_p == 100:
                            self.progress_signal.emit(global_p, f"진행 중: {script_name} ({local_p}%)", self._get_time_info())
                            last_update_time = now
                except: pass
            
            now = time.time()
            if now - last_update_time > 1.0:
                self.progress_signal.emit(int(w_start), f"실행 중: {script_name}", self._get_time_info())
                last_update_time = now
        
        self.current_process.wait()
        
        # If stop flag exists, simply return (don't raise error, caller handles it)
        if self._check_stop():
            return

        if self.current_process.returncode != 0:
            raise RuntimeError(f"Script failed with code {self.current_process.returncode}")
        
        self.progress_signal.emit(w_end, f"완료: {script_name}", self._get_time_info())