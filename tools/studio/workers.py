# Project MUSE - workers.py
# Updated for Dual-Task Training Workflow

import sys
import os
import cv2
import subprocess
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
    [Updated V4] Multi-Step Pipeline (Targeted)
    1. Filter Bad Data (Target Only)
    2. Train Seg (Target Only)
    3. Train Pose (Target Only)
    4. Convert (Target Only)
    """
    log_signal = Signal(str)
    progress_signal = Signal(int, str) 
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, root_dir, mode="train"):
        super().__init__()
        self.root_dir = root_dir
        self.tools_dir = os.path.join(root_dir, "tools")
        self.mode = mode
        self.current_process = None
        
        self.target_profile = self._detect_active_profile()

    def _detect_active_profile(self):
        data_root = os.path.join(self.root_dir, "recorded_data", "personal_data")
        if not os.path.exists(data_root): return "default"
        
        profiles = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        if not profiles: return "default"
        
        latest_profile = max(profiles, key=os.path.getmtime)
        return os.path.basename(latest_profile)

    def request_early_stop(self):
        flag_path = os.path.join(self.root_dir, "recorded_data", "personal_data", "stop_training.flag")
        try:
            with open(flag_path, "w") as f: f.write("STOP")
            self.log_signal.emit("\n[CMD] 중단 요청 신호 전송 완료.")
        except Exception as e:
            self.log_signal.emit(f"\n[ERROR] 중단 신호 생성 실패: {e}")

    def run(self):
        try:
            self.log_signal.emit(f"[INFO] Target Profile Detected: {self.target_profile}")
            
            if self.mode == "analyze":
                self._run_analysis()
            elif self.mode == "train":
                self._run_training_pipeline()
            else:
                raise ValueError("Invalid Mode")
            
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(str(e))

    def _run_analysis(self):
        self.progress_signal.emit(0, "영상 분석 중 (첫 프레임 추출)...")
        # run_labeling.py currently scans all, but preview is fast enough. 
        # Future optimization: Make labeling script accept profile arg too.
        self.run_script(
            os.path.join(self.tools_dir, "auto_labeling", "run_labeling.py"), 
            ["personal_data", "--mode", "preview"]
        )
        self.progress_signal.emit(100, "분석 완료")

    def _run_training_pipeline(self):
        # 1. Full Labeling
        self.progress_signal.emit(10, "Step 1/5: 정밀 라벨링 (Auto Labeling)...")
        self.run_script(
            os.path.join(self.tools_dir, "auto_labeling", "run_labeling.py"), 
            ["personal_data", "--mode", "full"]
        )
        
        # 2. Data Filtering (Target Only)
        self.progress_signal.emit(25, f"Step 2/5: 데이터 정제 ({self.target_profile})...")
        self.run_script(
            os.path.join(self.tools_dir, "auto_labeling", "filter_bad_data.py"), 
            [self.target_profile]
        )

        # 3. Train Segmentation (Target Only)
        self.progress_signal.emit(40, f"Step 3/5: Seg 학습 ({self.target_profile})...")
        self.run_script(
            os.path.join(self.tools_dir, "train_student.py"), 
            ["personal_data", "--task", "seg", "--profile", self.target_profile]
        )

        # 4. Train Pose (Target Only)
        self.progress_signal.emit(65, f"Step 4/5: Pose 학습 ({self.target_profile})...")
        self.run_script(
            os.path.join(self.tools_dir, "train_student.py"), 
            ["personal_data", "--task", "pose", "--profile", self.target_profile]
        )
        
        # 5. Conversion (Target Only)
        self.progress_signal.emit(85, f"Step 5/5: 변환 및 최적화 ({self.target_profile})...")
        self.run_script(
            os.path.join(self.tools_dir, "convert_student_to_trt.py"), 
            ["--profile", self.target_profile]
        )
        
        self.progress_signal.emit(100, "선택된 프로파일 학습 완료!")

    def run_script(self, script_path, args):
        cmd = [sys.executable, script_path] + args
        self.log_signal.emit(f"\n[START] Executing: {os.path.basename(script_path)} {' '.join(args)}")
        
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        self.current_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            startupinfo=startupinfo
        )
        
        for line in self.current_process.stdout:
            line = line.strip()
            if line:
                self.log_signal.emit(line)
        
        self.current_process.wait()
        if self.current_process.returncode != 0:
            raise RuntimeError(f"Script failed with code {self.current_process.returncode}")