# Project MUSE - workers.py
# Background threads for Studio UI (Task-Aware)

import sys
import os
import cv2
import subprocess
from PySide6.QtCore import QThread, Signal

class CameraLoader(QThread):
    """
    [Background Worker] Camera Loader
    """
    finished = Signal(object, int) # cap_obj, camera_index
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
    [Updated] 2-Step Pipeline Worker
    modes: "analyze" (Preview) / "train" (Full Pipeline)
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

    def run(self):
        try:
            if self.mode == "analyze":
                self._run_analysis()
            elif self.mode == "train":
                self._run_training()
            else:
                raise ValueError("Invalid Mode")
            
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(str(e))

    def _run_analysis(self):
        self.progress_signal.emit(0, "영상 분석 중 (첫 프레임 추출)...")
        # run_labeling.py --mode preview
        self.run_script(
            os.path.join(self.tools_dir, "auto_labeling", "run_labeling.py"), 
            ["personal_data", "--mode", "preview"]
        )
        self.progress_signal.emit(100, "분석 완료")

    def _run_training(self):
        # Step 1: Full Labeling
        self.progress_signal.emit(10, "Step 1/3: 정밀 라벨링 (Full Propagation)...")
        self.run_script(
            os.path.join(self.tools_dir, "auto_labeling", "run_labeling.py"), 
            ["personal_data", "--mode", "full"]
        )
        
        # Step 2: Training
        self.progress_signal.emit(40, "Step 2/3: AI 모델 학습 중 (Training)...")
        self.run_script(os.path.join(self.tools_dir, "train_student.py"), ["personal_data"])
        
        # Step 3: Conversion
        self.progress_signal.emit(80, "Step 3/3: 실시간 엔진 변환 중 (Optimization)...")
        self.run_script(os.path.join(self.tools_dir, "convert_student_to_trt.py"), [])
        
        self.progress_signal.emit(100, "완료! 모든 작업이 끝났습니다.")

    def run_script(self, script_path, args):
        cmd = [sys.executable, script_path] + args
        self.log_signal.emit(f"\n[START] Executing: {os.path.basename(script_path)} {' '.join(args)}")
        
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            startupinfo=startupinfo
        )
        
        for line in process.stdout:
            line = line.strip()
            if line:
                self.log_signal.emit(line)
        
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Script failed with code {process.returncode}")