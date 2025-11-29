# Project MUSE - src/data_collection/validator.py
# Created for AI Beauty Cam Project
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import os
import glob
import json
import numpy as np
from src.utils.logger import get_logger

class DataValidator:
    def __init__(self, data_dir="recorded_data"):
        self.logger = get_logger("Validator")
        self.data_dir = data_dir
        self.video_files = []
        self.current_video_idx = 0
        self.cap = None
        self.total_frames = 0
        self.current_frame_pos = 0
        
        # 검수 상태
        self.bad_frames = set() # 삭제할 프레임 인덱스
        
    def load_videos(self):
        """녹화된 영상 목록을 로드합니다."""
        if not os.path.exists(self.data_dir):
            self.logger.warning(f"데이터 폴더가 없습니다: {self.data_dir}")
            return False
            
        # avi 파일 검색
        self.video_files = sorted(glob.glob(os.path.join(self.data_dir, "*.avi")))
        if not self.video_files:
            self.logger.warning("검수할 녹화 파일이 없습니다.")
            return False
            
        self.logger.info(f"검수 대상 파일: {len(self.video_files)}개")
        return True

    def start_review(self):
        """검수 UI 실행 (OpenCV HighGUI)"""
        if not self.load_videos():
            return

        file_path = self.video_files[0] # 가장 최신(또는 첫번째) 파일
        self.cap = cv2.VideoCapture(file_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.logger.info(f"검수 시작: {os.path.basename(file_path)} ({self.total_frames} frames)")

        window_name = "MUSE Data Validator (Space: Play/Pause, Left/Right: Seek, Del: Mark Bad, Esc: Quit)"
        cv2.namedWindow(window_name)
        
        paused = True
        
        while True:
            # 프레임 이동
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_pos)
            ret, frame = self.cap.read()
            if not ret:
                self.current_frame_pos = 0 # 루프
                continue

            # UI 오버레이
            display = frame.copy()
            status = "BAD (Will be deleted)" if self.current_frame_pos in self.bad_frames else "GOOD"
            color = (0, 0, 255) if self.current_frame_pos in self.bad_frames else (0, 255, 0)
            
            cv2.putText(display, f"Frame: {self.current_frame_pos}/{self.total_frames}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Status: {status}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 진행바
            bar_width = int((self.current_frame_pos / self.total_frames) * display.shape[1])
            cv2.rectangle(display, (0, display.shape[0]-10), (bar_width, display.shape[0]), (0, 255, 255), -1)

            cv2.imshow(window_name, display)

            # 키 입력 처리
            key = cv2.waitKey(0 if paused else 30) & 0xFF
            
            if key == 27: # ESC
                break
            elif key == 32: # Space (Play/Pause)
                paused = not paused
            elif key == 81 or key == 2: # Left Arrow (Previous)
                self.current_frame_pos = max(0, self.current_frame_pos - 1)
                paused = True
            elif key == 83 or key == 3: # Right Arrow (Next)
                self.current_frame_pos = min(self.total_frames - 1, self.current_frame_pos + 1)
                paused = True
            elif key == 255 or key == 127: # Delete Key (Mark Bad) - OS에 따라 키코드 다를 수 있음 (Del=46 or 127)
                self._toggle_bad_frame()
            elif key == ord('d'): # 'd' 키로도 삭제 마킹 가능하게
                self._toggle_bad_frame()

            if not paused:
                self.current_frame_pos = (self.current_frame_pos + 1) % self.total_frames

        self.cap.release()
        cv2.destroyAllWindows()
        
        # 종료 시 처리 결과 저장
        self._save_validation_result(file_path)

    def _toggle_bad_frame(self):
        if self.current_frame_pos in self.bad_frames:
            self.bad_frames.remove(self.current_frame_pos)
        else:
            self.bad_frames.add(self.current_frame_pos)

    def _save_validation_result(self, video_path):
        """검수 결과를 JSON으로 저장 (나중에 학습기에서 이 리스트를 보고 나쁜 프레임을 건너뜀)"""
        json_path = video_path.replace(".avi", "_validation.json")
        data = {
            "video_path": video_path,
            "total_frames": self.total_frames,
            "bad_frames": list(self.bad_frames)
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        self.logger.info(f"💾 검수 결과 저장 완료: {json_path} (삭제할 프레임: {len(self.bad_frames)}개)")