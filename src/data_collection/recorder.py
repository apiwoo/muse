# Project MUSE - src/data_collection/recorder.py
# Created for AI Beauty Cam Project
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import time
import os
import threading
import queue
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.config import Config

class DataRecorder:
    def __init__(self, output_dir="recorded_data"):
        self.logger = get_logger("DataRecorder")
        self.output_dir = output_dir
        self.is_recording = False
        self.frame_queue = queue.Queue()
        self.record_thread = None
        self.video_writer = None
        self.metadata = [] # 프레임별 메타데이터 (timestamp, anomaly_score 등)
        
        # 저장 경로 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start_recording(self, width, height, fps=60):
        """녹화를 시작합니다."""
        if self.is_recording:
            self.logger.warning("이미 녹화 중입니다.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"raw_data_{timestamp}.avi")
        
        # 코덱 설정 (MJPG - 고화질, 빠른 저장)
        # 무손실을 원하면 'FFV1' 등을 쓸 수 있으나 용량이 매우 큼
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        if not self.video_writer.isOpened():
            self.logger.error("비디오 파일을 생성할 수 없습니다.")
            return

        self.is_recording = True
        self.frame_queue = queue.Queue()
        self.metadata = []
        
        # 비동기 저장 스레드 시작
        self.record_thread = threading.Thread(target=self._write_loop)
        self.record_thread.start()
        
        self.logger.info(f"🎥 녹화 시작: {filename} ({width}x{height} @ {fps}fps)")

    def stop_recording(self):
        """녹화를 종료합니다."""
        if not self.is_recording:
            return

        self.logger.info("녹화 종료 요청...")
        self.is_recording = False
        
        if self.record_thread:
            self.record_thread.join()
            
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        self.logger.info("✅ 녹화 완료 및 파일 저장됨.")

    def add_frame(self, frame, landmarks=None):
        """
        메인 루프에서 호출. 프레임을 큐에 넣습니다.
        landmarks: 현재 프레임의 Face Mesh 결과 (있으면 저장)
        """
        if not self.is_recording:
            return
            
        # 메인 스레드 부하를 줄이기 위해 복사본을 큐에 넣음
        # (메모리가 넉넉하다는 가정 하에 High-End 전략)
        self.frame_queue.put((frame.copy(), time.time(), landmarks))

    def _write_loop(self):
        """별도 스레드에서 파일 쓰기 작업 수행 (IO 병목 해결)"""
        while self.is_recording or not self.frame_queue.empty():
            try:
                # 큐에서 프레임 꺼내기 (타임아웃 1초)
                frame_data = self.frame_queue.get(timeout=1.0)
                frame, timestamp, landmarks = frame_data
                
                # 비디오 파일에 쓰기
                if self.video_writer:
                    self.video_writer.write(frame)
                    
                # 메타데이터 저장 (추후 학습용)
                # 여기서는 간단히 타임스탬프만 저장하지만, 
                # 나중에는 랜드마크 좌표나 이상치 점수도 저장해야 함
                self.metadata.append({
                    "timestamp": timestamp,
                    "has_face": landmarks is not None
                })
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"녹화 중 오류 발생: {e}")