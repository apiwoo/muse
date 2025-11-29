# Project MUSE - src/core/camera.py
# Created for AI Beauty Cam Project
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
from src.utils.config import Config
from src.utils.logger import get_logger

class Camera:
    def __init__(self):
        self.logger = get_logger("Camera")
        self.cap = None
        self.is_running = False

    def start(self):
        """웹캠 연결을 시작합니다."""
        self.logger.info(f"웹캠 연결 시도 (Index: {Config.CAMERA_INDEX})...")
        
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        
        # 해상도 및 FPS 강제 설정 (C920 최적화)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.FPS)
        
        # 설정된 실제 해상도 확인
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if not self.cap.isOpened():
            self.logger.error("웹캠을 열 수 없습니다!")
            return False
            
        self.is_running = True
        self.logger.info(f"웹캠 연결 성공: {w}x{h} @ {Config.FPS}fps")
        return True

    def read(self):
        """프레임 하나를 읽어옵니다."""
        if not self.is_running or self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning("프레임을 읽지 못했습니다.")
            return None
            
        return frame

    def stop(self):
        """웹캠 연결을 해제합니다."""
        if self.cap:
            self.cap.release()
        self.is_running = False
        self.logger.info("웹캠 연결 해제됨.")