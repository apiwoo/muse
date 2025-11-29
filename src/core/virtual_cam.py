# Project MUSE - src/core/virtual_cam.py
# Created for AI Beauty Cam Project
# (C) 2025 MUSE Corp. All rights reserved.

import pyvirtualcam
import numpy as np
from src.utils.config import Config
from src.utils.logger import get_logger

class VirtualCamera:
    def __init__(self):
        self.logger = get_logger("VirtualCam")
        self.cam = None
        self.is_running = False

    def start(self):
        """가상 카메라를 시작합니다."""
        try:
            self.logger.info(f"가상 카메라 시작 시도 ({Config.WIDTH}x{Config.HEIGHT})...")
            
            # OBS 등이 인식할 가상 카메라 생성
            self.cam = pyvirtualcam.Camera(
                width=Config.WIDTH, 
                height=Config.HEIGHT, 
                fps=Config.FPS,
                fmt=pyvirtualcam.PixelFormat.BGR # OpenCV는 기본이 BGR이므로 BGR로 설정
            )
            
            self.is_running = True
            self.logger.info(f"가상 카메라 구동 중: {self.cam.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"가상 카메라 시작 실패: {e}")
            self.logger.error("OBS Virtual Camera가 설치되어 있는지, 다른 프로그램이 사용 중인지 확인하세요.")
            return False

    def send(self, frame):
        """
        처리된 프레임을 가상 카메라로 전송합니다.
        frame: OpenCV 이미지 (BGR 포맷)
        """
        if not self.is_running or self.cam is None:
            return

        try:
            # pyvirtualcam에 프레임 전송
            self.cam.send(frame)
            
            # 다음 프레임 대기 (FPS 조절)
            self.cam.sleep_until_next_frame()
            
        except Exception as e:
            self.logger.warning(f"프레임 전송 오류: {e}")

    def stop(self):
        """가상 카메라를 종료합니다."""
        if self.cam:
            self.cam.close()
        self.is_running = False
        self.logger.info("가상 카메라 종료됨.")