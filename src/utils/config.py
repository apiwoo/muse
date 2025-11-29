# Project MUSE - src/utils/config.py
# Created for AI Beauty Cam Project
# (C) 2025 MUSE Corp. All rights reserved.

class Config:
    # --- 시스템 설정 ---
    APP_NAME = "MUSE Beauty Cam"
    VERSION = "4.0.0"
    
    # --- 카메라 설정 (C920 기준) ---
    CAMERA_INDEX = 1      # 웹캠 번호 (보통 0번, 안되면 1번)
    WIDTH = 1920          # 해상도 가로
    HEIGHT = 1080          # 해상도 세로
    FPS = 30              # 목표 FPS
    
    # --- 가상 카메라 설정 ---
    VIRTUAL_CAM_fmt = "BGR" # pyvirtualcam 입력 포맷