# Project MUSE - src/main.py
# Created for AI Beauty Cam Project
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import cv2
import time
import numpy as np

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.camera import Camera
from src.core.virtual_cam import VirtualCamera
from src.ai.tracker import FaceTracker
from src.graphics.renderer import Renderer

# [NEW] 데이터 수집 모듈 import
from src.data_collection.recorder import DataRecorder
from src.data_collection.guide_ui import GuideUI
from src.data_collection.validator import DataValidator

def main():
    logger = get_logger("Main")
    logger.info("🚀 Project MUSE (v7.1 High-End Integrated) 시작...")

    # 1. 모듈 초기화
    cam = Camera()
    vcam = VirtualCamera()
    tracker = FaceTracker()
    
    # 렌더러 초기화
    try:
        renderer = Renderer()
    except Exception as e:
        logger.error(f"렌더러 초기화 실패: {e}")
        return

    # [NEW] 레코더 및 가이드 초기화
    recorder = DataRecorder()
    guide = GuideUI()

    # 2. 장치 시작
    if not cam.start():
        return
    if not vcam.start():
        cam.stop()
        return

    logger.info("✅ 시스템 준비 완료!")
    logger.info("⌨️ [R]: 녹화 시작/중지 | [V]: 데이터 검수 모드 | [Q]: 종료")

    prev_time = 0

    try:
        while True:
            # (1) 입력
            frame = cam.read()
            if frame is None:
                continue

            # (2) 처리 (Face Mesh)
            results = tracker.process(frame)

            # [NEW] 녹화 중이면 프레임 저장
            if recorder.is_recording:
                # 랜드마크 데이터도 함께 저장 (나중에 학습용)
                # results 객체 전체를 넘기기보다 필요한 값만 추출해서 넘기는 것이 좋음 (여기선 간략화)
                recorder.add_frame(frame, results)

            # (3) 렌더링
            output_frame = renderer.render(frame, results)
            if output_frame is None:
                output_frame = frame
            else:
                output_frame = output_frame.copy()

            # [NEW] 가이드 UI (녹화 중일 때만 표시)
            if guide.is_active:
                guide.draw(output_frame)

            # FPS 표시
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            
            status_text = "REC" if recorder.is_recording else "LIVE"
            status_color = (0, 0, 255) if recorder.is_recording else (0, 255, 0)
            
            cv2.putText(output_frame, f"FPS: {int(fps)} | {status_text}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # (4) 출력
            vcam.send(output_frame)
            cv2.imshow("MUSE Preview", output_frame)
            
            # (5) 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('r'): # [R]ecord
                if recorder.is_recording:
                    recorder.stop_recording()
                    guide.is_active = False # 가이드 끄기
                else:
                    recorder.start_recording(Config.WIDTH, Config.HEIGHT)
                    guide.start() # 가이드 시작
                    
            elif key == ord('v'): # [V]alidate
                if recorder.is_recording:
                    logger.warning("녹화 중에는 검수 모드를 켤 수 없습니다.")
                else:
                    logger.info("🔍 검수 모드 진입 (Main Loop 일시 정지)...")
                    # 카메라 잠깐 멈추고 검수기 실행 (블로킹 방식)
                    # 실제 앱에서는 별도 프로세스나 윈도우로 띄우는 게 좋음
                    cv2.destroyWindow("MUSE Preview") # 충돌 방지
                    validator = DataValidator()
                    validator.start_review()
                    logger.info("검수 완료. 라이브 모드 복귀.")

    except KeyboardInterrupt:
        logger.info("사용자 중단 요청.")
    
    finally:
        if recorder.is_recording:
            recorder.stop_recording()
        
        logger.info("시스템 종료 중...")
        cam.stop()
        vcam.stop()
        cv2.destroyAllWindows()
        logger.info("Bye!")

if __name__ == "__main__":
    main()