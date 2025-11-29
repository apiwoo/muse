# Project MUSE - src/main.py
# Created for AI Beauty Cam Project
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import cv2
import time
import numpy as np

# 현재 파일(main.py)의 상위 폴더(MUSE_Project)를 파이썬 검색 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.camera import Camera
from src.core.virtual_cam import VirtualCamera
from src.ai.tracker import FaceTracker  # [NEW] AI 트래커 추가

def main():
    logger = get_logger("Main")
    logger.info("🚀 Project MUSE (Phase 2: AI Tracking) 시작...")

    # 1. 모듈 초기화
    cam = Camera()
    vcam = VirtualCamera()
    tracker = FaceTracker() # [NEW] AI 엔진 로드

    # 2. 장치 시작
    if not cam.start():
        logger.error("프로그램을 종료합니다 (카메라 오류).")
        return

    if not vcam.start():
        logger.error("프로그램을 종료합니다 (가상 카메라 오류).")
        cam.stop()
        return

    logger.info("✅ 시스템 준비 완료! (Ctrl+C로 종료)")
    logger.info("👉 OBS에서 얼굴에 '그물망(Mesh)'이 씌워지는지 확인하세요.")

    # FPS 계산용 변수
    prev_time = 0

    # 3. 메인 루프
    try:
        while True:
            # (1) 입력: 웹캠
            frame = cam.read()
            if frame is None:
                continue

            # (2) 처리: AI 얼굴 추적 [NEW]
            results = tracker.process(frame)

            # (3) 시각화: 디버그용 그리기 (얼굴 위에 선 그리기) [NEW]
            # 나중에는 이 부분이 OpenGL 렌더링으로 대체됩니다.
            if results and results.multi_face_landmarks:
                tracker.draw_debug(frame, results)

            # FPS 표시
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # (4) 출력: 가상 카메라 전송
            vcam.send(frame)

            # (옵션) 로컬 미리보기
            cv2.imshow("MUSE Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("사용자 중단 요청.")
    
    finally:
        logger.info("시스템 종료 중...")
        cam.stop()
        vcam.stop()
        cv2.destroyAllWindows()
        logger.info("Bye!")

if __name__ == "__main__":
    main()