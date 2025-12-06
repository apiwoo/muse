# Project MUSE - viewport.py
# Created for Mode A (Visual Supremacy)
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2
import numpy as np

class Viewport(QLabel):
    """
    [UI Component] OpenCV 이미지를 받아 화면에 표시하는 뷰포트
    - 역할: BGR(OpenCV) -> RGB(Qt) 변환 및 렌더링
    - 최적화: 비율 유지하며 리사이징
    - [Fix] CuPy(GPU Array) 호환성 추가 및 안전성 강화
    """
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        
        # [Fix] Expanding -> Ignored
        # 이미지가 위젯 크기를 강제로 늘리는 피드백 루프 방지
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        self.setStyleSheet("background-color: #121212; border: 1px solid #333;")
        # [한글화] 대기 문구
        self.setText("카메라 신호를 기다리는 중...")
        self.setMinimumSize(640, 360)

    def update_image(self, cv_img):
        """
        OpenCV 이미지를 받아 QLabel에 표시
        :param cv_img: numpy array or cupy array (BGR format)
        """
        if cv_img is None:
            return

        # [Critical Fix] 데이터 타입 강제 변환 및 검증
        # 1. 이미 Numpy 배열인 경우 패스
        if isinstance(cv_img, np.ndarray):
            pass
        # 2. CuPy 배열인 경우 (.get() 메서드 보유)
        elif hasattr(cv_img, 'get'):
            try:
                cv_img = cv_img.get()
            except Exception:
                return # 변환 실패 시 무시
        # 3. PyTorch 텐서인 경우 (.cpu().numpy() 보유)
        elif hasattr(cv_img, 'cpu') and hasattr(cv_img, 'numpy'):
            try:
                cv_img = cv_img.cpu().numpy()
            except Exception:
                return
        # 4. 기타 타입이면 강제 변환 시도
        else:
            try:
                cv_img = np.array(cv_img)
            except Exception:
                return

        # 최종적으로 Numpy 배열이 아니면 중단
        if not isinstance(cv_img, np.ndarray):
            return

        # 1. BGR -> RGB 변환
        # (BeautyEngine은 BGR을 출력하므로 변환 필요)
        try:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        except Exception:
            # [Safety] cv2.error 및 기타 모든 예외를 무시하여 크래시 방지
            # 데이터 형식이 맞지 않는 프레임은 드롭합니다.
            return

        # 2. QImage 생성
        try:
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 3. 뷰포트 크기에 맞춰 스케일링 (비율 유지)
            # SmoothTransformation은 퀄리티가 좋지만 약간 느릴 수 있음. 성능 이슈 시 FastTransformation 사용.
            if self.width() > 0 and self.height() > 0:
                scaled_pixmap = QPixmap.fromImage(qt_img).scaled(
                    self.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                # 4. 화면 갱신
                self.setPixmap(scaled_pixmap)
        except Exception:
            # 리사이징이나 QImage 생성 중 오류 발생 시 무시
            pass