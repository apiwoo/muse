# Project MUSE - viewport.py
# Created for Mode A (Visual Supremacy)
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
import cv2

class Viewport(QLabel):
    """
    [UI Component] OpenCV 이미지를 받아 화면에 표시하는 뷰포트
    - 역할: BGR(OpenCV) -> RGB(Qt) 변환 및 렌더링
    - 최적화: 비율 유지하며 리사이징
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
        :param cv_img: numpy array (BGR format)
        """
        if cv_img is None:
            return

        # 1. BGR -> RGB 변환
        # (BeautyEngine은 BGR을 출력하므로 변환 필요)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # 2. QImage 생성
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 3. 뷰포트 크기에 맞춰 스케일링 (비율 유지)
        # SmoothTransformation은 퀄리티가 좋지만 약간 느릴 수 있음. 성능 이슈 시 FastTransformation 사용.
        scaled_pixmap = QPixmap.fromImage(qt_img).scaled(
            self.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )

        # 4. 화면 갱신
        self.setPixmap(scaled_pixmap)