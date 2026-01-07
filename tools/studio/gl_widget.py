# Project MUSE - gl_widget.py
# QLabel-based Camera Viewport (OpenGL-free fallback)
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QImage, QPixmap
import cv2


class CameraGLWidget(QLabel):
    """
    [Camera Viewport - QLabel Based]
    OpenGL 대신 QLabel + QPixmap을 사용하여 호환성 보장
    D3D11/OpenGL 호환 문제 없이 모든 Windows 환경에서 작동
    """
    gl_ready = Signal()  # 호환성을 위해 시그널 유지

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setScaledContents(False)  # 비율 유지를 위해 False

        # 상태 추적
        self._initialized = False
        self._frame_width = 0
        self._frame_height = 0

        print("[Camera] CameraGLWidget created (QLabel mode)")

    def showEvent(self, event):
        """위젯이 화면에 표시될 때 호출됨"""
        super().showEvent(event)
        if not self._initialized:
            self._initialized = True
            print("[Camera] Widget visible - emitting gl_ready signal")
            self.gl_ready.emit()

    @Slot(object)
    def render(self, frame):
        """
        프레임 렌더링
        frame: Numpy array (BGR format from OpenCV)
        """
        if frame is None:
            return

        try:
            # CuPy array인 경우 numpy로 변환
            if hasattr(frame, 'get'):
                frame = frame.get()

            h, w = frame.shape[:2]
            self._frame_width = w
            self._frame_height = h

            # BGR -> RGB 변환
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            # QImage 생성
            bytes_per_line = 3 * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 위젯 크기에 맞게 스케일링 (비율 유지)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"[Camera] Render error: {e}")

    def cleanup(self):
        """리소스 정리"""
        self.clear()
        print("[Camera] Cleanup complete")
