# Project MUSE - main_window.py
# Created for Mode A (Visual Supremacy)
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QDockWidget
from PySide6.QtCore import Qt

from ui.viewport import Viewport
from ui.beauty_panel import BeautyPanel

class MainWindow(QMainWindow):
    """
    [Main Application Window]
    - 중앙: Viewport (카메라 프리뷰)
    - 우측: BeautyPanel (조절 패널)
    - 역할: UI 레이아웃 구성 및 Worker Thread와의 연결 고리
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Project MUSE: Visual Supremacy (v2.0 GUI)")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: #121212; color: #F0F0F0;")

        self._init_ui()

    def _init_ui(self):
        # 1. 중앙 위젯 (뷰포트)
        self.viewport = Viewport()
        self.setCentralWidget(self.viewport)

        # 2. 우측 도킹 패널 (뷰티 컨트롤)
        self.dock_panel = QDockWidget("Controls", self)
        self.dock_panel.setAllowedAreas(Qt.RightDockWidgetArea)
        self.dock_panel.setFeatures(QDockWidget.NoDockWidgetFeatures) # 이동 불가, 닫기 불가
        
        self.beauty_panel = BeautyPanel()
        self.dock_panel.setWidget(self.beauty_panel)
        
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_panel)

    def connect_worker(self, worker):
        """
        [Critical] Worker Thread와 UI 연결
        Worker(로직) -> Signal -> UI(메인쓰레드)
        UI(조작) -> Signal -> Worker(로직)
        """
        # 1. 영상 수신: Worker가 프레임을 보내면 Viewport에 그림
        worker.frame_processed.connect(self.viewport.update_image)

        # 2. 파라미터 송신: UI 슬라이더가 변하면 Worker에 전달
        self.beauty_panel.paramChanged.connect(worker.update_params)
        
        print("🔗 [MainWindow] UI와 Worker 스레드 연결 완료")