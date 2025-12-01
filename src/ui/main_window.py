# Project MUSE - main_window.py
# Created for Mode A (Visual Supremacy)
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QDockWidget, QLabel
from PySide6.QtCore import Qt, Signal

from ui.viewport import Viewport
from ui.beauty_panel import BeautyPanel

class MainWindow(QMainWindow):
    """
    [Main Application Window]
    - 중앙: Viewport (카메라 프리뷰)
    - 우측: BeautyPanel (조절 패널)
    - 역할: UI 레이아웃 구성 및 Worker Thread와의 연결 고리
    """
    # [New] 배경 리셋 요청 시그널 (Worker가 수신)
    request_bg_reset = Signal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Project MUSE: Visual Supremacy (v2.1 GUI)")
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

        # 상태 표시줄
        self.status_label = QLabel("Ready. Press 'B' to reset background.")
        self.status_label.setStyleSheet("padding: 5px; color: #888;")
        self.statusBar().addWidget(self.status_label)

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
        
        # 3. [New] 배경 리셋 신호 연결
        self.request_bg_reset.connect(worker.reset_background)
        
        print("🔗 [MainWindow] UI와 Worker 스레드 연결 완료")

    def keyPressEvent(self, event):
        """
        [New] 키보드 입력 감지
        - B 키: 배경 리셋
        """
        if event.key() == Qt.Key_B:
            print("⌨️ [Key] 'B' Pressed -> Request Background Reset")
            self.request_bg_reset.emit()
            self.status_label.setText("Background Reset Triggered!")
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """
        [Critical] 창 닫기(X버튼) 클릭 시 호출.
        이 함수가 없으면 백그라운드 스레드가 돌고 있을 때 앱이 완전히 꺼지지 않을 수 있습니다.
        """
        print("❌ [MainWindow] 창 닫기 감지. 프로그램 종료 절차를 시작합니다.")
        event.accept() # 이벤트를 수락하여 Qt에게 창을 닫으라고 알림