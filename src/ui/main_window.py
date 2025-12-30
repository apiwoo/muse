# Project MUSE - main_window.py
# Created for Mode A (Visual Supremacy)
# (C) 2025 MUSE Corp. All rights reserved.

import os

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

        # [한글화] 윈도우 타이틀
        self.setWindowTitle("프로젝트 MUSE: AI 방송 시스템 (v2.1 GUI)")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: #121212; color: #F0F0F0;")

        self._init_ui()

    def _init_ui(self):
        # 1. 중앙 위젯 (뷰포트)
        self.viewport = Viewport()
        self.setCentralWidget(self.viewport)

        # 2. 우측 도킹 패널 (뷰티 컨트롤)
        # [한글화] 패널 타이틀
        self.dock_panel = QDockWidget("제어판 (Controls)", self)
        self.dock_panel.setAllowedAreas(Qt.RightDockWidgetArea)
        self.dock_panel.setFeatures(QDockWidget.NoDockWidgetFeatures) # 이동 불가, 닫기 불가
        
        self.beauty_panel = BeautyPanel()
        self.dock_panel.setWidget(self.beauty_panel)
        
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_panel)

        # 상태 표시줄
        # [한글화] 안내 문구
        self.status_label = QLabel("준비됨. 배경이 바뀌었다면 'B' 키를 눌러 리셋하세요.")
        self.status_label.setStyleSheet("padding: 5px; color: #888;")
        self.statusBar().addWidget(self.status_label)

    def connect_worker(self, worker):
        """
        [Critical] Worker Thread와 UI 연결
        Worker(로직) -> Signal -> UI(메인쓰레드)
        UI(조작) -> Signal -> Worker(로직)
        """
        # [V5.0] 엔진 시작 전에 배경 존재 여부 먼저 확인 (깜빡임 방지)
        bg_path = os.path.join(
            worker.root_dir, "recorded_data", "personal_data",
            worker.current_profile_name, "background.jpg"
        )
        has_bg = os.path.exists(bg_path)
        self.beauty_panel.set_background_status(has_bg)
        print(f"[INIT] Background pre-check: {has_bg} ({bg_path})")

        # 1. 영상 수신: Worker가 프레임을 보내면 Viewport에 그림
        worker.frame_processed.connect(self.viewport.update_image)

        # 2. 파라미터 송신: UI 슬라이더가 변하면 Worker에 전달
        self.beauty_panel.paramChanged.connect(worker.update_params)

        # 3. [New] 배경 리셋 신호 연결
        self.request_bg_reset.connect(worker.reset_background)

        # 4. [V5.0] 배경 상태 시그널 연결
        worker.bgStatusChanged.connect(self.beauty_panel.set_background_status)

        # 5. [V5.0] 배경 캡처 버튼 -> Worker 배경 리셋
        self.beauty_panel.bgCaptureRequested.connect(worker.reset_background)

        print("[LINK] [MainWindow] UI와 Worker 스레드 연결 완료")

    def keyPressEvent(self, event):
        """
        [New] 키보드 입력 감지
        - B 키: 배경 리셋
        """
        if event.key() == Qt.Key_B:
            print("[KEY] 'B' Pressed -> Request Background Reset")
            self.request_bg_reset.emit()
            # [한글화] 상태 업데이트
            self.status_label.setText("배경 리셋 중...")
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """
        [Critical] 창 닫기(X버튼) 클릭 시 호출.
        이 함수가 없으면 백그라운드 스레드가 돌고 있을 때 앱이 완전히 꺼지지 않을 수 있습니다.
        """
        print("[EXIT] [MainWindow] 창 닫기 감지. 프로그램 종료 절차를 시작합니다.")
        event.accept() # 이벤트를 수락하여 Qt에게 창을 닫으라고 알림