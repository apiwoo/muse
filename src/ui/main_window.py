# Project MUSE - main_window.py
# Broadcast Page (Converted from MainWindow for QStackedWidget integration)
# V26.0: QML UI Integration (Hybrid Approach)
# (C) 2025 MUSE Corp. All rights reserved.

import os
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal

from ui.viewport import Viewport
from ui.qml_loader import QmlLoader
from bridge.beauty_bridge import BeautyBridge


class BroadcastPage(QWidget):
    """
    [Broadcast Page]
    방송 화면 (뷰포트 + 뷰티 패널)
    QStackedWidget 페이지로 사용 가능하도록 QWidget으로 변환됨
    """
    # 시그널 정의
    request_bg_reset = Signal()  # 배경 리셋 요청 (Worker가 수신)
    go_home = Signal()           # 메인 메뉴로 돌아가기 요청

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None  # BeautyWorker 참조

        # Discord Style 배경
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1f22;
                color: #dbdee1;
                font-family: 'Inter', 'Pretendard', 'Segoe UI', sans-serif;
            }
        """)

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 메인 컨텐츠 영역
        content_area = QWidget()
        content_area.setStyleSheet("background-color: #1e1f22;")
        content_layout = QHBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # 뷰포트 (중앙 위젯)
        self.viewport = Viewport()
        content_layout.addWidget(self.viewport, stretch=1)

        # 우측 패널
        panel_container = QWidget()
        panel_container.setFixedWidth(320)
        panel_container.setStyleSheet("background-color: #2b2d31;")
        panel_layout = QVBoxLayout(panel_container)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        # 패널 헤더 (홈 버튼 포함)
        panel_header = QWidget()
        panel_header.setFixedHeight(40)
        panel_header.setStyleSheet("background-color: #2b2d31;")
        header_layout = QHBoxLayout(panel_header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.setSpacing(8)

        lbl_header = QLabel("BEAUTY")
        lbl_header.setStyleSheet("""
            color: #949ba4;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        """)
        header_layout.addWidget(lbl_header)

        header_layout.addStretch()

        # 홈으로 버튼
        self.btn_home = QPushButton("HOME")
        self.btn_home.setCursor(Qt.PointingHandCursor)
        self.btn_home.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #b5bac1;
                border: none;
                border-radius: 3px;
                padding: 6px 10px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #36373d;
                color: #dbdee1;
            }
            QPushButton:pressed {
                background-color: #404249;
            }
        """)
        self.btn_home.clicked.connect(self._on_home_clicked)
        header_layout.addWidget(self.btn_home)

        panel_layout.addWidget(panel_header)

        # 구분선
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: rgba(255, 255, 255, 0.06);")
        panel_layout.addWidget(separator)

        # 뷰티 패널 (QML 버전)
        self.beauty_bridge = BeautyBridge()
        self.beauty_panel_qml = QmlLoader.load(
            "panels/BeautyPanelQml.qml",
            bridges={"beautyBridge": self.beauty_bridge}
        )
        panel_layout.addWidget(self.beauty_panel_qml)

        # 레거시 호환용 참조
        self.beauty_panel = self.beauty_bridge

        content_layout.addWidget(panel_container, stretch=0)
        main_layout.addWidget(content_area, stretch=1)

        # 상태 표시줄
        self.status_label = QLabel("준비됨. 배경이 바뀌었다면 'B' 키를 눌러 리셋하세요.")
        self.status_label.setStyleSheet("""
            padding: 8px 16px;
            color: rgba(255, 255, 255, 0.4);
            font-size: 11px;
            font-weight: 400;
            background-color: #1e1f22;
        """)
        main_layout.addWidget(self.status_label)

    def _on_home_clicked(self):
        """홈으로 버튼 클릭"""
        self.go_home.emit()

    def connect_worker(self, worker):
        """
        Worker Thread와 UI 연결
        """
        self.worker = worker

        # 배경 존재 여부 확인
        bg_path = os.path.join(
            worker.root_dir, "recorded_data", "personal_data",
            worker.current_profile_name, "background.jpg"
        )
        has_bg = os.path.exists(bg_path)
        self.beauty_bridge.set_background_status(has_bg)
        print(f"[INIT] Background pre-check: {has_bg} ({bg_path})")

        # 영상 수신: Worker가 프레임을 보내면 Viewport에 그림
        worker.frame_processed.connect(self.viewport.update_image)

        # 파라미터 송신: UI 슬라이더가 변하면 Worker에 전달
        self.beauty_bridge.paramChanged.connect(worker.update_params)

        # 배경 리셋 신호 연결
        self.request_bg_reset.connect(worker.reset_background)

        # 배경 상태 시그널 연결
        worker.bgStatusChanged.connect(self.beauty_bridge.set_background_status)

        # 배경 캡처 버튼 -> Worker 배경 리셋
        self.beauty_bridge.bgCaptureRequested.connect(worker.reset_background)

        print("[LINK] [BroadcastPage] UI와 Worker 스레드 연결 완료 (QML Mode)")

    def disconnect_worker(self):
        """Worker 연결 해제 (홈으로 돌아갈 때)"""
        if self.worker:
            try:
                self.worker.frame_processed.disconnect(self.viewport.update_image)
                self.beauty_bridge.paramChanged.disconnect(self.worker.update_params)
                self.request_bg_reset.disconnect(self.worker.reset_background)
                self.worker.bgStatusChanged.disconnect(self.beauty_bridge.set_background_status)
                self.beauty_bridge.bgCaptureRequested.disconnect(self.worker.reset_background)
            except RuntimeError:
                pass  # 이미 연결 해제된 경우
            self.worker = None

    def set_profile_info(self, profile_name):
        """프로필 정보 표시"""
        self.beauty_bridge.set_profile_info(profile_name)

    def keyPressEvent(self, event):
        """키보드 입력 감지"""
        if event.key() == Qt.Key_B:
            print("[KEY] 'B' Pressed -> Request Background Reset")
            self.request_bg_reset.emit()
            self.status_label.setText("배경 리셋 중...")
        else:
            super().keyPressEvent(event)


# 레거시 호환용 MainWindow (독립 실행 시 사용)
from ui.frameless_base import FramelessMixin
from bridge.titlebar_bridge import TitleBarBridge
from PySide6.QtWidgets import QMainWindow


class MainWindow(FramelessMixin, QMainWindow):
    """
    [Legacy Main Window]
    독립 실행 시 사용되는 QMainWindow 래퍼
    내부적으로 BroadcastPage를 사용
    """
    request_bg_reset = Signal()
    go_home = Signal()

    def __init__(self):
        super().__init__()

        # Frameless 윈도우 설정
        self.setup_frameless()

        self.setWindowTitle("프로젝트 MUSE: AI 방송 시스템")
        self.resize(1280, 720)
        self.setMinimumSize(960, 540)

        self.setStyleSheet("""
            background-color: #313338;
            color: #dbdee1;
            font-family: 'Inter', 'Pretendard', 'Segoe UI', sans-serif;
        """)

        self._init_ui()

    def _init_ui(self):
        central_container = QWidget()
        central_layout = QVBoxLayout(central_container)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # 커스텀 타이틀바 (QML 버전)
        self.titlebar_bridge = TitleBarBridge(self)
        self.titlebar_bridge.title = "PROJECT MUSE"
        self.titlebar = QmlLoader.load(
            "panels/TitleBarQml.qml",
            bridges={"titlebarBridge": self.titlebar_bridge}
        )
        self.titlebar.setFixedHeight(30)
        central_layout.addWidget(self.titlebar)

        # BroadcastPage를 메인 컨텐츠로 사용
        self.broadcast_page = BroadcastPage()
        self.broadcast_page.go_home.connect(self.go_home.emit)
        self.broadcast_page.request_bg_reset.connect(self.request_bg_reset.emit)
        central_layout.addWidget(self.broadcast_page, stretch=1)

        self.setCentralWidget(central_container)

        # 레거시 호환용 속성
        self.beauty_bridge = self.broadcast_page.beauty_bridge
        self.beauty_panel = self.broadcast_page.beauty_panel  # BeautyBridge 참조
        self.viewport = self.broadcast_page.viewport
        self.status_label = self.broadcast_page.status_label

    def connect_worker(self, worker):
        """Worker 연결 (레거시 호환)"""
        self.broadcast_page.connect_worker(worker)

    def set_profile_info(self, profile_name):
        """프로필 정보 표시 (레거시 호환)"""
        self.broadcast_page.set_profile_info(profile_name)

    def keyPressEvent(self, event):
        """키보드 입력 (레거시 호환)"""
        if event.key() == Qt.Key_B:
            self.request_bg_reset.emit()
            self.broadcast_page.status_label.setText("배경 리셋 중...")
        else:
            super().keyPressEvent(event)

    def changeEvent(self, event):
        """윈도우 상태 변경 감지 (최대화 등)"""
        super().changeEvent(event)
        if hasattr(self, 'titlebar_bridge'):
            self.titlebar_bridge.update_maximized_state()

    def closeEvent(self, event):
        """창 닫기 이벤트"""
        print("[EXIT] [MainWindow] 창 닫기 감지. 프로그램 종료 절차를 시작합니다.")
        event.accept()
