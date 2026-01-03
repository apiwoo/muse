# Project MUSE - main.py
# Unified Client Application (Single Window, Multi-Page)
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import signal

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QFontDatabase, QFont
import qdarktheme

# Add Paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add tools directory for studio pages
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

from utils.cuda_helper import setup_cuda_environment
setup_cuda_environment()

from core.engine_loop import BeautyWorker
from ui.launcher import MainMenuPage
from ui.main_window import BroadcastPage
from ui.titlebar import TitleBar
from ui.frameless_base import FramelessMixin

# Studio Pages (New 5-Step Structure)
from studio.pages import (
    Step1_ProfileSelect, Step2_CameraConnect,
    Step3_DataRecording, Step4_AiAnalysis, Step5_ModelTraining
)
from studio.timeline_widget import StudioTimeline
from studio.settings_dialogs import (
    Step1SettingsDialog, Step2SettingsDialog, Step3SettingsDialog,
    Step4SettingsDialog, Step5SettingsDialog
)


class StudioContainer(QWidget):
    """
    스튜디오 컨테이너 - 타임라인 + 스텝 페이지 + 하단 네비게이션
    """
    go_home = Signal()
    go_broadcast = Signal()

    def __init__(self, personal_data_dir: str, root_dir: str, parent=None):
        super().__init__(parent)
        self.personal_data_dir = personal_data_dir
        self.root_dir = root_dir
        self.current_step = 0
        self.selected_profile = None
        self.selected_camera = 0

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 상단: 타임라인
        self.timeline = StudioTimeline()
        layout.addWidget(self.timeline)

        # 중단: 스텝 페이지들
        self.step_stack = QStackedWidget()
        self.step_stack.setStyleSheet("background-color: #1e1f22;")

        # Step 1: 프로필 선택
        self.step1 = Step1_ProfileSelect(self.personal_data_dir)
        self.step_stack.addWidget(self.step1)

        # Step 2: 카메라 연결
        self.step2 = Step2_CameraConnect()
        self.step_stack.addWidget(self.step2)

        # Step 3: 데이터 녹화
        self.step3 = Step3_DataRecording(os.path.join(self.root_dir, "recorded_data"))
        self.step_stack.addWidget(self.step3)

        # Step 4: AI 분석
        self.step4 = Step4_AiAnalysis(self.root_dir)
        self.step_stack.addWidget(self.step4)

        # Step 5: 모델 학습
        self.step5 = Step5_ModelTraining(self.root_dir)
        self.step_stack.addWidget(self.step5)

        layout.addWidget(self.step_stack, stretch=1)

        # 하단: 네비게이션 바
        nav_bar = QWidget()
        nav_bar.setObjectName("BottomNavBar")
        nav_bar.setStyleSheet("""
            #BottomNavBar {
                background-color: #232428;
                border-top: 1px solid rgba(255, 255, 255, 0.06);
            }
        """)
        nav_bar.setFixedHeight(70)

        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(30, 15, 30, 15)

        # 홈 버튼
        self.btn_home = QPushButton("메인 메뉴")
        self.btn_home.setObjectName("BtnHome")
        self.btn_home.setCursor(Qt.PointingHandCursor)
        self.btn_home.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #949ba4;
                border: none;
                font-size: 13px;
                font-weight: 500;
                padding: 8px 16px;
            }
            QPushButton:hover {
                color: #dbdee1;
            }
        """)
        self.btn_home.clicked.connect(self._on_home_clicked)
        nav_layout.addWidget(self.btn_home)

        nav_layout.addStretch()

        # 이전 버튼
        self.btn_prev = QPushButton("이전")
        self.btn_prev.setObjectName("BtnPrev")
        self.btn_prev.setCursor(Qt.PointingHandCursor)
        self.btn_prev.setStyleSheet("""
            QPushButton {
                background-color: #4e5058;
                color: #dbdee1;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #5c5f66;
            }
            QPushButton:disabled {
                background-color: #383a40;
                color: #6d6f78;
            }
        """)
        self.btn_prev.clicked.connect(self._go_prev)
        nav_layout.addWidget(self.btn_prev)

        # 다음 버튼
        self.btn_next = QPushButton("다음 단계로")
        self.btn_next.setObjectName("BtnNext")
        self.btn_next.setCursor(Qt.PointingHandCursor)
        self.btn_next.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00D4DB, stop:1 #7B61FF);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #33E0E6, stop:1 #9580FF);
            }
            QPushButton:disabled {
                background: #4e5058;
                color: #6d6f78;
            }
        """)
        self.btn_next.clicked.connect(self._go_next)
        nav_layout.addWidget(self.btn_next)

        layout.addWidget(nav_bar)

        # 초기 상태 설정
        self._update_nav_buttons()

    def _connect_signals(self):
        # 타임라인 클릭
        self.timeline.step_clicked.connect(self._on_timeline_step_clicked)
        self.timeline.settings_clicked.connect(self._on_settings_clicked)

        # 스텝 완료 시그널
        self.step1.step_completed.connect(lambda: self._on_step_completed(0))
        self.step1.profile_selected.connect(self._on_profile_selected)

        self.step2.step_completed.connect(lambda: self._on_step_completed(1))
        self.step2.camera_ready.connect(self._on_camera_ready)

        self.step3.step_completed.connect(lambda: self._on_step_completed(2))
        self.step4.step_completed.connect(lambda: self._on_step_completed(3))
        self.step5.step_completed.connect(lambda: self._on_step_completed(4))
        self.step5.training_finished.connect(self._on_training_finished)

    def _on_profile_selected(self, profile_name):
        self.selected_profile = profile_name

    def _on_camera_ready(self, cam_index):
        self.selected_camera = cam_index

    def _on_step_completed(self, step_index):
        self.timeline.mark_step_completed(step_index)
        self._update_nav_buttons()

    def _on_timeline_step_clicked(self, step_index):
        # 완료된 단계만 이동 가능
        if step_index < self.current_step:
            self._go_to_step(step_index)

    def _on_settings_clicked(self, step_index):
        # 각 단계별 설정 다이얼로그 표시
        if step_index == 0 and self.selected_profile:
            profile_path = os.path.join(self.personal_data_dir, self.selected_profile)
            dlg = Step1SettingsDialog(self.selected_profile, profile_path, self)
            dlg.exec()
        elif step_index == 1:
            dlg = Step2SettingsDialog([], self.selected_camera, parent=self)
            dlg.exec()
        elif step_index == 2:
            dlg = Step3SettingsDialog(parent=self)
            dlg.exec()
        elif step_index == 3:
            dlg = Step4SettingsDialog(parent=self)
            dlg.exec()
        elif step_index == 4:
            dlg = Step5SettingsDialog(parent=self)
            dlg.exec()

    def _go_to_step(self, step_index):
        # 현재 페이지 비활성화
        current_page = self.step_stack.currentWidget()
        if hasattr(current_page, 'deactivate'):
            current_page.deactivate()

        self.current_step = step_index
        self.step_stack.setCurrentIndex(step_index)
        self.timeline.set_current_step(step_index)

        # 새 페이지 활성화
        new_page = self.step_stack.currentWidget()
        if hasattr(new_page, 'activate'):
            new_page.activate()

        # Step 3 설정 (녹화 페이지)
        if step_index == 2 and self.selected_profile:
            profile_dir = os.path.join(self.personal_data_dir, self.selected_profile)
            self.step3.setup_session(self.selected_camera, self.selected_profile, profile_dir)

        self._update_nav_buttons()

    def _go_prev(self):
        if self.current_step > 0:
            self._go_to_step(self.current_step - 1)

    def _go_next(self):
        if self.current_step < 4:
            # 현재 스텝 완료 체크
            current_page = self.step_stack.currentWidget()
            if hasattr(current_page, 'is_completed') and current_page.is_completed():
                self._go_to_step(self.current_step + 1)

    def _on_home_clicked(self):
        # 모든 페이지 비활성화
        for i in range(self.step_stack.count()):
            page = self.step_stack.widget(i)
            if hasattr(page, 'deactivate'):
                page.deactivate()
        self.go_home.emit()

    def _on_training_finished(self):
        self.btn_next.setText("학습 완료! 방송 시작하기")
        self.btn_next.setEnabled(True)
        self.btn_next.clicked.disconnect()
        self.btn_next.clicked.connect(self._start_broadcast_after_training)

    def _start_broadcast_after_training(self):
        self.go_broadcast.emit()

    def _update_nav_buttons(self):
        # 이전 버튼
        self.btn_prev.setEnabled(self.current_step > 0)

        # 다음 버튼
        current_page = self.step_stack.currentWidget()
        is_completed = hasattr(current_page, 'is_completed') and current_page.is_completed()

        if self.current_step == 4:  # 마지막 단계
            self.btn_next.setText("학습 시작")
            self.btn_next.setEnabled(True)
        else:
            self.btn_next.setText("다음 단계로")
            self.btn_next.setEnabled(is_completed)

    def reset(self):
        """스튜디오 상태 초기화"""
        self.current_step = 0
        self.selected_profile = None
        self.selected_camera = 0
        self.timeline.reset_all()
        self.step_stack.setCurrentIndex(0)
        self._update_nav_buttons()

    def activate(self):
        """스튜디오 활성화 시 호출"""
        self.step1.activate()


class UnifiedMuseApp(FramelessMixin, QMainWindow):
    """
    통합 MUSE 앱
    QStackedWidget을 사용하여 모든 화면을 페이지로 관리

    페이지 인덱스 (V3 - 5 Step Studio):
    - 0: MainMenuPage - 메인 메뉴 (프로필 설정 + 액션 버튼)
    - 1: BroadcastPage - 방송 화면
    - 2: StudioContainer - 스튜디오 (5단계 통합)
    """
    request_profile_switch = Signal(int)

    # 페이지 인덱스 상수
    PAGE_MAIN_MENU = 0
    PAGE_BROADCAST = 1
    PAGE_STUDIO = 2

    def __init__(self):
        super().__init__()

        # Frameless 윈도우 설정
        self.setup_frameless()

        self.setWindowTitle("PROJECT MUSE - Unified Client")
        self.resize(1280, 800)
        self.setMinimumSize(960, 540)

        # 기본 스타일
        self.setStyleSheet("""
            background-color: #313338;
            color: #dbdee1;
            font-family: 'Inter', 'Pretendard', 'Segoe UI', sans-serif;
        """)

        # Worker 참조
        self.worker = None

        # 경로 설정
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.personal_data_dir = os.path.join(self.root_dir, "recorded_data", "personal_data")
        os.makedirs(self.personal_data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "recorded_data", "backup"), exist_ok=True)

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        # 전체 컨테이너
        central_container = QWidget()
        central_layout = QVBoxLayout(central_container)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # 커스텀 타이틀바
        self.titlebar = TitleBar(self, title="PROJECT MUSE")
        central_layout.addWidget(self.titlebar)

        # QStackedWidget - 모든 페이지 관리
        self.stack = QStackedWidget()
        central_layout.addWidget(self.stack, stretch=1)

        self.setCentralWidget(central_container)

        # === 페이지 생성 ===

        # 0: 메인 메뉴 (프로필 설정 포함)
        self.main_menu = MainMenuPage()
        self.stack.addWidget(self.main_menu)

        # 1: 방송 화면
        self.broadcast_page = BroadcastPage()
        self.stack.addWidget(self.broadcast_page)

        # 2: 스튜디오 컨테이너 (5단계 통합)
        self.studio = StudioContainer(self.personal_data_dir, self.root_dir)
        self.stack.addWidget(self.studio)

    def _connect_signals(self):
        # === 메인 메뉴 시그널 ===
        self.main_menu.start_broadcast.connect(self._start_broadcast)
        self.main_menu.open_studio.connect(self._show_studio)

        # === 방송 페이지 시그널 ===
        self.broadcast_page.go_home.connect(self._stop_and_go_menu)

        # === 스튜디오 시그널 ===
        self.studio.go_home.connect(self._show_menu)
        self.studio.go_broadcast.connect(self._start_broadcast_from_studio)

    # === 페이지 전환 메서드 ===

    def _show_menu(self):
        """메인 메뉴로 이동"""
        self.titlebar.set_title("PROJECT MUSE")
        self.main_menu.refresh_on_show()
        self.stack.setCurrentIndex(self.PAGE_MAIN_MENU)

    def _show_studio(self):
        """스튜디오로 이동"""
        self.titlebar.set_title("AI 모델 학습 스튜디오")
        self.studio.reset()
        self.studio.activate()
        self.stack.setCurrentIndex(self.PAGE_STUDIO)

    def _start_broadcast(self, profile, mode):
        """방송 시작"""
        print(f"[START] Launching Engine with Profile: {profile}, Mode: {mode}")

        # 기존 워커가 있으면 정리
        if self.worker:
            self._cleanup_worker()

        # 타이틀 업데이트
        self.titlebar.set_title(f"AI 엔진 로딩 중... ({profile})")
        QApplication.processEvents()

        # BeautyWorker 생성 및 시작
        self.worker = BeautyWorker(start_profile=profile, run_mode=mode)

        # 시그널 연결
        self.worker.slider_sync_requested.connect(
            self.broadcast_page.beauty_panel.update_sliders_from_config
        )
        self.request_profile_switch.connect(self.worker.switch_profile)

        # 방송 페이지에 워커 연결
        self.broadcast_page.connect_worker(self.worker)
        self.broadcast_page.set_profile_info(profile)

        # 워커 시작
        self.worker.start()

        # 방송 페이지로 전환
        self.titlebar.set_title(f"방송 중 - {profile.upper()}")
        self.stack.setCurrentIndex(self.PAGE_BROADCAST)

    def _start_broadcast_from_studio(self):
        """스튜디오에서 학습 완료 후 방송 시작"""
        profile = self.studio.selected_profile
        if profile:
            self._start_broadcast(profile, "PERSONAL")
        else:
            self._show_menu()

    def _stop_and_go_menu(self):
        """방송 중지 후 메인 메뉴로 이동"""
        print("[STOP] Stopping broadcast and returning to menu...")

        # 워커 정리
        self._cleanup_worker()

        # 메인 메뉴로 이동
        self._show_menu()

    def _cleanup_worker(self):
        """워커 정리"""
        if self.worker:
            # 연결 해제
            self.broadcast_page.disconnect_worker()

            # 워커 중지
            print("[STOP] Stopping worker thread...")
            self.worker.stop()
            self.worker.wait()
            print("[OK] Worker stopped.")

            self.worker = None

    # === 키보드 이벤트 ===

    def keyPressEvent(self, event):
        """키보드 입력 처리"""
        # 방송 페이지에서만 단축키 처리
        if self.stack.currentIndex() == self.PAGE_BROADCAST and self.worker:
            current_profiles = self.worker.profiles
            matched = False

            key_int = event.key()

            if key_int in [Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta]:
                super().keyPressEvent(event)
                return

            pressed_seq = QKeySequence(event.keyCombination())

            for idx, p_name in enumerate(current_profiles):
                config = self.worker.profile_mgr.get_config(p_name)
                hotkey_str = config.get("hotkey", "")

                if hotkey_str:
                    target_seq = QKeySequence(hotkey_str)
                    if pressed_seq.matches(target_seq) == QKeySequence.ExactMatch:
                        print(f"[KEY] Hotkey '{hotkey_str}' -> Switch to '{p_name}'")
                        self.request_profile_switch.emit(idx)
                        self.broadcast_page.set_profile_info(p_name)
                        matched = True
                        break

            if not matched:
                # B키: 배경 리셋 (BroadcastPage에서 처리)
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """창 닫기 이벤트"""
        print("[EXIT] Application closing...")

        # 워커 정리
        if self.worker:
            self._cleanup_worker()

        event.accept()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")

    # 폰트 로드
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fonts_dir = os.path.join(root_dir, "assets", "fonts")
    if os.path.exists(fonts_dir):
        import glob
        font_files = glob.glob(os.path.join(fonts_dir, "*.otf")) + glob.glob(os.path.join(fonts_dir, "*.ttf"))
        for font_path in font_files:
            font_id = QFontDatabase.addApplicationFont(font_path)
            if font_id >= 0:
                print(f"[FONT] Loaded: {os.path.basename(font_path)}")

    # 폰트 설정
    app_font = QFont("Inter", 10)
    app_font.setFamilies(["Inter", "Pretendard", "Segoe UI", "Malgun Gothic"])
    app_font.setWeight(QFont.Normal)
    app_font.setHintingPreference(QFont.PreferNoHinting)
    app.setFont(app_font)

    # 통합 앱 실행
    window = UnifiedMuseApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
