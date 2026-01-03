# Project MUSE - launcher.py
# Main Menu Page (Unified Entry Point)
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import cv2
import glob
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QComboBox, QLineEdit, QMessageBox,
    QGroupBox, QFrame, QKeySequenceEdit, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap, QKeySequence

try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False

from utils.config import ProfileManager


class MainMenuPage(QWidget):
    """
    [Main Menu Page]
    통합 앱의 메인 화면
    - 프로필 선택/생성/삭제
    - 카메라 ID 지정
    - 구동 모드 선택
    - 방송 시작 및 스튜디오 열기 버튼 (하단 강조)

    3단 레이아웃:
    - 상단: 타이틀
    - 중단: 설정 (프로필 목록 + 프로필 설정)
    - 하단: 액션 버튼 (방송 시작, 스튜디오 열기)
    """
    # 시그널 정의
    start_broadcast = Signal(str, str)  # 방송 시작 (profile_name, run_mode)
    open_studio = Signal()              # 학습 스튜디오 열기

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet("""
            QWidget {
                background-color: #1e1f22;
                color: #dbdee1;
                font-family: 'Inter', 'Pretendard', 'Segoe UI', sans-serif;
            }
            QGroupBox {
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 8px;
                margin-top: 20px;
                font-weight: 600;
                color: #5865f2;
                background: #2b2d31;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                font-size: 11px;
                letter-spacing: 0.02em;
            }
            QListWidget {
                background-color: #1e1f22;
                border: 1px solid rgba(255, 255, 255, 0.06);
                color: #dbdee1;
                border-radius: 8px;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 10px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #5865f2;
                color: white;
            }
            QListWidget::item:hover:!selected {
                background-color: rgba(255, 255, 255, 0.06);
            }
            QLabel {
                color: #949ba4;
            }
            QLineEdit, QComboBox, QKeySequenceEdit {
                background-color: #383a40;
                border: none;
                padding: 10px;
                color: #dbdee1;
                border-radius: 4px;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus {
                outline: 2px solid #5865f2;
            }
            QPushButton {
                background-color: #4e5058;
                border: 1px solid #5c5f66;
                padding: 10px 16px;
                color: #ffffff;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #5c5f66;
                border-color: #6d6f78;
            }
            QPushButton:pressed {
                background-color: #3f4248;
            }
            QPushButton#Danger {
                background-color: #da373c;
                border: 1px solid #e5484d;
            }
            QPushButton#Danger:hover {
                background-color: #c62f33;
                border-color: #da373c;
            }
            QRadioButton {
                color: #949ba4;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid #4e5058;
                background: transparent;
            }
            QRadioButton::indicator:checked {
                background-color: #5865f2;
                border-color: #5865f2;
            }
            QRadioButton:disabled {
                color: #6d6f78;
            }
            /* 하단 강조 버튼 스타일 */
            QPushButton#BigPrimary {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00D4DB, stop:1 #7B61FF);
                font-size: 18px;
                font-weight: 700;
                padding: 20px;
                border-radius: 12px;
                border: none;
                color: white;
            }
            QPushButton#BigPrimary:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00E5EC, stop:1 #8B71FF);
            }
            QPushButton#BigPrimary:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00B5BB, stop:1 #6A50E0);
            }
            QPushButton#BigAccent {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF6D00, stop:1 #FF9100);
                font-size: 18px;
                font-weight: 700;
                padding: 20px;
                border-radius: 12px;
                border: none;
                color: white;
            }
            QPushButton#BigAccent:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF8F00, stop:1 #FFAB00);
            }
            QPushButton#BigAccent:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E05F00, stop:1 #E08000);
            }
        """)

        self.pm = ProfileManager()
        self.selected_profile = None
        self.selected_mode = "STANDARD"
        self.available_cameras = self._scan_cameras()

        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.root_dir, "assets", "models", "personal")

        self._init_ui()
        self._refresh_list()

    def _create_separator(self):
        """섹션 구분선 생성"""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: rgba(255, 255, 255, 0.06); margin: 8px 0;")
        return line

    def _scan_cameras(self):
        cams = []
        if HAS_PYGRABBER:
            try:
                graph = FilterGraph()
                devices = graph.get_input_devices()
                for i, name in enumerate(devices):
                    cams.append((i, name))
            except: pass
        if not cams:
            for i in range(5):
                cams.append((i, f"Camera Device {i}"))
        return cams

    def _init_ui(self):
        # === 전체 3단 레이아웃 ===
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # === 상단: 타이틀 영역 ===
        title_area = QWidget()
        title_area.setStyleSheet("background-color: #313338;")
        title_layout = QVBoxLayout(title_area)
        title_layout.setContentsMargins(30, 20, 30, 20)
        title_layout.setSpacing(4)

        lbl_title = QLabel("PROJECT MUSE")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("""
            color: #5865f2;
            font-size: 28px;
            font-weight: 800;
            letter-spacing: 0.05em;
        """)
        title_layout.addWidget(lbl_title)

        lbl_subtitle = QLabel("AI 실시간 방송 시스템 v5.3")
        lbl_subtitle.setAlignment(Qt.AlignCenter)
        lbl_subtitle.setStyleSheet("color: #949ba4; font-size: 12px;")
        title_layout.addWidget(lbl_subtitle)

        outer_layout.addWidget(title_area)

        # === 중단: 설정 영역 ===
        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: #1e1f22;")
        main_layout = QHBoxLayout(content_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- 왼쪽 패널: 프로필 목록 ---
        left_panel = QVBoxLayout()

        lbl_list = QLabel("프로필 목록")
        lbl_list.setStyleSheet("font-size: 11px; font-weight: 600; color: #949ba4; text-transform: uppercase; letter-spacing: 0.02em;")
        left_panel.addWidget(lbl_list)

        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._on_profile_selected)
        left_panel.addWidget(self.list_widget)

        # 새 프로필 생성
        grp_create = QGroupBox("새 프로필 생성")
        create_layout = QVBoxLayout()
        self.input_new_name = QLineEdit()
        self.input_new_name.setPlaceholderText("프로필 이름 (예: side_cam)")

        hk_layout = QHBoxLayout()
        hk_layout.addWidget(QLabel("단축키:"))
        self.input_new_hotkey = QKeySequenceEdit()
        self.input_new_hotkey.setKeySequence(QKeySequence(""))
        hk_layout.addWidget(self.input_new_hotkey)

        btn_create = QPushButton("생성")
        btn_create.clicked.connect(self._create_profile)

        create_layout.addWidget(self.input_new_name)
        create_layout.addLayout(hk_layout)
        create_layout.addWidget(btn_create)
        grp_create.setLayout(create_layout)
        left_panel.addWidget(grp_create)

        main_layout.addLayout(left_panel, stretch=2)

        # --- 오른쪽 패널: 프로필 설정 ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # 프로필 설정 헤더
        lbl_settings = QLabel("선택된 프로필 설정")
        lbl_settings.setStyleSheet("""
            font-size: 11px;
            font-weight: 600;
            color: #949ba4;
            text-transform: uppercase;
            letter-spacing: 0.02em;
            padding-bottom: 5px;
        """)
        right_panel.addWidget(lbl_settings)

        # 설정 컨테이너 (스크롤 가능)
        from PySide6.QtWidgets import QScrollArea
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 8px;
                background: #2b2d31;
            }
            QScrollBar:vertical {
                background: #2b2d31;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #4e5058;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        settings_content = QWidget()
        settings_content.setStyleSheet("background: #2b2d31;")
        info_layout = QVBoxLayout(settings_content)
        info_layout.setContentsMargins(15, 15, 15, 15)
        info_layout.setSpacing(12)

        # 카메라 설정
        lbl_cam = QLabel("연결된 카메라")
        lbl_cam.setStyleSheet("color: #dbdee1; font-weight: 600; font-size: 12px;")
        info_layout.addWidget(lbl_cam)
        self.combo_cam = QComboBox()
        self.combo_cam.setMinimumHeight(36)
        for idx, name in self.available_cameras:
            self.combo_cam.addItem(f"[{idx}] {name}", idx)
        info_layout.addWidget(self.combo_cam)

        info_layout.addWidget(self._create_separator())

        # 단축키 설정
        lbl_hotkey = QLabel("지정 단축키")
        lbl_hotkey.setStyleSheet("color: #dbdee1; font-weight: 600; font-size: 12px;")
        info_layout.addWidget(lbl_hotkey)
        self.edit_hotkey = QKeySequenceEdit()
        self.edit_hotkey.setMinimumHeight(36)
        info_layout.addWidget(self.edit_hotkey)

        info_layout.addWidget(self._create_separator())

        # 배경 상태
        self.lbl_bg_status = QLabel("배경 상태: 확인 중...")
        self.lbl_bg_status.setStyleSheet("font-size: 12px; color: #949ba4;")
        self.lbl_bg_status.setWordWrap(True)
        info_layout.addWidget(self.lbl_bg_status)

        info_layout.addWidget(self._create_separator())

        # 구동 모드 선택
        lbl_mode = QLabel("구동 모드 선택")
        lbl_mode.setStyleSheet("color: #dbdee1; font-weight: 600; font-size: 12px;")
        info_layout.addWidget(lbl_mode)

        self.mode_group = QButtonGroup(self)

        self.rb_standard = QRadioButton("기본 (Standard)")
        self.rb_standard.setStyleSheet("font-size: 12px;")
        self.rb_high = QRadioButton("고정밀 (LoRA)")
        self.rb_high.setStyleSheet("font-size: 12px;")
        self.rb_personal = QRadioButton("퍼스널 (Personal)")
        self.rb_personal.setStyleSheet("font-size: 12px;")

        self.mode_group.addButton(self.rb_standard, 0)
        self.mode_group.addButton(self.rb_high, 1)
        self.mode_group.addButton(self.rb_personal, 2)

        self.rb_standard.setChecked(True)
        self.mode_group.buttonClicked.connect(self._on_mode_changed)

        info_layout.addWidget(self.rb_standard)
        info_layout.addWidget(self.rb_high)
        info_layout.addWidget(self.rb_personal)

        info_layout.addWidget(self._create_separator())

        # 설정 저장 버튼
        btn_save = QPushButton("설정 저장")
        btn_save.setMinimumHeight(38)
        btn_save.clicked.connect(self._save_current_settings)
        info_layout.addWidget(btn_save)

        info_layout.addStretch()

        settings_scroll.setWidget(settings_content)
        right_panel.addWidget(settings_scroll)

        # 프로필 삭제 버튼
        btn_delete = QPushButton("프로필 삭제")
        btn_delete.setObjectName("Danger")
        btn_delete.setMinimumHeight(38)
        btn_delete.clicked.connect(self._delete_profile)
        right_panel.addWidget(btn_delete)

        main_layout.addLayout(right_panel, stretch=3)

        outer_layout.addWidget(content_widget, stretch=1)

        # === 하단: 액션 버튼 영역 ===
        action_area = QWidget()
        action_area.setStyleSheet("background-color: #232428;")
        action_layout = QVBoxLayout(action_area)
        action_layout.setContentsMargins(30, 20, 30, 25)
        action_layout.setSpacing(15)

        # 안내 텍스트
        lbl_action_hint = QLabel("시작하기")
        lbl_action_hint.setStyleSheet("""
            color: #949ba4;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        """)
        action_layout.addWidget(lbl_action_hint)

        # 버튼 컨테이너
        btn_container = QHBoxLayout()
        btn_container.setSpacing(20)

        # 방송 시작 버튼 (카드 스타일)
        broadcast_card = QWidget()
        broadcast_card.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00D4DB, stop:1 #7B61FF);
                border-radius: 16px;
            }
            QWidget:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00E5EC, stop:1 #8B71FF);
            }
        """)
        broadcast_card.setCursor(Qt.PointingHandCursor)
        broadcast_card.setFixedHeight(100)
        broadcast_layout = QVBoxLayout(broadcast_card)
        broadcast_layout.setContentsMargins(20, 15, 20, 15)
        broadcast_layout.setSpacing(4)

        lbl_broadcast_title = QLabel("방송 시작하기")
        lbl_broadcast_title.setStyleSheet("color: white; font-size: 20px; font-weight: 700; background: transparent;")
        broadcast_layout.addWidget(lbl_broadcast_title)

        lbl_broadcast_desc = QLabel("선택한 프로필로 AI 방송을 시작합니다")
        lbl_broadcast_desc.setStyleSheet("color: rgba(255,255,255,0.8); font-size: 12px; background: transparent;")
        broadcast_layout.addWidget(lbl_broadcast_desc)

        broadcast_layout.addStretch()

        self.btn_start_broadcast = QPushButton()
        self.btn_start_broadcast.setFixedSize(0, 0)  # 숨김 (클릭 이벤트용)
        broadcast_card.mousePressEvent = lambda e: self._on_start_broadcast()

        btn_container.addWidget(broadcast_card, stretch=1)

        # 스튜디오 열기 버튼 (카드 스타일)
        studio_card = QWidget()
        studio_card.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF6D00, stop:1 #FF9100);
                border-radius: 16px;
            }
            QWidget:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF8F00, stop:1 #FFAB00);
            }
        """)
        studio_card.setCursor(Qt.PointingHandCursor)
        studio_card.setFixedHeight(100)
        studio_layout = QVBoxLayout(studio_card)
        studio_layout.setContentsMargins(20, 15, 20, 15)
        studio_layout.setSpacing(4)

        lbl_studio_title = QLabel("AI 학습 스튜디오")
        lbl_studio_title.setStyleSheet("color: white; font-size: 20px; font-weight: 700; background: transparent;")
        studio_layout.addWidget(lbl_studio_title)

        lbl_studio_desc = QLabel("나만의 AI 모델을 학습시킵니다")
        lbl_studio_desc.setStyleSheet("color: rgba(255,255,255,0.8); font-size: 12px; background: transparent;")
        studio_layout.addWidget(lbl_studio_desc)

        studio_layout.addStretch()

        self.btn_open_studio = QPushButton()
        self.btn_open_studio.setFixedSize(0, 0)  # 숨김 (클릭 이벤트용)
        studio_card.mousePressEvent = lambda e: self._on_open_studio()

        btn_container.addWidget(studio_card, stretch=1)

        action_layout.addLayout(btn_container)

        outer_layout.addWidget(action_area)

    def _refresh_list(self):
        self.pm.scan_profiles()
        self.list_widget.clear()
        profiles = self.pm.get_profile_list()

        for p in profiles:
            cfg = self.pm.get_config(p)
            hotkey = cfg.get("hotkey", "")
            if not hotkey: hotkey = "(없음)"

            can_personal = self._check_personal(p)
            can_lora = self._check_lora(p)

            tags = []
            if can_personal: tags.append("Personal")
            if can_lora: tags.append("LoRA")

            tag_str = f"[{'|'.join(tags)}]" if tags else "[Standard]"

            item_text = f"{tag_str}  {p.upper()}  (Key: {hotkey})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, p)

            if tags: item.setForeground(Qt.white)

            self.list_widget.addItem(item)

        if self.list_widget.count() > 0:
            if not self.selected_profile:
                self.list_widget.setCurrentRow(0)
                self._on_profile_selected(self.list_widget.item(0))
            else:
                items = self.list_widget.findItems(self.selected_profile.upper(), Qt.MatchContains)
                if items:
                    self.list_widget.setCurrentItem(items[0])
                    self._on_profile_selected(items[0])

    def _check_personal(self, p):
        s = os.path.join(self.model_dir, f"student_seg_{p}.engine")
        k = os.path.join(self.model_dir, f"student_pose_{p}.engine")
        return os.path.exists(s) and os.path.exists(k)

    def _check_lora(self, p):
        return os.path.exists(os.path.join(self.model_dir, f"vitpose_lora_{p}.engine"))

    def _on_profile_selected(self, item):
        p_name = item.data(Qt.UserRole)
        self.selected_profile = p_name

        config = self.pm.get_config(p_name)
        cam_id = config.get("camera_id", 0)
        hotkey = config.get("hotkey", "")

        idx = self.combo_cam.findData(cam_id)
        if idx >= 0: self.combo_cam.setCurrentIndex(idx)

        self.edit_hotkey.setKeySequence(QKeySequence(hotkey))

        bg_path = os.path.join(self.pm.get_profile_path(p_name), "background.jpg")
        if os.path.exists(bg_path):
            self.lbl_bg_status.setText("배경: 있음")
            self.lbl_bg_status.setStyleSheet("color: #23a55a; font-size: 12px;")
        else:
            self.lbl_bg_status.setText("배경: 없음")
            self.lbl_bg_status.setStyleSheet("color: #f0b232; font-size: 12px;")

        can_personal = self._check_personal(p_name)
        can_lora = self._check_lora(p_name)

        self.rb_personal.setEnabled(can_personal)
        self.rb_high.setEnabled(can_lora)

        if can_personal:
            self.rb_personal.setChecked(True)
            self.selected_mode = "PERSONAL"
        elif can_lora:
            self.rb_high.setChecked(True)
            self.selected_mode = "LORA"
        else:
            self.rb_standard.setChecked(True)
            self.selected_mode = "STANDARD"

        self.rb_personal.setText(f"퍼스널 {'[OK]' if can_personal else '[학습필요]'}")
        self.rb_high.setText(f"고정밀 LoRA {'[OK]' if can_lora else '[학습필요]'}")

    def _on_mode_changed(self, btn):
        if btn == self.rb_standard: self.selected_mode = "STANDARD"
        elif btn == self.rb_high: self.selected_mode = "LORA"
        elif btn == self.rb_personal: self.selected_mode = "PERSONAL"

    def _create_profile(self):
        name = self.input_new_name.text().strip()
        if not name: return
        cam_id = self.combo_cam.currentData()
        hotkey_seq = self.input_new_hotkey.keySequence().toString(QKeySequence.NativeText)

        if self.pm.create_profile(name, cam_id, hotkey_seq):
            self.input_new_name.clear()
            self.input_new_hotkey.setKeySequence(QKeySequence(""))
            self.selected_profile = name
            self._refresh_list()
        else:
            QMessageBox.warning(self, "오류", "이미 존재하는 이름입니다.")

    def _save_current_settings(self):
        if not self.selected_profile: return
        cam_id = self.combo_cam.currentData()
        hotkey_seq = self.edit_hotkey.keySequence().toString(QKeySequence.NativeText)
        self.pm.update_camera_id(self.selected_profile, cam_id)
        self.pm.update_hotkey(self.selected_profile, hotkey_seq)
        QMessageBox.information(self, "저장", "설정이 저장되었습니다.")
        self._refresh_list()

    def _delete_profile(self):
        if not self.selected_profile or self.selected_profile == "default":
            return
        ret = QMessageBox.question(self, "삭제 확인", f"정말 삭제하시겠습니까?", QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            self.pm.delete_profile(self.selected_profile)
            self.selected_profile = None
            self._refresh_list()

    def _on_start_broadcast(self):
        """방송 시작 버튼 클릭"""
        if self.selected_profile:
            self.start_broadcast.emit(self.selected_profile, self.selected_mode)
        else:
            QMessageBox.warning(self, "알림", "프로필을 선택해주세요.")

    def _on_open_studio(self):
        """스튜디오 열기 버튼 클릭"""
        self.open_studio.emit()

    def get_start_config(self):
        """레거시 호환용: (profile_name, run_mode) 반환"""
        return self.selected_profile, self.selected_mode

    def refresh_on_show(self):
        """페이지가 표시될 때 호출하여 목록 새로고침"""
        self._refresh_list()
