# Project MUSE - launcher.py
# The Gatekeeper: Profile & Camera Manager
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import cv2
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QListWidget, QListWidgetItem, QComboBox, QLineEdit, QMessageBox, 
    QGroupBox, QFrame, QKeySequenceEdit
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap, QKeySequence

# Try to import pygrabber for camera names
try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False

from utils.config import ProfileManager

class LauncherDialog(QDialog):
    """
    [App Launcher]
    - 프로필 선택/생성/삭제
    - 카메라 ID 지정
    - 배경 유무 확인
    - 엔진 시작
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MUSE 스튜디오 설정 (v5.1 - Hotkey Support)")
        self.resize(850, 550)
        self.setStyleSheet("""
            QDialog { background-color: #1E1E1E; color: #EEE; font-family: 'Segoe UI'; }
            QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 20px; font-weight: bold; color: #00ADB5; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QListWidget { background-color: #252525; border: 1px solid #333; color: white; border-radius: 5px; font-size: 14px; }
            QListWidget::item { padding: 8px; }
            QListWidget::item:selected { background-color: #00ADB5; color: white; }
            QLabel { color: #CCC; }
            QLineEdit, QComboBox, QKeySequenceEdit { background-color: #333; border: 1px solid #555; padding: 5px; color: white; border-radius: 4px; }
            QPushButton { background-color: #444; border: none; padding: 8px 15px; color: white; border-radius: 4px; }
            QPushButton:hover { background-color: #555; }
            QPushButton#Primary { background-color: #00ADB5; font-weight: bold; font-size: 14px; }
            QPushButton#Primary:hover { background-color: #00C4CC; }
            QPushButton#Danger { background-color: #D32F2F; }
            QPushButton#Danger:hover { background-color: #E53935; }
        """)

        self.pm = ProfileManager()
        self.selected_profile = None
        self.available_cameras = self._scan_cameras()

        self._init_ui()
        self._refresh_list()

    def _scan_cameras(self):
        cams = []
        if HAS_PYGRABBER:
            try:
                graph = FilterGraph()
                devices = graph.get_input_devices()
                for i, name in enumerate(devices):
                    cams.append((i, name))
            except: pass
        
        # Fallback if empty
        if not cams:
            for i in range(5):
                cams.append((i, f"Camera Device {i}"))
        return cams

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # === LEFT: Profile List ===
        left_panel = QVBoxLayout()
        
        lbl_list = QLabel("📁 프로필 목록")
        lbl_list.setStyleSheet("font-size: 14px; font-weight: bold; color: white;")
        left_panel.addWidget(lbl_list)

        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._on_profile_selected)
        left_panel.addWidget(self.list_widget)

        # Create New
        grp_create = QGroupBox("새 프로필 생성")
        create_layout = QVBoxLayout()
        
        self.input_new_name = QLineEdit()
        self.input_new_name.setPlaceholderText("프로필 이름 (예: side_cam)")
        
        # [New] Hotkey Input for creation
        hk_layout = QHBoxLayout()
        hk_layout.addWidget(QLabel("단축키:"))
        self.input_new_hotkey = QKeySequenceEdit()
        self.input_new_hotkey.setKeySequence(QKeySequence(""))
        self.input_new_hotkey.setToolTip("이 프로필을 불러올 단축키를 입력하세요 (예: F1, Ctrl+1)")
        hk_layout.addWidget(self.input_new_hotkey)
        
        btn_create = QPushButton("생성")
        btn_create.clicked.connect(self._create_profile)
        
        create_layout.addWidget(self.input_new_name)
        create_layout.addLayout(hk_layout)
        create_layout.addWidget(btn_create)
        
        grp_create.setLayout(create_layout)
        left_panel.addWidget(grp_create)

        main_layout.addLayout(left_panel, stretch=2)

        # === RIGHT: Settings ===
        right_panel = QVBoxLayout()
        
        # Info Group
        grp_info = QGroupBox("선택된 프로필 설정")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(15)
        
        # Camera Select
        info_layout.addWidget(QLabel("연결된 카메라:"))
        self.combo_cam = QComboBox()
        for idx, name in self.available_cameras:
            self.combo_cam.addItem(f"[{idx}] {name}", idx)
        info_layout.addWidget(self.combo_cam)
        
        # [New] Hotkey Edit
        info_layout.addWidget(QLabel("지정 단축키:"))
        self.edit_hotkey = QKeySequenceEdit()
        info_layout.addWidget(self.edit_hotkey)
        
        # Background Status
        self.lbl_bg_status = QLabel("배경 상태: 확인 중...")
        self.lbl_bg_status.setStyleSheet("font-size: 12px; color: #888;")
        info_layout.addWidget(self.lbl_bg_status)

        # Save Button
        btn_save = QPushButton("설정 저장")
        btn_save.clicked.connect(self._save_current_settings)
        info_layout.addWidget(btn_save)

        grp_info.setLayout(info_layout)
        right_panel.addWidget(grp_info)

        # Delete Button
        btn_delete = QPushButton("프로필 삭제")
        btn_delete.setObjectName("Danger")
        btn_delete.clicked.connect(self._delete_profile)
        right_panel.addWidget(btn_delete)

        right_panel.addStretch()

        # Start Button
        self.btn_start = QPushButton("MUSE 방송 시작  🚀")
        self.btn_start.setObjectName("Primary")
        self.btn_start.setFixedHeight(50)
        self.btn_start.clicked.connect(self.accept) # Close dialog with Accepted result
        right_panel.addWidget(self.btn_start)

        main_layout.addLayout(right_panel, stretch=3)

    def _refresh_list(self):
        self.pm.scan_profiles()
        self.list_widget.clear()
        profiles = self.pm.get_profile_list()
        
        for p in profiles:
            cfg = self.pm.get_config(p)
            hotkey = cfg.get("hotkey", "")
            if not hotkey: hotkey = "(없음)"
            
            item_text = f"[{hotkey}]  {p.upper()}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, p)
            self.list_widget.addItem(item)
        
        if self.list_widget.count() > 0:
            if not self.selected_profile:
                self.list_widget.setCurrentRow(0)
                self._on_profile_selected(self.list_widget.item(0))
            else:
                # Keep selection if possible
                items = self.list_widget.findItems(self.selected_profile.upper(), Qt.MatchContains)
                if items:
                    self.list_widget.setCurrentItem(items[0])
                    self._on_profile_selected(items[0])

    def _on_profile_selected(self, item):
        p_name = item.data(Qt.UserRole)
        self.selected_profile = p_name
        
        config = self.pm.get_config(p_name)
        cam_id = config.get("camera_id", 0)
        hotkey = config.get("hotkey", "")
        
        # Set Combo
        idx = self.combo_cam.findData(cam_id)
        if idx >= 0: self.combo_cam.setCurrentIndex(idx)
        
        # Set Hotkey
        self.edit_hotkey.setKeySequence(QKeySequence(hotkey))
        
        # Check Background
        bg_path = os.path.join(self.pm.get_profile_path(p_name), "background.jpg")
        if os.path.exists(bg_path):
            self.lbl_bg_status.setText("✅ 배경 이미지 있음 (준비됨)")
            self.lbl_bg_status.setStyleSheet("color: #00ADB5;")
        else:
            self.lbl_bg_status.setText("⚠️ 배경 없음 (방송 시작 후 'B'를 눌러 촬영하세요)")
            self.lbl_bg_status.setStyleSheet("color: #FFA726;")

    def _create_profile(self):
        name = self.input_new_name.text().strip()
        if not name: return
        
        # Default to selected camera
        cam_id = self.combo_cam.currentData()
        # [New] Get Hotkey
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
        
        QMessageBox.information(self, "저장", f"[{self.selected_profile}] 설정이 저장되었습니다.")
        self._refresh_list() # Update list label

    def _delete_profile(self):
        if not self.selected_profile: return
        if self.selected_profile == "default":
            QMessageBox.warning(self, "불가", "기본 프로필은 삭제할 수 없습니다.")
            return
            
        ret = QMessageBox.question(self, "삭제 확인", f"정말 '{self.selected_profile}' 프로필을 삭제하시겠습니까?\n(모든 학습 데이터가 삭제됩니다)", 
                                   QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            self.pm.delete_profile(self.selected_profile)
            self.selected_profile = None # Clear selection
            self._refresh_list()

    def get_start_config(self):
        """Return selected profile name to start engine with"""
        return self.selected_profile