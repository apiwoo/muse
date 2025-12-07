# Project MUSE - launcher.py
# The Gatekeeper: Profile & Camera Manager
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import cv2
import glob
import subprocess
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QListWidget, QListWidgetItem, QComboBox, QLineEdit, QMessageBox, 
    QGroupBox, QFrame, QKeySequenceEdit
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap, QKeySequence

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
    - 배경 유무 확인 및 AI 모델 상태 표시
    - 학습 도구(Studio) 실행 기능 추가
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MUSE 스튜디오 설정 (v5.2 - Hybrid Mode UI)")
        self.resize(850, 600)
        self.setStyleSheet("""
            QDialog { background-color: #1E1E1E; color: #EEE; font-family: 'Segoe UI'; }
            QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 20px; font-weight: bold; color: #00ADB5; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QListWidget { background-color: #252525; border: 1px solid #333; color: white; border-radius: 5px; font-size: 14px; }
            QListWidget::item { padding: 10px; }
            QListWidget::item:selected { background-color: #00ADB5; color: white; }
            QLabel { color: #CCC; }
            QLineEdit, QComboBox, QKeySequenceEdit { background-color: #333; border: 1px solid #555; padding: 5px; color: white; border-radius: 4px; }
            QPushButton { background-color: #444; border: none; padding: 8px 15px; color: white; border-radius: 4px; }
            QPushButton:hover { background-color: #555; }
            QPushButton#Primary { background-color: #00ADB5; font-weight: bold; font-size: 14px; }
            QPushButton#Primary:hover { background-color: #00C4CC; }
            QPushButton#Danger { background-color: #D32F2F; }
            QPushButton#Danger:hover { background-color: #E53935; }
            QPushButton#Accent { background-color: #E65100; color: white; font-weight: bold; } 
            QPushButton#Accent:hover { background-color: #FF6F00; }
        """)

        self.pm = ProfileManager()
        self.selected_profile = None
        self.available_cameras = self._scan_cameras()
        
        # 모델 경로 확인용
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.root_dir, "assets", "models", "personal")

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
        
        # [New] Studio Launch Button
        btn_launch_studio = QPushButton("🎥 AI 모델 학습 스튜디오 열기")
        btn_launch_studio.setObjectName("Accent")
        btn_launch_studio.setFixedHeight(45)
        btn_launch_studio.setToolTip("데이터 녹화 및 AI 학습 도구를 실행합니다.")
        btn_launch_studio.clicked.connect(self._launch_studio_tool)
        left_panel.addWidget(btn_launch_studio)

        main_layout.addLayout(left_panel, stretch=2)

        # === RIGHT: Settings ===
        right_panel = QVBoxLayout()
        
        grp_info = QGroupBox("선택된 프로필 설정")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(15)
        
        info_layout.addWidget(QLabel("연결된 카메라:"))
        self.combo_cam = QComboBox()
        for idx, name in self.available_cameras:
            self.combo_cam.addItem(f"[{idx}] {name}", idx)
        info_layout.addWidget(self.combo_cam)
        
        info_layout.addWidget(QLabel("지정 단축키:"))
        self.edit_hotkey = QKeySequenceEdit()
        info_layout.addWidget(self.edit_hotkey)
        
        # Status Labels
        self.lbl_bg_status = QLabel("배경 상태: 확인 중...")
        self.lbl_bg_status.setStyleSheet("font-size: 12px; color: #888;")
        info_layout.addWidget(self.lbl_bg_status)
        
        self.lbl_model_status = QLabel("모델 상태: 확인 중...")
        self.lbl_model_status.setStyleSheet("font-size: 12px; color: #888;")
        info_layout.addWidget(self.lbl_model_status)

        btn_save = QPushButton("설정 저장")
        btn_save.clicked.connect(self._save_current_settings)
        info_layout.addWidget(btn_save)

        grp_info.setLayout(info_layout)
        right_panel.addWidget(grp_info)

        btn_delete = QPushButton("프로필 삭제")
        btn_delete.setObjectName("Danger")
        btn_delete.clicked.connect(self._delete_profile)
        right_panel.addWidget(btn_delete)

        right_panel.addStretch()

        self.btn_start = QPushButton("MUSE 방송 시작  🚀")
        self.btn_start.setObjectName("Primary")
        self.btn_start.setFixedHeight(50)
        self.btn_start.clicked.connect(self.accept)
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
            
            # [New] Check for Model
            has_model = self._check_model_exists(p)
            status_tag = "[모델 보유]" if has_model else "[기본 엔진]"
            
            item_text = f"{status_tag}  {p.upper()}  (Key: {hotkey})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, p)
            
            # Highlight if model exists
            if has_model:
                item.setForeground(Qt.cyan)
                
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

    def _check_model_exists(self, profile_name):
        seg_path = os.path.join(self.model_dir, f"student_seg_{profile_name}.engine")
        pose_path = os.path.join(self.model_dir, f"student_pose_{profile_name}.engine")
        return os.path.exists(seg_path) and os.path.exists(pose_path)

    def _on_profile_selected(self, item):
        p_name = item.data(Qt.UserRole)
        self.selected_profile = p_name
        
        config = self.pm.get_config(p_name)
        cam_id = config.get("camera_id", 0)
        hotkey = config.get("hotkey", "")
        
        idx = self.combo_cam.findData(cam_id)
        if idx >= 0: self.combo_cam.setCurrentIndex(idx)
        
        self.edit_hotkey.setKeySequence(QKeySequence(hotkey))
        
        # Check Background
        bg_path = os.path.join(self.pm.get_profile_path(p_name), "background.jpg")
        if os.path.exists(bg_path):
            self.lbl_bg_status.setText("✅ 배경 이미지 있음 (준비됨)")
            self.lbl_bg_status.setStyleSheet("color: #00ADB5;")
        else:
            self.lbl_bg_status.setText("⚠️ 배경 없음 (방송 시작 후 'B'를 눌러 촬영하세요)")
            self.lbl_bg_status.setStyleSheet("color: #FFA726;")
            
        # Check Model
        if self._check_model_exists(p_name):
            self.lbl_model_status.setText("✅ 개인화 모델 학습됨 (고품질)")
            self.lbl_model_status.setStyleSheet("color: #00ADB5;")
        else:
            self.lbl_model_status.setText("ℹ️ 기본 모델 사용 (MODNet+ViTPose)")
            self.lbl_model_status.setStyleSheet("color: #BBB;")

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
        
        QMessageBox.information(self, "저장", f"[{self.selected_profile}] 설정이 저장되었습니다.")
        self._refresh_list()

    def _delete_profile(self):
        if not self.selected_profile: return
        if self.selected_profile == "default":
            QMessageBox.warning(self, "불가", "기본 프로필은 삭제할 수 없습니다.")
            return
            
        ret = QMessageBox.question(self, "삭제 확인", f"정말 '{self.selected_profile}' 프로필을 삭제하시겠습니까?", 
                                   QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            self.pm.delete_profile(self.selected_profile)
            self.selected_profile = None 
            self._refresh_list()

    def _launch_studio_tool(self):
        """별도 프로세스로 학습 스튜디오 실행"""
        studio_script = os.path.join(self.root_dir, "tools", "muse_studio.py")
        if os.path.exists(studio_script):
            try:
                subprocess.Popen([sys.executable, studio_script])
            except Exception as e:
                QMessageBox.critical(self, "오류", f"스튜디오 실행 실패: {e}")
        else:
            QMessageBox.critical(self, "오류", f"파일을 찾을 수 없습니다: {studio_script}")

    def get_start_config(self):
        return self.selected_profile