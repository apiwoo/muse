# Project MUSE - beauty_panel.py
# Clean Version: No Capture Button
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QTabWidget, QCheckBox
)
from PySide6.QtCore import Signal, Qt
from ui.controls.sliders import ModernSlider

class BeautyPanel(QWidget):
    """
    [UI Panel] 뷰티 파라미터를 조절하는 우측 사이드바
    """
    paramChanged = Signal(dict)

    def __init__(self):
        super().__init__()
        
        self.setStyleSheet("background-color: #1E1E1E;")
        self.setFixedWidth(320)

        # 초기 파라미터
        self.current_params = {
            'eye_scale': 0.0,
            'face_v': 0.0,
            'waist_slim': 0.0,
            'show_body_debug': False
        }

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 20, 10, 20)

        # 타이틀
        title = QLabel("MUSE ENGINE")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #FFFFFF; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 탭 위젯 생성
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; background: #252525; }
            QTabBar::tab {
                background: #1E1E1E; color: #888; padding: 8px 20px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background: #333; color: #00ADB5; font-weight: bold; }
        """)

        # 1. Face 탭
        face_tab = QWidget()
        face_layout = QVBoxLayout()
        face_layout.setSpacing(20)
        
        # Face > Warping Group
        face_warp_group = QGroupBox("Face Reshape")
        self._style_groupbox(face_warp_group)
        fw_layout = QVBoxLayout()
        
        self.slider_eye = ModernSlider("Eye Size", initial_value=0.0)
        self.slider_eye.valueChanged.connect(lambda v: self._update_param('eye_scale', v))
        fw_layout.addWidget(self.slider_eye)

        self.slider_chin = ModernSlider("V-Line", initial_value=0.0)
        self.slider_chin.valueChanged.connect(lambda v: self._update_param('face_v', v))
        fw_layout.addWidget(self.slider_chin)
        
        face_warp_group.setLayout(fw_layout)
        face_layout.addWidget(face_warp_group)
        face_layout.addStretch()
        face_tab.setLayout(face_layout)

        # 2. Body 탭
        body_tab = QWidget()
        body_layout = QVBoxLayout()
        body_layout.setSpacing(20)

        # Body > Debug Tools
        debug_group = QGroupBox("Debug Tools")
        self._style_groupbox(debug_group)
        debug_layout = QVBoxLayout()
        
        self.chk_body_debug = QCheckBox("Show Skeleton (뼈대 보기)")
        self.chk_body_debug.setStyleSheet("color: #DDD; font-size: 13px;")
        self.chk_body_debug.toggled.connect(lambda v: self._update_param('show_body_debug', v))
        debug_layout.addWidget(self.chk_body_debug)
        
        debug_group.setLayout(debug_layout)
        body_layout.addWidget(debug_group)

        # Body > Reshape Group
        body_warp_group = QGroupBox("Body Reshape")
        self._style_groupbox(body_warp_group)
        bw_layout = QVBoxLayout()

        self.slider_waist = ModernSlider("Waist Slim", initial_value=0.0)
        self.slider_waist.valueChanged.connect(lambda v: self._update_param('waist_slim', v))
        bw_layout.addWidget(self.slider_waist)

        body_warp_group.setLayout(bw_layout)
        body_layout.addWidget(body_warp_group)
        body_layout.addStretch()
        body_tab.setLayout(body_layout)

        # 탭 추가
        tabs.addTab(face_tab, "FACE")
        tabs.addTab(body_tab, "BODY")
        layout.addWidget(tabs)

        # 하단 정보
        layout.addStretch()
        info_label = QLabel("Mode A: Visual Supremacy (MediaPipe)")
        info_label.setStyleSheet("color: #555; font-size: 10px;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

        self.setLayout(layout)

    def _style_groupbox(self, group):
        group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                color: #AAA;
                font-weight: bold;
                background: #2A2A2A;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)

    def _update_param(self, key, value):
        self.current_params[key] = value
        self.paramChanged.emit(self.current_params)