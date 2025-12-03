# Project MUSE - beauty_panel.py
# Clean Version: No Capture Button, Auto-Sync Support
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QTabWidget, QCheckBox, QFrame
)
from PySide6.QtCore import Signal, Qt
from ui.controls.sliders import ModernSlider

class BeautyPanel(QWidget):
    """
    [UI Panel] 뷰티 파라미터를 조절하는 우측 사이드바
    Modern Design Applied
    """
    paramChanged = Signal(dict)

    def __init__(self):
        super().__init__()
        
        # Modern Stylesheet for Panel
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #EEEEEE;
                font-family: 'Segoe UI', sans-serif;
            }
            QTabWidget::pane {
                border: none;
                background: #1E1E1E;
                border-top: 2px solid #00ADB5; /* Teal Line */
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background: #121212;
                color: #888;
                padding: 10px 30px;
                font-weight: bold;
                font-size: 13px;
                border: none;
            }
            QTabBar::tab:selected {
                color: #00ADB5;
                background: #1E1E1E; /* Blend with Pane */
            }
            QTabBar::tab:hover {
                color: #FFFFFF;
            }
            QGroupBox {
                border: none;
                margin-top: 10px;
                background: #1E1E1E;
                border-radius: 8px;
                padding-top: 25px; /* Title Space */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #00ADB5; /* Accent Color */
                font-size: 12px;
                font-weight: bold;
                text-transform: uppercase;
            }
            QCheckBox {
                color: #AAA;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #555;
                background: #2D2D2D;
            }
            QCheckBox::indicator:checked {
                background: #00ADB5;
                border-color: #00ADB5;
            }
        """)
        
        self.setFixedWidth(350) # Slightly wider

        self.current_params = {
            'eye_scale': 0.0,
            'face_v': 0.0,
            'head_scale': 0.0,
            'shoulder_narrow': 0.0,
            'ribcage_slim': 0.0, 
            'waist_slim': 0.0,
            'hip_widen': 0.0,
            'show_body_debug': False
        }

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # 1. Header Area
        header = QFrame()
        header.setStyleSheet("background-color: #121212; padding: 20px;")
        h_layout = QVBoxLayout(header)
        
        self.title_label = QLabel("MUSE ENGINE")
        self.title_label.setStyleSheet("font-size: 22px; font-weight: 800; letter-spacing: 1px; color: #FFF;")
        self.title_label.setAlignment(Qt.AlignCenter)
        h_layout.addWidget(self.title_label)
        
        self.info_label = QLabel("ACTIVE PROFILE: DEFAULT")
        self.info_label.setStyleSheet("color: #666; font-size: 11px; font-weight: bold;")
        self.info_label.setAlignment(Qt.AlignCenter)
        h_layout.addWidget(self.info_label)
        
        layout.addWidget(header)

        # 2. Tabs
        tabs = QTabWidget()

        # --- Face Tab ---
        face_tab = QWidget()
        face_layout = QVBoxLayout()
        face_layout.setContentsMargins(15, 20, 15, 20)
        face_layout.setSpacing(25)
        
        face_group = QGroupBox("Facial Geometry")
        f_inner = QVBoxLayout()
        f_inner.setSpacing(15)
        
        self.slider_eye = ModernSlider("EYES", 0.0)
        self.slider_eye.valueChanged.connect(lambda v: self._update_param('eye_scale', v))
        f_inner.addWidget(self.slider_eye)

        self.slider_chin = ModernSlider("V-LINE", 0.0)
        self.slider_chin.valueChanged.connect(lambda v: self._update_param('face_v', v))
        f_inner.addWidget(self.slider_chin)

        self.slider_head = ModernSlider("HEAD", 0.0)
        self.slider_head.valueChanged.connect(lambda v: self._update_param('head_scale', v))
        f_inner.addWidget(self.slider_head)
        
        face_group.setLayout(f_inner)
        face_layout.addWidget(face_group)
        face_layout.addStretch()
        face_tab.setLayout(face_layout)

        # --- Body Tab ---
        body_tab = QWidget()
        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(15, 20, 15, 20)
        body_layout.setSpacing(25)

        # Debug
        debug_group = QGroupBox("Visualization")
        d_inner = QVBoxLayout()
        self.chk_body_debug = QCheckBox("Show Skeleton Overlay")
        self.chk_body_debug.toggled.connect(lambda v: self._update_param('show_body_debug', v))
        d_inner.addWidget(self.chk_body_debug)
        debug_group.setLayout(d_inner)
        body_layout.addWidget(debug_group)

        # Body Sliders
        body_group = QGroupBox("Body Morphing")
        b_inner = QVBoxLayout()
        b_inner.setSpacing(15)

        self.slider_shoulder = ModernSlider("SHOULDERS", 0.0)
        self.slider_shoulder.valueChanged.connect(lambda v: self._update_param('shoulder_narrow', v))
        b_inner.addWidget(self.slider_shoulder)

        self.slider_ribcage = ModernSlider("RIBCAGE", 0.0)
        self.slider_ribcage.valueChanged.connect(lambda v: self._update_param('ribcage_slim', v))
        b_inner.addWidget(self.slider_ribcage)

        self.slider_waist = ModernSlider("WAIST", 0.0)
        self.slider_waist.valueChanged.connect(lambda v: self._update_param('waist_slim', v))
        b_inner.addWidget(self.slider_waist)

        self.slider_hip = ModernSlider("HIPS", 0.0)
        self.slider_hip.valueChanged.connect(lambda v: self._update_param('hip_widen', v))
        b_inner.addWidget(self.slider_hip)

        body_group.setLayout(b_inner)
        body_layout.addWidget(body_group)
        body_layout.addStretch()
        body_tab.setLayout(body_layout)

        tabs.addTab(face_tab, "FACE")
        tabs.addTab(body_tab, "BODY")
        layout.addWidget(tabs)

        self.setLayout(layout)

    def _update_param(self, key, value):
        self.current_params[key] = value
        self.paramChanged.emit(self.current_params)

    def set_profile_info(self, profile_name):
        self.info_label.setText(f"ACTIVE PROFILE: {profile_name.upper()}")

    def update_sliders_from_config(self, params):
        self.blockSignals(True)
        # Block Children Signals
        for s in [self.slider_eye, self.slider_chin, self.slider_head, 
                  self.slider_shoulder, self.slider_ribcage, self.slider_waist, self.slider_hip]:
            s.blockSignals(True)
        self.chk_body_debug.blockSignals(True)

        if 'eye_scale' in params: self.slider_eye.set_value(params['eye_scale'])
        if 'face_v' in params: self.slider_chin.set_value(params['face_v'])
        if 'head_scale' in params: self.slider_head.set_value(params['head_scale'])
        
        if 'shoulder_narrow' in params: self.slider_shoulder.set_value(params['shoulder_narrow'])
        if 'ribcage_slim' in params: self.slider_ribcage.set_value(params['ribcage_slim'])
        if 'waist_slim' in params: self.slider_waist.set_value(params['waist_slim'])
        if 'hip_widen' in params: self.slider_hip.set_value(params['hip_widen'])
        
        if 'show_body_debug' in params: 
            self.chk_body_debug.setChecked(bool(params['show_body_debug']))

        self.current_params.update(params)

        for s in [self.slider_eye, self.slider_chin, self.slider_head, 
                  self.slider_shoulder, self.slider_ribcage, self.slider_waist, self.slider_hip]:
            s.blockSignals(False)
        self.chk_body_debug.blockSignals(False)
        self.blockSignals(False)
        
        self.paramChanged.emit(self.current_params)