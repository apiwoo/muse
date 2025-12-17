# Project MUSE - beauty_panel.py
# Integrated UI Layout (Single Tab for all Controls)
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QCheckBox, QFrame, QScrollArea
)
from PySide6.QtCore import Signal, Qt
from ui.controls.sliders import ModernSlider

class BeautyPanel(QWidget):
    """
    [UI Panel] 뷰티 파라미터를 조절하는 우측 사이드바
    Updated: 통합 탭 레이아웃 및 새로운 슬라이더 추가
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
            QGroupBox {
                border: none;
                margin-top: 10px;
                background: #1E1E1E;
                border-radius: 8px;
                padding-top: 25px; /* Title Space */
                padding-bottom: 15px;
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
                font-size: 13px;
                margin-left: 10px;
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
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #121212;
                width: 8px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #333;
                min-height: 20px;
                border-radius: 4px;
            }
        """)
        
        self.setFixedWidth(360) 

        self.current_params = {
            'eye_scale': 0.0,
            'face_v': 0.0,
            'nose_slim': 0.0, # [New]
            'head_scale': 0.0,
            'shoulder_narrow': 0.0,
            'ribcage_slim': 0.0, 
            'waist_slim': 0.0,
            'hip_widen': 0.0,
            'skin_smooth': 0.0, # [New]
            'show_body_debug': False
        }

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Header Area
        header = QFrame()
        header.setStyleSheet("background-color: #121212; padding: 20px;")
        h_layout = QVBoxLayout(header)
        
        self.title_label = QLabel("MUSE 뷰티 엔진")
        self.title_label.setStyleSheet("font-size: 22px; font-weight: 800; letter-spacing: 1px; color: #FFF;")
        self.title_label.setAlignment(Qt.AlignCenter)
        h_layout.addWidget(self.title_label)
        
        self.info_label = QLabel("통합 제어 모드")
        self.info_label.setStyleSheet("color: #666; font-size: 11px; font-weight: bold;")
        self.info_label.setAlignment(Qt.AlignCenter)
        h_layout.addWidget(self.info_label)
        
        main_layout.addWidget(header)

        # 2. Scroll Area for Controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 10, 15, 20)
        content_layout.setSpacing(20)

        # --- Group 1: 얼굴 보정 (Face) ---
        face_group = QGroupBox("얼굴 윤곽 (Face Shape)")
        f_inner = QVBoxLayout()
        f_inner.setSpacing(15)
        
        self.slider_chin = ModernSlider("턱 깎기 (V-Line)", 0.0)
        self.slider_chin.valueChanged.connect(lambda v: self._update_param('face_v', v))
        f_inner.addWidget(self.slider_chin)

        self.slider_eye = ModernSlider("눈 크기 조절", 0.0)
        self.slider_eye.valueChanged.connect(lambda v: self._update_param('eye_scale', v))
        f_inner.addWidget(self.slider_eye)

        self.slider_nose = ModernSlider("콧볼 조절", 0.0)
        self.slider_nose.valueChanged.connect(lambda v: self._update_param('nose_slim', v))
        f_inner.addWidget(self.slider_nose)

        # [Hidden Params] - 요청에 의해 숨김 처리 (하지만 인스턴스는 유지)
        self.slider_head = ModernSlider("머리 크기", 0.0)
        self.slider_head.setVisible(False) 
        self.slider_head.valueChanged.connect(lambda v: self._update_param('head_scale', v))
        f_inner.addWidget(self.slider_head)
        
        face_group.setLayout(f_inner)
        content_layout.addWidget(face_group)

        # --- Group 2: 전신 보정 (Body) ---
        body_group = QGroupBox("체형 보정 (Body Shape)")
        b_inner = QVBoxLayout()
        b_inner.setSpacing(15)

        self.slider_waist = ModernSlider("허리 줄이기", 0.0)
        self.slider_waist.valueChanged.connect(lambda v: self._update_param('waist_slim', v))
        b_inner.addWidget(self.slider_waist)

        self.slider_hip = ModernSlider("골반 늘리기", 0.0)
        self.slider_hip.valueChanged.connect(lambda v: self._update_param('hip_widen', v))
        b_inner.addWidget(self.slider_hip)

        # [Hidden Params]
        self.slider_shoulder = ModernSlider("어깨 보정", 0.0)
        self.slider_shoulder.setVisible(False)
        self.slider_shoulder.valueChanged.connect(lambda v: self._update_param('shoulder_narrow', v))
        b_inner.addWidget(self.slider_shoulder)

        self.slider_ribcage = ModernSlider("흉곽 줄임", 0.0)
        self.slider_ribcage.setVisible(False)
        self.slider_ribcage.valueChanged.connect(lambda v: self._update_param('ribcage_slim', v))
        b_inner.addWidget(self.slider_ribcage)

        body_group.setLayout(b_inner)
        content_layout.addWidget(body_group)

        # --- Group 3: 피부 및 효과 (Skin & Effect) ---
        skin_group = QGroupBox("피부 및 효과 (Skin & FX)")
        s_inner = QVBoxLayout()
        s_inner.setSpacing(15)

        self.slider_skin = ModernSlider("피부 보정 (Smooth)", 0.0)
        self.slider_skin.valueChanged.connect(lambda v: self._update_param('skin_smooth', v))
        s_inner.addWidget(self.slider_skin)

        skin_group.setLayout(s_inner)
        content_layout.addWidget(skin_group)

        # --- Debug ---
        debug_group = QGroupBox("설정")
        d_inner = QVBoxLayout()
        self.chk_body_debug = QCheckBox("AI 관절 보기 (Debug)")
        self.chk_body_debug.toggled.connect(lambda v: self._update_param('show_body_debug', v))
        d_inner.addWidget(self.chk_body_debug)
        debug_group.setLayout(d_inner)
        content_layout.addWidget(debug_group)

        content_layout.addStretch()
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        self.setLayout(main_layout)

    def _update_param(self, key, value):
        self.current_params[key] = value
        self.paramChanged.emit(self.current_params)

    def set_profile_info(self, profile_name):
        self.info_label.setText(f"프로파일: {profile_name.upper()}")

    def update_sliders_from_config(self, params):
        self.blockSignals(True)
        # Block Children Signals
        sliders = [
            self.slider_eye, self.slider_chin, self.slider_nose,
            self.slider_head, self.slider_shoulder, self.slider_ribcage, 
            self.slider_waist, self.slider_hip, self.slider_skin
        ]
        for s in sliders: s.blockSignals(True)
        self.chk_body_debug.blockSignals(True)

        # Map Params to Sliders
        if 'eye_scale' in params: self.slider_eye.set_value(params['eye_scale'])
        if 'face_v' in params: self.slider_chin.set_value(params['face_v'])
        if 'nose_slim' in params: self.slider_nose.set_value(params['nose_slim'])
        if 'head_scale' in params: self.slider_head.set_value(params['head_scale'])
        
        if 'waist_slim' in params: self.slider_waist.set_value(params['waist_slim'])
        if 'hip_widen' in params: self.slider_hip.set_value(params['hip_widen'])
        if 'shoulder_narrow' in params: self.slider_shoulder.set_value(params['shoulder_narrow'])
        if 'ribcage_slim' in params: self.slider_ribcage.set_value(params['ribcage_slim'])
        
        if 'skin_smooth' in params: self.slider_skin.set_value(params['skin_smooth'])
        
        if 'show_body_debug' in params: 
            self.chk_body_debug.setChecked(bool(params['show_body_debug']))

        self.current_params.update(params)

        for s in sliders: s.blockSignals(False)
        self.chk_body_debug.blockSignals(False)
        self.blockSignals(False)
        
        self.paramChanged.emit(self.current_params)