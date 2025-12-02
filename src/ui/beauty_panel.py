# Project MUSE - beauty_panel.py
# Clean Version: No Capture Button, Auto-Sync Support
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

        # [V4.0] 파라미터 리셋 (The Quartet)
        self.current_params = {
            'eye_scale': 0.0,
            'face_v': 0.0,
            'head_scale': 0.0,
            'shoulder_narrow': 0.0, # [1] 어깨
            'ribcage_slim': 0.0,    # [2] 흉통
            'waist_slim': 0.0,      # [3] 허리
            'hip_widen': 0.0,       # [4] 골반
            'show_body_debug': False
        }

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 20, 10, 20)

        # 타이틀
        self.title_label = QLabel("MUSE ENGINE")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #FFFFFF; margin-bottom: 10px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

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
        
        face_warp_group = QGroupBox("Face Reshape")
        self._style_groupbox(face_warp_group)
        fw_layout = QVBoxLayout()
        
        self.slider_eye = ModernSlider("Eye Size", initial_value=0.0)
        self.slider_eye.valueChanged.connect(lambda v: self._update_param('eye_scale', v))
        fw_layout.addWidget(self.slider_eye)

        self.slider_chin = ModernSlider("V-Line", initial_value=0.0)
        self.slider_chin.valueChanged.connect(lambda v: self._update_param('face_v', v))
        fw_layout.addWidget(self.slider_chin)

        self.slider_head = ModernSlider("Head Size", initial_value=0.0)
        self.slider_head.valueChanged.connect(lambda v: self._update_param('head_scale', v))
        fw_layout.addWidget(self.slider_head)
        
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
        body_warp_group = QGroupBox("Body Quartet")
        self._style_groupbox(body_warp_group)
        bw_layout = QVBoxLayout()

        # [1] 어깨 (Narrow)
        self.slider_shoulder = ModernSlider("Shoulder", initial_value=0.0)
        self.slider_shoulder.valueChanged.connect(lambda v: self._update_param('shoulder_narrow', v))
        bw_layout.addWidget(self.slider_shoulder)

        # [2] 흉통 (Ribcage)
        self.slider_ribcage = ModernSlider("Ribcage", initial_value=0.0)
        self.slider_ribcage.valueChanged.connect(lambda v: self._update_param('ribcage_slim', v))
        bw_layout.addWidget(self.slider_ribcage)

        # [3] 허리 (Waist)
        self.slider_waist = ModernSlider("Waist", initial_value=0.0)
        self.slider_waist.valueChanged.connect(lambda v: self._update_param('waist_slim', v))
        bw_layout.addWidget(self.slider_waist)

        # [4] 골반 (Hip)
        self.slider_hip = ModernSlider("Hip", initial_value=0.0)
        self.slider_hip.valueChanged.connect(lambda v: self._update_param('hip_widen', v))
        bw_layout.addWidget(self.slider_hip)

        body_warp_group.setLayout(bw_layout)
        body_layout.addWidget(body_warp_group)
        body_layout.addStretch()
        body_tab.setLayout(body_layout)

        # 탭 추가
        tabs.addTab(face_tab, "FACE")
        tabs.addTab(body_tab, "BODY")
        layout.addWidget(tabs)

        layout.addStretch()
        self.info_label = QLabel("Mode: Default")
        self.info_label.setStyleSheet("color: #555; font-size: 10px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

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

    def set_profile_info(self, profile_name):
        self.title_label.setText(f"MUSE: {profile_name.upper()}")
        self.info_label.setText(f"Active Profile: {profile_name}")

    def update_sliders_from_config(self, params):
        """
        [New] 외부 설정값(JSON)을 슬라이더에 반영
        - 중요: 슬라이더 값 변경 시 _update_param이 호출되어 다시 시그널을 보내는 것을 방지해야 함.
        - blockSignals를 사용하여 조용히 UI만 업데이트.
        """
        # 시그널 차단 (루프 방지)
        self.blockSignals(True)
        self.slider_eye.blockSignals(True)
        self.slider_chin.blockSignals(True)
        self.slider_head.blockSignals(True)
        self.slider_shoulder.blockSignals(True)
        self.slider_ribcage.blockSignals(True)
        self.slider_waist.blockSignals(True)
        self.slider_hip.blockSignals(True)
        self.chk_body_debug.blockSignals(True)

        # 값 설정
        if 'eye_scale' in params: self.slider_eye.set_value(params['eye_scale'])
        if 'face_v' in params: self.slider_chin.set_value(params['face_v'])
        if 'head_scale' in params: self.slider_head.set_value(params['head_scale'])
        
        if 'shoulder_narrow' in params: self.slider_shoulder.set_value(params['shoulder_narrow'])
        if 'ribcage_slim' in params: self.slider_ribcage.set_value(params['ribcage_slim'])
        if 'waist_slim' in params: self.slider_waist.set_value(params['waist_slim'])
        if 'hip_widen' in params: self.slider_hip.set_value(params['hip_widen'])
        
        if 'show_body_debug' in params: 
            self.chk_body_debug.setChecked(bool(params['show_body_debug']))

        # 내부 변수 동기화
        self.current_params.update(params)

        # 차단 해제
        self.chk_body_debug.blockSignals(False)
        self.slider_hip.blockSignals(False)
        self.slider_waist.blockSignals(False)
        self.slider_ribcage.blockSignals(False)
        self.slider_shoulder.blockSignals(False)
        self.slider_head.blockSignals(False)
        self.slider_chin.blockSignals(False)
        self.slider_eye.blockSignals(False)
        self.blockSignals(False)
        
        # 워커에게 "설정값 바뀌었어"라고 한번 알려주는게 좋음
        self.paramChanged.emit(self.current_params)