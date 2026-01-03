# Project MUSE - beauty_panel.py
# V25.0: Extended UI for High-Precision Pipeline
# - Added: Advanced Skin Settings (Flatten, Radius, Detail)
# - Added: Color Grading (Temperature, Tint)
# - Added: Pipeline Mode Toggle
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QCheckBox, QFrame, QScrollArea, QHBoxLayout, QPushButton
)
from PySide6.QtCore import Signal, Qt
from ui.controls.sliders import ModernSlider


class BeautyPanel(QWidget):
    """
    [UI Panel] 뷰티 파라미터를 조절하는 우측 사이드바
    V25.0: 고급 피부 설정 및 색상 그레이딩 추가
    """
    paramChanged = Signal(dict)
    bgCaptureRequested = Signal()  # [V5.0] 배경 저장 버튼 클릭 시 emit

    def __init__(self):
        super().__init__()

        self.setStyleSheet("""
            QWidget {
                background-color: #0A0A0A;
                color: #E0E0E0;
                font-family: Pretendard, Malgun Gothic, sans-serif;
            }
            QGroupBox {
                border: 1px solid rgba(255, 255, 255, 0.04);
                margin-top: 12px;
                background: rgba(255, 255, 255, 0.02);
                border-radius: 16px;
                padding-top: 28px;
                padding-bottom: 18px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                color: #00D4DB;
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 1.5px;
                text-transform: uppercase;
            }
            QCheckBox {
                color: rgba(255, 255, 255, 0.6);
                spacing: 10px;
                font-size: 13px;
                font-weight: 500;
                margin-left: 12px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                background: rgba(255, 255, 255, 0.05);
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00D4DB, stop:1 #7B61FF);
                border-color: transparent;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 6px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.08);
                min-height: 30px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00D4DB;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self.setFixedWidth(360)

        # Parameters
        self.current_params = {
            # 기존 파라미터
            'eye_scale': 0.0,
            'face_v': 0.0,
            'nose_slim': 0.0,
            'head_scale': 0.0,
            'shoulder_narrow': 0.0,
            'ribcage_slim': 0.0,
            'waist_slim': 0.0,
            'hip_widen': 0.0,
            'skin_smooth': 0.0,
            'show_body_debug': False,

            # 색상 조정
            'color_temperature': 0.0,   # 색온도 (-1 Cool ~ 1 Warm)
            'teeth_whiten': 0.0         # 치아 미백 (0 ~ 1)
        }

        # [V5.0] 배경 상태
        self.has_background = False

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Header Area
        header = QFrame()
        header.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #151515, stop:1 #0A0A0A); padding: 28px 20px; border-bottom: 1px solid rgba(255, 255, 255, 0.04);")
        h_layout = QVBoxLayout(header)

        self.title_label = QLabel("MUSE 뷰티 엔진")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 600; letter-spacing: 3px; color: #FFFFFF; text-transform: uppercase;")
        self.title_label.setAlignment(Qt.AlignCenter)
        h_layout.addWidget(self.title_label)

        self.info_label = QLabel("V25.0 High-Precision Mode")
        self.info_label.setStyleSheet("color: #00D4DB; font-size: 11px; font-weight: 500; letter-spacing: 0.5px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        h_layout.addWidget(self.info_label)

        main_layout.addWidget(header)

        # 2. Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(15, 10, 15, 20)
        content_layout.setSpacing(20)

        # =====================================================================
        # [V5.0] 배경 필수 안내 영역
        # =====================================================================
        self.bg_required_frame = QFrame()
        self.bg_required_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 82, 82, 0.06);
                border: 1px solid rgba(255, 82, 82, 0.25);
                border-radius: 14px;
                padding: 18px;
            }
        """)
        bg_layout = QVBoxLayout(self.bg_required_frame)
        bg_layout.setSpacing(10)

        bg_title = QLabel("⚠️ 배경 저장 필요")
        bg_title.setStyleSheet("color: #FF5252; font-size: 13px; font-weight: 600; letter-spacing: 0.3px; border: none;")
        bg_layout.addWidget(bg_title)

        bg_desc = QLabel("배경을 저장해야 보정이 활성화됩니다.\n카메라 화면에서 벗어난 채로 배경을 저장해주세요.")
        bg_desc.setStyleSheet("color: rgba(255, 255, 255, 0.5); font-size: 12px; line-height: 1.5; border: none;")
        bg_desc.setWordWrap(True)
        bg_layout.addWidget(bg_desc)

        self.btn_capture_bg = QPushButton("📷 배경 저장하기 (단축키: B)")
        self.btn_capture_bg.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00D4DB, stop:1 #7B61FF);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 14px;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00E5EC, stop:1 #8B71FF);
            }
            QPushButton:pressed {
                background: #00BEC7;
            }
        """)
        self.btn_capture_bg.setCursor(Qt.PointingHandCursor)
        self.btn_capture_bg.clicked.connect(lambda: self.bgCaptureRequested.emit())
        bg_layout.addWidget(self.btn_capture_bg)

        content_layout.addWidget(self.bg_required_frame)

        # =====================================================================
        # Group 1: Face Shape (기존 유지)
        # =====================================================================
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

        self.slider_head = ModernSlider("머리 크기", 0.0)
        self.slider_head.setVisible(False)
        self.slider_head.valueChanged.connect(lambda v: self._update_param('head_scale', v))
        f_inner.addWidget(self.slider_head)

        face_group.setLayout(f_inner)
        self.face_group = face_group  # [V5.0] 참조 저장
        content_layout.addWidget(face_group)

        # =====================================================================
        # Group 2: Body Shape (기존 유지)
        # =====================================================================
        body_group = QGroupBox("체형 보정 (Body Shape)")
        b_inner = QVBoxLayout()
        b_inner.setSpacing(15)

        self.slider_waist = ModernSlider("허리 줄이기", 0.0)
        self.slider_waist.valueChanged.connect(lambda v: self._update_param('waist_slim', v))
        b_inner.addWidget(self.slider_waist)

        self.slider_hip = ModernSlider("골반 늘리기", 0.0)
        self.slider_hip.valueChanged.connect(lambda v: self._update_param('hip_widen', v))
        b_inner.addWidget(self.slider_hip)

        self.slider_shoulder = ModernSlider("어깨 보정", 0.0)
        self.slider_shoulder.setVisible(False)
        self.slider_shoulder.valueChanged.connect(lambda v: self._update_param('shoulder_narrow', v))
        b_inner.addWidget(self.slider_shoulder)

        self.slider_ribcage = ModernSlider("흉곽 줄임", 0.0)
        self.slider_ribcage.setVisible(False)
        self.slider_ribcage.valueChanged.connect(lambda v: self._update_param('ribcage_slim', v))
        b_inner.addWidget(self.slider_ribcage)

        body_group.setLayout(b_inner)
        self.body_group = body_group  # [V5.0] 참조 저장
        content_layout.addWidget(body_group)

        # =====================================================================
        # Group 3: Basic Skin (기존 + 개선)
        # =====================================================================
        skin_group = QGroupBox("피부 기본 (Skin Basic)")
        s_inner = QVBoxLayout()
        s_inner.setSpacing(15)

        self.slider_skin = ModernSlider("피부 결 보정", 0.0)
        self.slider_skin.valueChanged.connect(lambda v: self._update_param('skin_smooth', v))
        s_inner.addWidget(self.slider_skin)

        # Teeth Whitening Slider
        self.slider_teeth = ModernSlider("치아 미백", 0.0)
        self.slider_teeth.valueChanged.connect(lambda v: self._update_param('teeth_whiten', v))
        s_inner.addWidget(self.slider_teeth)

        skin_group.setLayout(s_inner)
        content_layout.addWidget(skin_group)

        # =====================================================================
        # Group 4: Color Grading
        # =====================================================================
        color_group = QGroupBox("색상 조정 (Color Grading)")
        c_inner = QVBoxLayout()
        c_inner.setSpacing(15)

        # 색온도 (Temperature)
        self.slider_temp = ModernSlider("색온도 (Cool ↔ Warm)", 0.5)
        self.slider_temp.valueChanged.connect(self._update_temperature)
        c_inner.addWidget(self.slider_temp)

        color_group.setLayout(c_inner)
        content_layout.addWidget(color_group)

        # =====================================================================
        # Group 5: Settings
        # =====================================================================
        debug_group = QGroupBox("설정 (Settings)")
        d_inner = QVBoxLayout()

        # 디버그 표시
        self.chk_body_debug = QCheckBox("AI 관절 / 마스크 보기")
        self.chk_body_debug.toggled.connect(lambda v: self._update_param('show_body_debug', v))
        d_inner.addWidget(self.chk_body_debug)

        debug_group.setLayout(d_inner)
        content_layout.addWidget(debug_group)

        content_layout.addStretch()
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        self.setLayout(main_layout)

        # [V5.0] 초기 상태: 배경 없음 → 워핑 비활성화
        self.set_background_status(False)

    # =========================================================================
    # Parameter Update Methods
    # =========================================================================
    def _update_param(self, key, value):
        self.current_params[key] = value
        self.paramChanged.emit(self.current_params)

    def _update_temperature(self, value):
        """UI 0.0~1.0 -> Logic -1.0~1.0"""
        temp_val = (value - 0.5) * 2.0
        self._update_param('color_temperature', temp_val)

    # =========================================================================
    # External Control
    # =========================================================================
    def set_profile_info(self, profile_name):
        self.info_label.setText(f"프로파일: {profile_name.upper()} | V25.0")

    def update_sliders_from_config(self, params):
        """Load config values into sliders"""
        self.blockSignals(True)

        # Block all sliders
        sliders = [
            self.slider_eye, self.slider_chin, self.slider_nose,
            self.slider_head, self.slider_shoulder, self.slider_ribcage,
            self.slider_waist, self.slider_hip, self.slider_skin,
            self.slider_teeth, self.slider_temp
        ]
        for s in sliders:
            s.blockSignals(True)

        self.chk_body_debug.blockSignals(True)

        # ---- Basic Params ----
        if 'eye_scale' in params:
            self.slider_eye.set_value(params['eye_scale'])
        if 'face_v' in params:
            self.slider_chin.set_value(params['face_v'])
        if 'nose_slim' in params:
            self.slider_nose.set_value(params['nose_slim'])
        if 'head_scale' in params:
            self.slider_head.set_value(params['head_scale'])

        if 'waist_slim' in params:
            self.slider_waist.set_value(params['waist_slim'])
        if 'hip_widen' in params:
            self.slider_hip.set_value(params['hip_widen'])
        if 'shoulder_narrow' in params:
            self.slider_shoulder.set_value(params['shoulder_narrow'])
        if 'ribcage_slim' in params:
            self.slider_ribcage.set_value(params['ribcage_slim'])

        if 'skin_smooth' in params:
            self.slider_skin.set_value(params['skin_smooth'])

        if 'teeth_whiten' in params:
            self.slider_teeth.set_value(params['teeth_whiten'])

        # ---- Color Grading ----
        if 'color_temperature' in params:
            # Logic -1.0~1.0 -> UI 0.0~1.0
            ui_val = (params['color_temperature'] / 2.0) + 0.5
            self.slider_temp.set_value(ui_val)

        # ---- Checkboxes ----
        if 'show_body_debug' in params:
            self.chk_body_debug.setChecked(bool(params['show_body_debug']))

        # Update internal state
        self.current_params.update(params)

        # Unblock all
        for s in sliders:
            s.blockSignals(False)
        self.chk_body_debug.blockSignals(False)
        self.blockSignals(False)

        # Emit updated params
        self.paramChanged.emit(self.current_params)

    def get_current_params(self):
        """Return current parameter dictionary"""
        return self.current_params.copy()

    def set_background_status(self, has_bg: bool):
        """
        [V5.0] 배경 상태에 따라 UI 활성화/비활성화

        Args:
            has_bg: True면 배경 있음 (워핑 활성화), False면 없음 (비활성화)
        """
        self.has_background = has_bg

        # 안내 영역 표시/숨김
        self.bg_required_frame.setVisible(not has_bg)

        # 워핑 관련 그룹 활성화/비활성화
        self.face_group.setEnabled(has_bg)
        self.body_group.setEnabled(has_bg)

        # 비활성화 시 슬라이더 값 0으로 리셋
        if not has_bg:
            self.slider_chin.set_value(0)
            self.slider_eye.set_value(0)
            self.slider_nose.set_value(0)
            self.slider_waist.set_value(0)
            self.slider_hip.set_value(0)
            self.slider_shoulder.set_value(0)
            self.slider_ribcage.set_value(0)
            self.current_params.update({
                'face_v': 0, 'eye_scale': 0, 'nose_slim': 0,
                'head_scale': 0, 'waist_slim': 0, 'hip_widen': 0,
                'shoulder_narrow': 0, 'ribcage_slim': 0
            })
