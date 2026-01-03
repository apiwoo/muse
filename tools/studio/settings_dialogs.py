# Project MUSE - settings_dialogs.py
# Advanced Settings Dialogs for Each Studio Step
# (C) 2025 MUSE Corp.

import os
import shutil
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QSlider,
    QGroupBox, QFormLayout, QMessageBox, QWidget
)
from PySide6.QtCore import Qt, Signal


class BaseSettingsDialog(QDialog):
    """Base class for settings dialogs"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(400)
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2d31;
            }
            QLabel {
                color: #dbdee1;
                font-size: 13px;
            }
            QLabel#Title {
                font-size: 16px;
                font-weight: 600;
                color: white;
                padding-bottom: 10px;
            }
            QLabel#Description {
                color: #949ba4;
                font-size: 12px;
                padding-bottom: 15px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #383a40;
                border: 1px solid #4e5058;
                border-radius: 4px;
                padding: 8px;
                color: #dbdee1;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #5865f2;
            }
            QGroupBox {
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: 600;
                color: #5865f2;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QPushButton {
                background-color: #4e5058;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                color: white;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #5c5f66;
            }
            QPushButton#Primary {
                background-color: #5865f2;
            }
            QPushButton#Primary:hover {
                background-color: #4752c4;
            }
            QPushButton#Danger {
                background-color: #ed4245;
            }
            QPushButton#Danger:hover {
                background-color: #c93b3e;
            }
        """)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(24, 24, 24, 24)
        self.main_layout.setSpacing(16)

    def add_title(self, title: str, description: str = ""):
        lbl_title = QLabel(title)
        lbl_title.setObjectName("Title")
        self.main_layout.addWidget(lbl_title)

        if description:
            lbl_desc = QLabel(description)
            lbl_desc.setObjectName("Description")
            lbl_desc.setWordWrap(True)
            self.main_layout.addWidget(lbl_desc)

    def add_buttons(self, save_text="저장", cancel_text="취소"):
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        btn_cancel = QPushButton(cancel_text)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        btn_save = QPushButton(save_text)
        btn_save.setObjectName("Primary")
        btn_save.clicked.connect(self.accept)
        btn_layout.addWidget(btn_save)

        self.main_layout.addLayout(btn_layout)


class Step1SettingsDialog(BaseSettingsDialog):
    """Profile settings dialog"""
    profile_renamed = Signal(str, str)  # old_name, new_name
    profile_deleted = Signal(str)
    data_reset = Signal(str)

    def __init__(self, profile_name: str, profile_path: str, parent=None):
        super().__init__("프로필 설정", parent)
        self.profile_name = profile_name
        self.profile_path = profile_path

        self.add_title(
            f"프로필: {profile_name}",
            "프로필 이름 변경, 삭제 또는 데이터 초기화를 수행할 수 있습니다."
        )

        # Rename section
        grp_rename = QGroupBox("이름 변경")
        rename_layout = QFormLayout(grp_rename)

        self.edit_name = QLineEdit(profile_name)
        rename_layout.addRow("새 이름:", self.edit_name)

        btn_rename = QPushButton("이름 변경")
        btn_rename.clicked.connect(self._on_rename)
        rename_layout.addRow("", btn_rename)

        self.main_layout.addWidget(grp_rename)

        # Danger zone
        grp_danger = QGroupBox("위험 영역")
        danger_layout = QVBoxLayout(grp_danger)

        btn_reset = QPushButton("데이터 초기화")
        btn_reset.setObjectName("Danger")
        btn_reset.clicked.connect(self._on_reset_data)
        danger_layout.addWidget(btn_reset)

        lbl_reset_info = QLabel("녹화된 영상과 분석 데이터를 모두 삭제합니다.")
        lbl_reset_info.setStyleSheet("color: #949ba4; font-size: 11px;")
        danger_layout.addWidget(lbl_reset_info)

        btn_delete = QPushButton("프로필 삭제")
        btn_delete.setObjectName("Danger")
        btn_delete.clicked.connect(self._on_delete)
        danger_layout.addWidget(btn_delete)

        lbl_delete_info = QLabel("프로필과 모든 관련 데이터를 완전히 삭제합니다.")
        lbl_delete_info.setStyleSheet("color: #949ba4; font-size: 11px;")
        danger_layout.addWidget(lbl_delete_info)

        self.main_layout.addWidget(grp_danger)

        self.main_layout.addStretch()

        # Close button
        btn_close = QPushButton("닫기")
        btn_close.clicked.connect(self.accept)
        self.main_layout.addWidget(btn_close)

    def _on_rename(self):
        new_name = self.edit_name.text().strip()
        if not new_name:
            QMessageBox.warning(self, "경고", "이름을 입력하세요.")
            return

        if new_name == self.profile_name:
            return

        reply = QMessageBox.question(
            self, "확인",
            f"프로필 이름을 '{self.profile_name}'에서 '{new_name}'으로 변경하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.profile_renamed.emit(self.profile_name, new_name)
            self.accept()

    def _on_reset_data(self):
        reply = QMessageBox.warning(
            self, "경고",
            f"'{self.profile_name}' 프로필의 모든 데이터를 삭제하시겠습니까?\n"
            "이 작업은 되돌릴 수 없습니다.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.data_reset.emit(self.profile_name)
            self.accept()

    def _on_delete(self):
        reply = QMessageBox.warning(
            self, "경고",
            f"'{self.profile_name}' 프로필을 완전히 삭제하시겠습니까?\n"
            "모든 데이터가 삭제되며 되돌릴 수 없습니다.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.profile_deleted.emit(self.profile_name)
            self.accept()


class Step2SettingsDialog(BaseSettingsDialog):
    """Camera settings dialog"""

    def __init__(self, available_cameras: list, current_camera: int = 0,
                 resolution: tuple = (1920, 1080), fps: int = 30, parent=None):
        super().__init__("카메라 설정", parent)

        self.add_title(
            "카메라 고급 설정",
            "카메라 장치, 해상도 및 프레임레이트를 설정합니다."
        )

        # Camera selection
        grp_camera = QGroupBox("카메라 장치")
        cam_layout = QFormLayout(grp_camera)

        self.combo_camera = QComboBox()
        for idx, name in available_cameras:
            self.combo_camera.addItem(f"[{idx}] {name}", idx)
        if self.combo_camera.findData(current_camera) >= 0:
            self.combo_camera.setCurrentIndex(self.combo_camera.findData(current_camera))
        cam_layout.addRow("카메라:", self.combo_camera)

        self.main_layout.addWidget(grp_camera)

        # Resolution
        grp_res = QGroupBox("해상도")
        res_layout = QFormLayout(grp_res)

        self.combo_resolution = QComboBox()
        self.combo_resolution.addItem("1920x1080 (Full HD)", (1920, 1080))
        self.combo_resolution.addItem("1280x720 (HD)", (1280, 720))
        self.combo_resolution.addItem("640x480 (VGA)", (640, 480))

        # Set current resolution
        for i in range(self.combo_resolution.count()):
            if self.combo_resolution.itemData(i) == resolution:
                self.combo_resolution.setCurrentIndex(i)
                break

        res_layout.addRow("해상도:", self.combo_resolution)

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(15, 60)
        self.spin_fps.setValue(fps)
        res_layout.addRow("FPS:", self.spin_fps)

        self.main_layout.addWidget(grp_res)

        self.main_layout.addStretch()
        self.add_buttons()

    def get_settings(self) -> dict:
        return {
            "camera_index": self.combo_camera.currentData(),
            "resolution": self.combo_resolution.currentData(),
            "fps": self.spin_fps.value()
        }


class Step3SettingsDialog(BaseSettingsDialog):
    """Recording settings dialog"""

    def __init__(self, quality: str = "high", frame_interval: int = 5,
                 append_mode: bool = True, parent=None):
        super().__init__("녹화 설정", parent)

        self.add_title(
            "녹화 고급 설정",
            "녹화 품질과 프레임 추출 간격을 설정합니다."
        )

        # Quality
        grp_quality = QGroupBox("녹화 품질")
        quality_layout = QFormLayout(grp_quality)

        self.combo_quality = QComboBox()
        self.combo_quality.addItem("낮음 (파일 크기 작음)", "low")
        self.combo_quality.addItem("중간", "medium")
        self.combo_quality.addItem("높음 (권장)", "high")

        quality_map = {"low": 0, "medium": 1, "high": 2}
        self.combo_quality.setCurrentIndex(quality_map.get(quality, 2))

        quality_layout.addRow("품질:", self.combo_quality)

        self.main_layout.addWidget(grp_quality)

        # Frame extraction
        grp_frames = QGroupBox("프레임 추출")
        frames_layout = QFormLayout(grp_frames)

        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(1, 30)
        self.spin_interval.setValue(frame_interval)
        self.spin_interval.setSuffix(" 프레임마다")
        frames_layout.addRow("추출 간격:", self.spin_interval)

        lbl_info = QLabel("낮을수록 더 많은 프레임이 추출되어 학습 품질이 향상되지만\n"
                          "분석 시간이 오래 걸립니다.")
        lbl_info.setStyleSheet("color: #949ba4; font-size: 11px;")
        frames_layout.addRow("", lbl_info)

        self.main_layout.addWidget(grp_frames)

        # Data mode
        grp_mode = QGroupBox("데이터 모드")
        mode_layout = QFormLayout(grp_mode)

        self.combo_mode = QComboBox()
        self.combo_mode.addItem("기존 데이터에 추가", True)
        self.combo_mode.addItem("새로 시작 (기존 삭제)", False)
        self.combo_mode.setCurrentIndex(0 if append_mode else 1)

        mode_layout.addRow("모드:", self.combo_mode)

        self.main_layout.addWidget(grp_mode)

        self.main_layout.addStretch()
        self.add_buttons()

    def get_settings(self) -> dict:
        return {
            "quality": self.combo_quality.currentData(),
            "frame_interval": self.spin_interval.value(),
            "append_mode": self.combo_mode.currentData()
        }


class Step4SettingsDialog(BaseSettingsDialog):
    """Analysis settings dialog"""

    def __init__(self, model_version: str = "sam2.1", quality_threshold: float = 0.5,
                 gpu_memory_limit: int = 8, parent=None):
        super().__init__("분석 설정", parent)

        self.add_title(
            "AI 분석 고급 설정",
            "분석 모델과 품질 필터 임계값을 설정합니다."
        )

        # Model selection
        grp_model = QGroupBox("분석 모델")
        model_layout = QFormLayout(grp_model)

        self.combo_model = QComboBox()
        self.combo_model.addItem("SAM 2.0 (빠름)", "sam2.0")
        self.combo_model.addItem("SAM 2.1 (권장)", "sam2.1")

        if model_version == "sam2.0":
            self.combo_model.setCurrentIndex(0)
        else:
            self.combo_model.setCurrentIndex(1)

        model_layout.addRow("모델:", self.combo_model)

        self.main_layout.addWidget(grp_model)

        # Quality filter
        grp_quality = QGroupBox("품질 필터")
        quality_layout = QFormLayout(grp_quality)

        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(0, 100)
        self.slider_threshold.setValue(int(quality_threshold * 100))

        self.lbl_threshold = QLabel(f"{quality_threshold:.2f}")
        self.slider_threshold.valueChanged.connect(
            lambda v: self.lbl_threshold.setText(f"{v / 100:.2f}")
        )

        threshold_widget = QWidget()
        threshold_layout = QHBoxLayout(threshold_widget)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.addWidget(self.slider_threshold)
        threshold_layout.addWidget(self.lbl_threshold)

        quality_layout.addRow("임계값:", threshold_widget)

        lbl_info = QLabel("높을수록 품질이 낮은 프레임이 더 많이 필터링됩니다.")
        lbl_info.setStyleSheet("color: #949ba4; font-size: 11px;")
        quality_layout.addRow("", lbl_info)

        self.main_layout.addWidget(grp_quality)

        # GPU settings
        grp_gpu = QGroupBox("GPU 설정")
        gpu_layout = QFormLayout(grp_gpu)

        self.spin_gpu = QSpinBox()
        self.spin_gpu.setRange(2, 24)
        self.spin_gpu.setValue(gpu_memory_limit)
        self.spin_gpu.setSuffix(" GB")
        gpu_layout.addRow("메모리 제한:", self.spin_gpu)

        self.main_layout.addWidget(grp_gpu)

        self.main_layout.addStretch()
        self.add_buttons()

    def get_settings(self) -> dict:
        return {
            "model_version": self.combo_model.currentData(),
            "quality_threshold": self.slider_threshold.value() / 100,
            "gpu_memory_limit": self.spin_gpu.value()
        }


class Step5SettingsDialog(BaseSettingsDialog):
    """Training settings dialog"""

    def __init__(self, epochs: int = 100, batch_size: int = 4,
                 learning_rate: float = 0.001, track: str = "STUDENT", parent=None):
        super().__init__("학습 설정", parent)

        self.add_title(
            "모델 학습 고급 설정",
            "학습 하이퍼파라미터를 설정합니다."
        )

        # Training track
        grp_track = QGroupBox("학습 트랙")
        track_layout = QFormLayout(grp_track)

        self.combo_track = QComboBox()
        self.combo_track.addItem("Student (전체 최적화, 빠름)", "STUDENT")
        self.combo_track.addItem("LoRA (정밀 보정, 고사양)", "LORA")

        if track == "LORA":
            self.combo_track.setCurrentIndex(1)
        else:
            self.combo_track.setCurrentIndex(0)

        track_layout.addRow("트랙:", self.combo_track)

        self.main_layout.addWidget(grp_track)

        # Hyperparameters
        grp_hyper = QGroupBox("하이퍼파라미터")
        hyper_layout = QFormLayout(grp_hyper)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(10, 500)
        self.spin_epochs.setValue(epochs)
        hyper_layout.addRow("에폭 수:", self.spin_epochs)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 32)
        self.spin_batch.setValue(batch_size)
        hyper_layout.addRow("배치 크기:", self.spin_batch)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 0.1)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setSingleStep(0.0001)
        self.spin_lr.setValue(learning_rate)
        hyper_layout.addRow("학습률:", self.spin_lr)

        self.main_layout.addWidget(grp_hyper)

        self.main_layout.addStretch()
        self.add_buttons()

    def get_settings(self) -> dict:
        return {
            "track": self.combo_track.currentData(),
            "epochs": self.spin_epochs.value(),
            "batch_size": self.spin_batch.value(),
            "learning_rate": self.spin_lr.value()
        }
