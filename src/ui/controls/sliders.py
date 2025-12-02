# Project MUSE - sliders.py
# Created for Mode A (Visual Supremacy)
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider
from PySide6.QtCore import Qt, Signal

class ModernSlider(QWidget):
    """
    [UI Component] 라벨, 슬라이더, 값 표시가 통합된 모던 슬라이더
    - 범위: 0.0 ~ 1.0 (float)
    - 내부적으로는 0 ~ 100 (int) 단계를 사용
    """
    valueChanged = Signal(float)  # 값이 변경될 때 float 값을 방출

    def __init__(self, label_text="Param", initial_value=0.0):
        super().__init__()
        
        # 레이아웃 설정
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # 1. 라벨 (이름)
        self.label = QLabel(label_text)
        self.label.setFixedWidth(80) # 고정 너비로 정렬 맞춤
        self.label.setStyleSheet("color: #E0E0E0; font-weight: bold;")

        # 2. 슬라이더
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100) # 0% ~ 100%
        self.slider.setValue(int(initial_value * 100))
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #3A3A3A;
                height: 8px;
                background: #2A2A2A;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00ADB5;
                border: 1px solid #00ADB5;
                width: 18px;
                height: 18px;
                margin: -7px 0;
                border-radius: 9px;
            }
        """)

        # 3. 값 표시 (수치)
        self.value_label = QLabel(f"{initial_value:.2f}")
        self.value_label.setFixedWidth(40)
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("color: #00ADB5; font-weight: bold;")

        # 이벤트 연결
        self.slider.valueChanged.connect(self._on_slider_change)

        # 위젯 배치
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)
        
        self.setLayout(layout)

    def _on_slider_change(self, val):
        float_val = val / 100.0
        self.value_label.setText(f"{float_val:.2f}")
        self.valueChanged.emit(float_val)

    def value(self):
        return self.slider.value() / 100.0

    def set_value(self, float_val):
        """[New] 외부에서 값을 강제로 변경 (라벨 업데이트 포함)"""
        # 시그널 루프 방지는 부모(Panel)에서 처리하거나 여기서 blockSignals 사용
        # 여기서는 단순히 값만 바꿈
        int_val = int(max(0.0, min(1.0, float_val)) * 100)
        self.slider.setValue(int_val)
        self.value_label.setText(f"{float_val:.2f}")