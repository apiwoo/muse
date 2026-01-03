# Project MUSE - sliders.py
# Created for Mode A (Visual Supremacy)
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PySide6.QtCore import Qt, Signal

class ModernSlider(QWidget):
    """
    [UI Component] Modern Slider with Value Indicator
    Style: Minimal, Teal Accent
    """
    valueChanged = Signal(float)

    def __init__(self, label_text="Param", initial_value=0.0):
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 1. Top Row (Label + Value)
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label_text)
        self.label.setStyleSheet("color: rgba(255, 255, 255, 0.55); font-weight: 500; font-size: 12px; letter-spacing: 0.3px;")
        top_row.addWidget(self.label)

        top_row.addStretch()

        self.value_label = QLabel(f"{initial_value:.2f}")
        self.value_label.setFixedWidth(50)
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("color: #00D4DB; font-weight: 600; font-family: Consolas, D2Coding, monospace; font-size: 11px; background: rgba(0, 212, 219, 0.08); padding: 4px 10px; border-radius: 6px;")
        top_row.addWidget(self.value_label)

        layout.addLayout(top_row)

        # 2. Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(initial_value * 100))

        # Stylesheet for custom slider look
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: none;
                height: 5px;
                background: rgba(255, 255, 255, 0.06);
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00D4DB, stop:1 #7B61FF);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: none;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #00D4DB;
            }
            QSlider::handle:horizontal:pressed {
                background: #00BEC7;
            }
        """)

        self.slider.valueChanged.connect(self._on_slider_change)
        layout.addWidget(self.slider)

        self.setLayout(layout)

    def _on_slider_change(self, val):
        float_val = val / 100.0
        self.value_label.setText(f"{float_val:.2f}")
        self.valueChanged.emit(float_val)

    def value(self):
        return self.slider.value() / 100.0

    def set_value(self, float_val):
        int_val = int(max(0.0, min(1.0, float_val)) * 100)
        self.slider.setValue(int_val)
        self.value_label.setText(f"{float_val:.2f}")