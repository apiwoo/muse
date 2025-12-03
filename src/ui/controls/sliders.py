# Project MUSE - sliders.py
# Created for Mode A (Visual Supremacy)
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider
from PySide6.QtCore import Qt, Signal

class ModernSlider(QWidget):
    """
    [UI Component] Modern Slider with Value Indicator
    Style: Minimal, Teal Accent
    """
    valueChanged = Signal(float)

    def __init__(self, label_text="Param", initial_value=0.0):
        super().__init__()
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        # 1. Label
        self.label = QLabel(label_text)
        self.label.setFixedWidth(90)
        self.label.setStyleSheet("color: #AAAAAA; font-weight: 600; font-size: 11px;")

        # 2. Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(initial_value * 100))
        
        # Stylesheet for custom slider look
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: none;
                height: 4px;
                background: #333333;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #00ADB5; /* Teal fill */
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #FFFFFF;
                border: 2px solid #00ADB5;
                width: 14px;
                height: 14px;
                margin: -5px 0; /* Center handle */
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #00ADB5;
                border-color: white;
            }
        """)

        # 3. Value
        self.value_label = QLabel(f"{initial_value:.2f}")
        self.value_label.setFixedWidth(35)
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("color: #00ADB5; font-weight: bold; font-family: monospace;")

        self.slider.valueChanged.connect(self._on_slider_change)

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
        int_val = int(max(0.0, min(1.0, float_val)) * 100)
        self.slider.setValue(int_val)
        self.value_label.setText(f"{float_val:.2f}")