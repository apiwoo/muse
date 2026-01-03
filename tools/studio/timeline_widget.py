# Project MUSE - timeline_widget.py
# Horizontal Timeline Widget for Studio Steps
# (C) 2025 MUSE Corp.

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QLinearGradient


class StepIndicator(QWidget):
    """Individual step indicator in the timeline"""
    clicked = Signal(int)  # step index
    settings_clicked = Signal(int)  # step index for settings

    STATE_PENDING = "pending"
    STATE_CURRENT = "current"
    STATE_COMPLETED = "completed"

    def __init__(self, index: int, label: str, parent=None):
        super().__init__(parent)
        self.index = index
        self.label_text = label
        self._state = self.STATE_PENDING
        self._hover = False
        self._gear_hover = False

        self.setFixedSize(100, 70)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self.update()

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self._gear_hover = False
        self.update()

    def mouseMoveEvent(self, event):
        # Check if hovering over gear icon area (top right of circle)
        circle_center = (self.width() // 2, 22)
        gear_pos = (circle_center[0] + 18, circle_center[1] - 12)
        gear_rect = (gear_pos[0] - 10, gear_pos[1] - 10, 20, 20)

        if (gear_rect[0] <= event.position().x() <= gear_rect[0] + gear_rect[2] and
            gear_rect[1] <= event.position().y() <= gear_rect[1] + gear_rect[3]):
            self._gear_hover = True
        else:
            self._gear_hover = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Check if clicking gear icon
            circle_center = (self.width() // 2, 22)
            gear_pos = (circle_center[0] + 18, circle_center[1] - 12)
            gear_rect = (gear_pos[0] - 10, gear_pos[1] - 10, 20, 20)

            if (gear_rect[0] <= event.position().x() <= gear_rect[0] + gear_rect[2] and
                gear_rect[1] <= event.position().y() <= gear_rect[1] + gear_rect[3]):
                self.settings_clicked.emit(self.index)
            else:
                # Only allow clicking completed steps
                if self._state == self.STATE_COMPLETED:
                    self.clicked.emit(self.index)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Colors
        cyan = QColor("#00D4DB")
        purple = QColor("#7B61FF")
        white = QColor("#FFFFFF")
        gray_light = QColor(255, 255, 255, 130)
        gray_dark = QColor(255, 255, 255, 77)
        bg_dark = QColor(255, 255, 255, 25)

        # Circle parameters
        circle_radius = 18
        circle_center_x = self.width() // 2
        circle_center_y = 22

        # Draw circle based on state
        if self._state == self.STATE_COMPLETED:
            # Completed: Solid cyan
            painter.setBrush(QBrush(cyan))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(
                circle_center_x - circle_radius,
                circle_center_y - circle_radius,
                circle_radius * 2,
                circle_radius * 2
            )

            # Draw checkmark
            painter.setPen(QPen(white, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(
                circle_center_x - 7, circle_center_y,
                circle_center_x - 2, circle_center_y + 6
            )
            painter.drawLine(
                circle_center_x - 2, circle_center_y + 6,
                circle_center_x + 8, circle_center_y - 5
            )

        elif self._state == self.STATE_CURRENT:
            # Current: Gradient cyan -> purple
            gradient = QLinearGradient(
                circle_center_x - circle_radius, circle_center_y,
                circle_center_x + circle_radius, circle_center_y
            )
            gradient.setColorAt(0, cyan)
            gradient.setColorAt(1, purple)

            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(
                circle_center_x - circle_radius,
                circle_center_y - circle_radius,
                circle_radius * 2,
                circle_radius * 2
            )

            # Draw step number
            painter.setPen(white)
            font = QFont("Inter", 12, QFont.Bold)
            painter.setFont(font)
            painter.drawText(
                circle_center_x - circle_radius,
                circle_center_y - circle_radius,
                circle_radius * 2,
                circle_radius * 2,
                Qt.AlignCenter,
                str(self.index + 1)
            )

        else:  # PENDING
            # Pending: Semi-transparent with border
            painter.setBrush(QBrush(bg_dark))
            painter.setPen(QPen(gray_dark, 1))
            painter.drawEllipse(
                circle_center_x - circle_radius,
                circle_center_y - circle_radius,
                circle_radius * 2,
                circle_radius * 2
            )

            # Draw step number
            painter.setPen(gray_light)
            font = QFont("Inter", 12, QFont.Bold)
            painter.setFont(font)
            painter.drawText(
                circle_center_x - circle_radius,
                circle_center_y - circle_radius,
                circle_radius * 2,
                circle_radius * 2,
                Qt.AlignCenter,
                str(self.index + 1)
            )

        # Draw label
        if self._state == self.STATE_CURRENT:
            painter.setPen(white)
            font = QFont("Inter", 10, QFont.Bold)
        elif self._state == self.STATE_COMPLETED:
            painter.setPen(cyan)
            font = QFont("Inter", 10, QFont.Normal)
        else:
            painter.setPen(gray_light)
            font = QFont("Inter", 10, QFont.Normal)

        painter.setFont(font)
        painter.drawText(
            0, 48,
            self.width(), 20,
            Qt.AlignCenter,
            self.label_text
        )

        # Draw gear icon (top right of circle)
        gear_x = circle_center_x + 18
        gear_y = circle_center_y - 12
        gear_size = 14

        if self._gear_hover:
            painter.setPen(QPen(white, 1.5))
        else:
            painter.setPen(QPen(QColor(255, 255, 255, 153), 1.5))

        # Simple gear representation (circle with notches)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(gear_x - 5, gear_y - 5, 10, 10)

        # Gear teeth (4 small lines)
        for angle in [0, 90, 180, 270]:
            import math
            rad = math.radians(angle)
            x1 = gear_x + 4 * math.cos(rad)
            y1 = gear_y + 4 * math.sin(rad)
            x2 = gear_x + 7 * math.cos(rad)
            y2 = gear_y + 7 * math.sin(rad)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))


class StudioTimeline(QWidget):
    """Horizontal timeline showing 5 studio steps"""
    step_clicked = Signal(int)
    settings_clicked = Signal(int)

    STEP_LABELS = ["프로필", "카메라", "녹화", "분석", "학습"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(80)
        self.setStyleSheet("background-color: #1e1f22;")

        self._current_step = 0
        self._completed_steps = set()

        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(40, 5, 40, 5)
        layout.setSpacing(0)

        self.indicators = []

        for i, label in enumerate(self.STEP_LABELS):
            # Add step indicator
            indicator = StepIndicator(i, label)
            indicator.clicked.connect(self._on_step_clicked)
            indicator.settings_clicked.connect(self._on_settings_clicked)
            self.indicators.append(indicator)

            layout.addWidget(indicator)

            # Add connecting line (except after last step)
            if i < len(self.STEP_LABELS) - 1:
                line = QWidget()
                line.setFixedHeight(2)
                line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                line.setStyleSheet("background-color: rgba(255, 255, 255, 0.2);")
                line.setObjectName(f"line_{i}")
                layout.addWidget(line)

        # Set initial state
        self._update_states()

    def _on_step_clicked(self, index):
        self.step_clicked.emit(index)

    def _on_settings_clicked(self, index):
        self.settings_clicked.emit(index)

    def set_current_step(self, index: int):
        """Set the currently active step"""
        self._current_step = index
        self._update_states()

    def mark_step_completed(self, index: int):
        """Mark a step as completed"""
        self._completed_steps.add(index)
        self._update_states()

    def reset_step(self, index: int):
        """Reset a step to pending state"""
        self._completed_steps.discard(index)
        self._update_states()

    def reset_all(self):
        """Reset all steps"""
        self._current_step = 0
        self._completed_steps.clear()
        self._update_states()

    def _update_states(self):
        """Update visual states of all indicators"""
        for i, indicator in enumerate(self.indicators):
            if i in self._completed_steps:
                indicator.state = StepIndicator.STATE_COMPLETED
            elif i == self._current_step:
                indicator.state = StepIndicator.STATE_CURRENT
            else:
                indicator.state = StepIndicator.STATE_PENDING

        # Update connecting lines
        for i in range(len(self.STEP_LABELS) - 1):
            line = self.findChild(QWidget, f"line_{i}")
            if line:
                if i < self._current_step or i in self._completed_steps:
                    line.setStyleSheet("background-color: #00D4DB;")
                else:
                    line.setStyleSheet("background-color: rgba(255, 255, 255, 0.2);")
