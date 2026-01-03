# Project MUSE - titlebar.py
# Discord Style Custom Title Bar
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor


class WindowButton(QPushButton):
    """Discord 스타일 윈도우 컨트롤 버튼 (SVG-like 렌더링)"""

    def __init__(self, button_type, parent=None):
        super().__init__(parent)
        self.button_type = button_type  # "minimize", "maximize", "close"
        self.is_maximized = False
        self.setFixedSize(46, 30)
        self.setCursor(Qt.PointingHandCursor)
        self._hovered = False

        self.setStyleSheet("background: transparent; border: none;")

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 호버 배경
        if self._hovered:
            if self.button_type == "close":
                painter.fillRect(self.rect(), QColor("#da373c"))
            else:
                painter.fillRect(self.rect(), QColor(255, 255, 255, 15))

        # 아이콘 색상
        if self._hovered and self.button_type == "close":
            pen = QPen(QColor("#ffffff"))
        else:
            pen = QPen(QColor(255, 255, 255, 200))

        pen.setWidthF(1.0)
        painter.setPen(pen)

        cx = self.width() // 2
        cy = self.height() // 2

        if self.button_type == "minimize":
            # 가로 직선
            painter.drawLine(cx - 5, cy, cx + 5, cy)

        elif self.button_type == "maximize":
            if self.is_maximized:
                # 복원 아이콘 (겹친 사각형)
                painter.drawRect(cx - 3, cy - 5, 7, 7)
                painter.drawRect(cx - 5, cy - 3, 7, 7)
            else:
                # 단일 사각형
                painter.drawRect(cx - 5, cy - 5, 10, 10)

        elif self.button_type == "close":
            # X 표시
            painter.drawLine(cx - 4, cy - 4, cx + 4, cy + 4)
            painter.drawLine(cx + 4, cy - 4, cx - 4, cy + 4)

        painter.end()

    def set_maximized(self, maximized):
        self.is_maximized = maximized
        self.update()


class TitleBar(QWidget):
    """Discord 스타일 타이틀바 (QMainWindow용)"""

    def __init__(self, parent=None, title="PROJECT MUSE"):
        super().__init__(parent)
        self.setFixedHeight(30)
        self._drag_pos = None
        self._title = title
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 0, 0)
        layout.setSpacing(0)

        # 앱 타이틀 (Discord는 좌측에 작게)
        self.lbl_title = QLabel(self._title)
        self.lbl_title.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.7);
                font-size: 12px;
                font-weight: 600;
                font-family: 'Inter', 'Pretendard', sans-serif;
            }
        """)
        layout.addWidget(self.lbl_title)

        # 중앙 빈 공간 (드래그 영역)
        layout.addStretch()

        # 윈도우 컨트롤 버튼
        self.btn_minimize = WindowButton("minimize")
        self.btn_minimize.clicked.connect(self._on_minimize)
        layout.addWidget(self.btn_minimize)

        self.btn_maximize = WindowButton("maximize")
        self.btn_maximize.clicked.connect(self._on_maximize)
        layout.addWidget(self.btn_maximize)

        self.btn_close = WindowButton("close")
        self.btn_close.clicked.connect(self._on_close)
        layout.addWidget(self.btn_close)

        # 타이틀바 스타일
        self.setStyleSheet("""
            QWidget {
                background-color: #313338;
            }
        """)

    def set_title(self, title):
        """타이틀 텍스트 변경"""
        self._title = title
        self.lbl_title.setText(title)

    def _on_minimize(self):
        self.window().showMinimized()

    def _on_maximize(self):
        if self.window().isMaximized():
            self.window().showNormal()
            self.btn_maximize.set_maximized(False)
        else:
            self.window().showMaximized()
            self.btn_maximize.set_maximized(True)

    def _on_close(self):
        self.window().close()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() == Qt.LeftButton:
            window = self.window()
            if window:
                # 최대화 상태에서 드래그 시 복원
                if window.isMaximized():
                    window.showNormal()
                    self._drag_pos = event.globalPosition().toPoint()
                    self.btn_maximize.set_maximized(False)
                else:
                    diff = event.globalPosition().toPoint() - self._drag_pos
                    window.move(window.pos() + diff)
                    self._drag_pos = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._on_maximize()
        super().mouseDoubleClickEvent(event)

    def changeEvent(self, event):
        """윈도우 상태 변경 감지"""
        super().changeEvent(event)
        window = self.window()
        if window:
            self.btn_maximize.set_maximized(window.isMaximized())


class DialogTitleBar(QWidget):
    """Discord 스타일 타이틀바 (QDialog용 - 닫기만)"""

    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        self.setFixedHeight(30)
        self._drag_pos = None
        self._title = title
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 0, 0)
        layout.setSpacing(0)

        self.lbl_title = QLabel(self._title)
        self.lbl_title.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.7);
                font-size: 12px;
                font-weight: 600;
                font-family: 'Inter', 'Pretendard', sans-serif;
            }
        """)
        layout.addWidget(self.lbl_title)

        layout.addStretch()

        self.btn_close = WindowButton("close")
        self.btn_close.clicked.connect(self._on_close)
        layout.addWidget(self.btn_close)

        self.setStyleSheet("""
            QWidget {
                background-color: #313338;
            }
        """)

    def set_title(self, title):
        """타이틀 텍스트 변경"""
        self._title = title
        self.lbl_title.setText(title)

    def _on_close(self):
        self.window().close()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() == Qt.LeftButton:
            window = self.window()
            if window:
                diff = event.globalPosition().toPoint() - self._drag_pos
                window.move(window.pos() + diff)
                self._drag_pos = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)
