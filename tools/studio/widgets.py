# Project MUSE - widgets.py
# Reusable Dialogs and Widgets

import ctypes
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit
)
from PySide6.QtCore import Qt

class ProfileActionDialog(QDialog):
    """
    [Custom Dialog] 기존 프로파일 선택 시 작업 유형 선택
    """
    def __init__(self, profile_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("작업 유형 선택")
        self.resize(500, 350)
        self.setStyleSheet("""
            QDialog {
                background-color: #0A0A0A;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QPushButton {
                border-radius: 12px;
                padding: 16px;
                font-weight: 600;
                font-size: 15px;
                border: none;
            }
        """)
        
        # Win32 Dark Mode for Dialog
        self._apply_dark_title_bar()
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 안내 문구
        lbl_title = QLabel(f"프로파일 [{profile_name}]")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-size: 20px; font-weight: 600; color: #00D4DB; margin-bottom: 8px; letter-spacing: 0.5px;")
        layout.addWidget(lbl_title)
        
        lbl_desc = QLabel("어떤 작업을 진행하시겠습니까?")
        lbl_desc.setAlignment(Qt.AlignCenter)
        lbl_desc.setStyleSheet("color: rgba(255, 255, 255, 0.5); margin-bottom: 24px;")
        layout.addWidget(lbl_desc)
        
        # 버튼 1: Append
        self.btn_append = QPushButton("이어서 학습 (Append)\n[데이터 추가 + 유지]")
        self.btn_append.setCursor(Qt.PointingHandCursor)
        self.btn_append.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2196F3, stop:1 #42A5F5); color: white;")
        
        # 버튼 2: Reset
        self.btn_reset = QPushButton("처음부터 다시 (Reset)\n[기존 데이터 삭제]")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E53935, stop:1 #F44336); color: white;")
        
        # 버튼 3: Cancel
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setStyleSheet("background-color: rgba(255, 255, 255, 0.06); color: rgba(255, 255, 255, 0.6);")
        
        layout.addWidget(self.btn_append)
        layout.addWidget(self.btn_reset)
        layout.addSpacing(10)
        layout.addWidget(self.btn_cancel)
        
        self.btn_append.clicked.connect(lambda: self.done(1))
        self.btn_reset.clicked.connect(lambda: self.done(2))
        self.btn_cancel.clicked.connect(lambda: self.done(0))

    def _apply_dark_title_bar(self):
        """다이얼로그에도 다크바 적용"""
        try:
            hwnd = int(self.winId())
            # DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)), 4)
        except: pass

class NewProfileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("새 프로파일")
        self.resize(400, 180)
        self.setStyleSheet("""
            QDialog {
                background-color: #0A0A0A;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QLineEdit {
                padding: 12px;
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                background: rgba(255, 255, 255, 0.04);
                color: white;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #00D4DB;
            }
            QPushButton {
                padding: 12px;
                border-radius: 10px;
                font-weight: 600;
            }
        """)
        
        # Win32 Dark Mode
        self._apply_dark_title_bar()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)
        
        layout.addWidget(QLabel("새 프로파일 이름 입력 (예: side_cam)"))
        
        self.input_name = QLineEdit()
        layout.addWidget(self.input_name)
        
        btn_box = QHBoxLayout()
        btn_ok = QPushButton("확인")
        btn_ok.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00D4DB, stop:1 #7B61FF); color: white; border: none;")
        btn_ok.clicked.connect(self.accept)
        
        btn_cancel = QPushButton("취소")
        btn_cancel.setStyleSheet("background-color: rgba(255, 255, 255, 0.06); color: rgba(255, 255, 255, 0.6); border: none;")
        btn_cancel.clicked.connect(self.reject)
        
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

    def get_name(self):
        return self.input_name.text().strip()

    def _apply_dark_title_bar(self):
        try:
            hwnd = int(self.winId())
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)), 4)
        except: pass