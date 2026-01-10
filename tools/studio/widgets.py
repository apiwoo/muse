# Project MUSE - widgets.py
# Reusable Dialogs and Widgets

import sys
import os
import ctypes

# src 경로 추가 (titlebar, frameless_base 사용)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QWidget
)
from PySide6.QtCore import Qt

from ui.titlebar import DialogTitleBar
from ui.frameless_base import FramelessMixin

class ProfileActionDialog(FramelessMixin, QDialog):
    """
    [Custom Dialog] 기존 프로파일 선택 시 작업 유형 선택
    """
    def __init__(self, profile_name, parent=None):
        super().__init__(parent)

        # [Custom Titlebar] Frameless 다이얼로그 설정
        self.setup_frameless_dialog()

        self.setWindowTitle("작업 유형 선택")
        self.resize(500, 400)
        # [Discord Style] 색상 팔레트
        self.setStyleSheet("""
            QDialog {
                background-color: #313338;
                font-family: 'Inter', 'Pretendard', 'Segoe UI', sans-serif;
            }
            QLabel {
                color: #dbdee1;
                font-size: 14px;
            }
            QPushButton {
                border-radius: 4px;
                padding: 14px;
                font-weight: 600;
                font-size: 14px;
                border: none;
            }
        """)

        # [Custom Titlebar] 전체 컨테이너
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # 1. 커스텀 타이틀바
        self.titlebar = DialogTitleBar(self, title="작업 유형 선택")
        outer_layout.addWidget(self.titlebar)

        # 2. 컨텐츠 영역
        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: #1e1f22;")
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # 안내 문구
        lbl_title = QLabel(f"프로파일: {profile_name}")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-size: 18px; font-weight: 600; color: #dbdee1; margin-bottom: 4px;")
        layout.addWidget(lbl_title)

        lbl_desc = QLabel("어떤 작업을 진행하시겠습니까?")
        lbl_desc.setAlignment(Qt.AlignCenter)
        lbl_desc.setStyleSheet("color: #949ba4; margin-bottom: 16px; font-size: 13px;")
        layout.addWidget(lbl_desc)

        # 버튼 1: Append (Discord Blue)
        self.btn_append = QPushButton("이어서 학습 (Append)\n데이터 추가 + 유지")
        self.btn_append.setCursor(Qt.PointingHandCursor)
        self.btn_append.setStyleSheet("""
            QPushButton {
                background-color: #5865f2;
                border: 1px solid #6875f5;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #4752c4;
            }
        """)

        # 버튼 2: Reset (Discord Red)
        self.btn_reset = QPushButton("처음부터 다시 (Reset)\n기존 데이터 삭제")
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #da373c;
                border: 1px solid #e5484d;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #c62f33;
            }
        """)

        # 버튼 3: Cancel
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #4e5058;
                border: 1px solid #5c5f66;
                color: #ffffff;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #5c5f66;
            }
        """)

        layout.addWidget(self.btn_append)
        layout.addWidget(self.btn_reset)
        layout.addSpacing(8)
        layout.addWidget(self.btn_cancel)

        self.btn_append.clicked.connect(lambda: self.done(1))
        self.btn_reset.clicked.connect(lambda: self.done(2))
        self.btn_cancel.clicked.connect(lambda: self.done(0))

        # [Custom Titlebar] 컨텐츠 영역을 외부 레이아웃에 추가
        outer_layout.addWidget(content_widget, stretch=1)

    # [Legacy] 아래 메서드는 커스텀 타이틀바로 대체됨 (참고용으로 주석 처리)
    # def _apply_dark_title_bar(self):
    #     """다이얼로그에도 다크바 적용"""
    #     try:
    #         hwnd = int(self.winId())
    #         ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)), 4)
    #     except: pass


class NewProfileDialog(FramelessMixin, QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # [Custom Titlebar] Frameless 다이얼로그 설정
        self.setup_frameless_dialog()

        self.setWindowTitle("프로필 만들기")
        self.resize(400, 220)
        # [Discord Style] 색상 팔레트
        self.setStyleSheet("""
            QDialog {
                background-color: #313338;
                font-family: 'Inter', 'Pretendard', 'Segoe UI', sans-serif;
            }
            QLabel {
                color: #949ba4;
                font-size: 13px;
            }
            QLineEdit {
                padding: 12px 14px;
                border-radius: 6px;
                border: 1px solid #4e5058;
                background: #2b2d31;
                color: #dbdee1;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #5865f2;
                background: #2b2d31;
            }
            QPushButton {
                padding: 10px;
                border-radius: 4px;
                font-weight: 600;
                border: none;
            }
        """)

        # [Custom Titlebar] 전체 컨테이너
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # 1. 커스텀 타이틀바
        self.titlebar = DialogTitleBar(self, title="프로필 만들기")
        outer_layout.addWidget(self.titlebar)

        # 2. 컨텐츠 영역
        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: #1e1f22;")
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        layout.addWidget(QLabel("프로필 이름을 입력하세요"))

        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("예: front, side_cam")
        layout.addWidget(self.input_name)

        btn_box = QHBoxLayout()
        btn_box.setSpacing(8)

        btn_ok = QPushButton("확인")
        btn_ok.setStyleSheet("""
            QPushButton {
                background-color: #5865f2;
                border: 1px solid #6875f5;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #4752c4;
            }
        """)
        btn_ok.clicked.connect(self.accept)

        btn_cancel = QPushButton("취소")
        btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #4e5058;
                border: 1px solid #5c5f66;
                color: #ffffff;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #5c5f66;
            }
        """)
        btn_cancel.clicked.connect(self.reject)

        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

        # [Custom Titlebar] 컨텐츠 영역을 외부 레이아웃에 추가
        outer_layout.addWidget(content_widget, stretch=1)

    def get_name(self):
        return self.input_name.text().strip()

    # [Legacy] 아래 메서드는 커스텀 타이틀바로 대체됨 (참고용으로 주석 처리)
    # def _apply_dark_title_bar(self):
    #     try:
    #         hwnd = int(self.winId())
    #         ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)), 4)
    #     except: pass
