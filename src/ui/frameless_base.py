# Project MUSE - frameless_base.py
# Frameless Window Utilities (Mixin)
# (C) 2025 MUSE Corp. All rights reserved.

import ctypes
from ctypes import wintypes
from PySide6.QtCore import Qt


class FramelessMixin:
    """
    Frameless 윈도우 공용 기능 믹스인
    QMainWindow 또는 QDialog와 함께 사용합니다.

    사용법:
        class MyWindow(FramelessMixin, QMainWindow):
            def __init__(self):
                super().__init__()
                self.setup_frameless()  # or setup_frameless_dialog() for QDialog
    """

    def setup_frameless(self):
        """Frameless 윈도우 설정 - QMainWindow용, __init__에서 호출"""
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowMinMaxButtonsHint)
        self._apply_window_shadow()

    def setup_frameless_dialog(self):
        """Frameless 다이얼로그 설정 - QDialog용, __init__에서 호출"""
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self._apply_window_shadow()

    def _apply_window_shadow(self):
        """Win32 API로 윈도우 그림자 효과 적용"""
        try:
            hwnd = int(self.winId())

            class MARGINS(ctypes.Structure):
                _fields_ = [
                    ("cxLeftWidth", ctypes.c_int),
                    ("cxRightWidth", ctypes.c_int),
                    ("cyTopHeight", ctypes.c_int),
                    ("cyBottomHeight", ctypes.c_int),
                ]

            margins = MARGINS(1, 1, 1, 1)
            ctypes.windll.dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(margins))
        except Exception as e:
            print(f"[UI] Window shadow failed: {e}")

    def nativeEvent(self, eventType, message):
        """Win32 네이티브 이벤트 처리 - 윈도우 리사이즈"""
        if eventType == b"windows_generic_MSG":
            try:
                msg = wintypes.MSG.from_address(int(message))

                # WM_NCHITTEST = 0x0084
                if msg.message == 0x0084:
                    BORDER_WIDTH = 6  # Discord 스타일 (더 얇은 테두리)
                    x = msg.lParam & 0xFFFF
                    y = (msg.lParam >> 16) & 0xFFFF

                    # 부호 처리 (음수 좌표)
                    if x > 32767:
                        x -= 65536
                    if y > 32767:
                        y -= 65536

                    rect = self.geometry()

                    # 윈도우 좌표로 변환
                    x -= rect.x()
                    y -= rect.y()
                    w = rect.width()
                    h = rect.height()

                    # 가장자리 판정
                    left = x < BORDER_WIDTH
                    right = x > w - BORDER_WIDTH
                    top = y < BORDER_WIDTH
                    bottom = y > h - BORDER_WIDTH

                    # HTLEFT=10, HTRIGHT=11, HTTOP=12, HTBOTTOM=15
                    # HTTOPLEFT=13, HTTOPRIGHT=14, HTBOTTOMLEFT=16, HTBOTTOMRIGHT=17
                    if top and left:
                        return True, 13  # HTTOPLEFT
                    if top and right:
                        return True, 14  # HTTOPRIGHT
                    if bottom and left:
                        return True, 16  # HTBOTTOMLEFT
                    if bottom and right:
                        return True, 17  # HTBOTTOMRIGHT
                    if left:
                        return True, 10  # HTLEFT
                    if right:
                        return True, 11  # HTRIGHT
                    if top:
                        return True, 12  # HTTOP
                    if bottom:
                        return True, 15  # HTBOTTOM
            except Exception:
                pass

        return super().nativeEvent(eventType, message)
