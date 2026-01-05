# Project MUSE - styles.py
# Shared Stylesheet for Studio UI (Discord Style)

STYLESHEET = """
    QMainWindow {
        background-color: #313338;
        font-family: 'Inter', 'Pretendard', 'Segoe UI', sans-serif;
    }
    QWidget {
        background-color: #1e1f22;
    }
    QLabel {
        color: #dbdee1;
        font-family: 'Inter', 'Pretendard', sans-serif;
        background: transparent;
    }
    QLabel#Title {
        font-size: 20px;
        font-weight: 600;
        color: #f2f3f5;
        letter-spacing: 0.02em;
        padding: 8px;
    }
    QLabel#Subtitle {
        font-size: 14px;
        font-weight: 400;
        color: #949ba4;
        margin-bottom: 16px;
    }
    QPushButton {
        background-color: #4e5058;
        border: none;
        color: #ffffff;
        font-size: 14px;
        font-weight: 500;
        border-radius: 3px;
        padding: 10px 16px;
    }
    QPushButton:hover {
        background-color: #6d6f78;
    }
    QPushButton:pressed {
        background-color: #4e5058;
    }
    QPushButton.primary {
        background-color: #5865f2;
        border: none;
        color: white;
        font-size: 14px;
        font-weight: 500;
        border-radius: 3px;
        padding: 10px 16px;
    }
    QPushButton.primary:hover {
        background-color: #4752c4;
    }
    QPushButton.primary:pressed {
        background-color: #3c45a5;
    }
    QPushButton.primary:disabled {
        background-color: #4e5058;
        color: #72767d;
    }
    QPushButton.card {
        background-color: #2b2d31;
        border: 1px solid #3f4147;
        color: #dbdee1;
        font-size: 14px;
        font-weight: 400;
        text-align: left;
        padding: 16px;
        border-radius: 8px;
    }
    QPushButton.card:hover {
        background-color: #36373d;
        border-color: #5865f2;
    }
    QComboBox, QLineEdit {
        background-color: #1e1f22;
        border: none;
        border-radius: 3px;
        padding: 10px;
        color: #dbdee1;
        font-size: 14px;
        font-weight: 400;
    }
    QComboBox:focus, QLineEdit:focus {
        outline: none;
    }
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    QComboBox QAbstractItemView {
        background-color: #1e1f22;
        border: none;
        selection-background-color: #404249;
    }
    QProgressBar {
        border: none;
        background-color: #4e5058;
        border-radius: 4px;
        height: 8px;
        text-align: center;
        color: transparent;
    }
    QProgressBar::chunk {
        background-color: #5865f2;
        border-radius: 4px;
    }
    QScrollArea {
        border: none;
        background-color: transparent;
    }
    QScrollBar:vertical {
        border: none;
        background: transparent;
        width: 6px;
        margin: 0;
    }
    QScrollBar::handle:vertical {
        background: #4e5058;
        min-height: 30px;
        border-radius: 3px;
    }
    QScrollBar::handle:vertical:hover {
        background: #6d6f78;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QFrame {
        border: none;
    }
    QGroupBox {
        border: 1px solid #3f4147;
        border-radius: 8px;
        margin-top: 16px;
        font-weight: 600;
        color: #949ba4;
        background: #2b2d31;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        font-size: 10px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Timeline Navigation Bar */
    #TimelineContainer {
        background-color: #1e1f22;
        border-bottom: 1px solid #3f4147;
    }

    /* Bottom Navigation Bar */
    #BottomNavBar {
        background-color: #2b2d31;
        border-top: 1px solid #3f4147;
    }

    #BottomNavBar QPushButton {
        padding: 12px 24px;
        font-size: 14px;
        font-weight: 500;
        border-radius: 3px;
    }

    #BottomNavBar QPushButton#BtnPrev {
        background-color: #4e5058;
        color: #dbdee1;
    }

    #BottomNavBar QPushButton#BtnPrev:hover {
        background-color: #6d6f78;
    }

    #BottomNavBar QPushButton#BtnPrev:disabled {
        background-color: #36373d;
        color: #72767d;
    }

    #BottomNavBar QPushButton#BtnNext {
        background-color: #5865f2;
        color: white;
    }

    #BottomNavBar QPushButton#BtnNext:hover {
        background-color: #4752c4;
    }

    #BottomNavBar QPushButton#BtnNext:disabled {
        background-color: #4e5058;
        color: #72767d;
    }

    #BottomNavBar QPushButton#BtnHome {
        background: transparent;
        color: #b5bac1;
        border: none;
    }

    #BottomNavBar QPushButton#BtnHome:hover {
        color: #dbdee1;
    }

    /* Step Page Common Styles */
    .StepPage {
        background-color: #1e1f22;
    }

    .StepPage QLabel#StepTitle {
        font-size: 20px;
        font-weight: 600;
        color: #f2f3f5;
        letter-spacing: 0.01em;
    }

    .StepPage QLabel#StepDescription {
        font-size: 14px;
        font-weight: 400;
        color: #949ba4;
        margin-bottom: 20px;
    }

    .StepPage QLabel#StatusSuccess {
        color: #23a55a;
        font-weight: 500;
    }

    .StepPage QLabel#StatusWarning {
        color: #f0b232;
        font-weight: 500;
    }

    .StepPage QLabel#StatusError {
        color: #da373c;
        font-weight: 500;
    }

    /* Tooltip */
    QToolTip {
        background-color: #111214;
        color: #dbdee1;
        border: none;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 14px;
    }
"""
