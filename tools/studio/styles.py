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
        font-size: 22px;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: 0.02em;
        padding: 8px;
    }
    QLabel#Subtitle {
        font-size: 13px;
        font-weight: 500;
        color: #949ba4;
        letter-spacing: 0.01em;
        margin-bottom: 16px;
    }
    QPushButton {
        background-color: #4e5058;
        border: 1px solid #5c5f66;
        color: #ffffff;
        font-size: 13px;
        font-weight: 600;
        border-radius: 6px;
        padding: 10px 16px;
    }
    QPushButton:hover {
        background-color: #5c5f66;
        border-color: #6d6f78;
    }
    QPushButton:pressed {
        background-color: #3f4248;
    }
    QPushButton.primary {
        background-color: #5865f2;
        border: 1px solid #6875f5;
        color: white;
        font-size: 14px;
        font-weight: 600;
        border-radius: 6px;
        padding: 12px;
    }
    QPushButton.primary:hover {
        background-color: #4752c4;
        border-color: #5865f2;
    }
    QPushButton.primary:pressed {
        background-color: #3c45a5;
    }
    QPushButton.primary:disabled {
        background-color: #4e5058;
        border-color: #5c5f66;
        color: #6d6f78;
    }
    QPushButton.card {
        background-color: #2b2d31;
        border: 1px solid #3f4147;
        color: #dbdee1;
        font-size: 14px;
        font-weight: 500;
        text-align: left;
        padding: 16px;
        border-radius: 8px;
    }
    QPushButton.card:hover {
        background-color: #383a40;
        border-color: #5865f2;
    }
    QComboBox, QLineEdit {
        background-color: #383a40;
        border: none;
        border-radius: 4px;
        padding: 10px;
        color: #dbdee1;
        font-size: 14px;
        font-weight: 400;
    }
    QComboBox:focus, QLineEdit:focus {
        outline: 2px solid #5865f2;
    }
    QComboBox::drop-down {
        border: none;
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
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        margin-top: 16px;
        font-weight: 600;
        color: #5865f2;
        background: #2b2d31;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        font-size: 11px;
        letter-spacing: 0.02em;
    }

    /* Timeline Navigation Bar */
    #TimelineContainer {
        background-color: #1e1f22;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Bottom Navigation Bar */
    #BottomNavBar {
        background-color: #232428;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    #BottomNavBar QPushButton {
        padding: 12px 24px;
        font-size: 14px;
        font-weight: 600;
        border-radius: 8px;
    }

    #BottomNavBar QPushButton#BtnPrev {
        background-color: #4e5058;
        color: #dbdee1;
    }

    #BottomNavBar QPushButton#BtnPrev:hover {
        background-color: #5c5f66;
    }

    #BottomNavBar QPushButton#BtnPrev:disabled {
        background-color: #383a40;
        color: #6d6f78;
    }

    #BottomNavBar QPushButton#BtnNext {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00D4DB, stop:1 #7B61FF);
        color: white;
    }

    #BottomNavBar QPushButton#BtnNext:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #33E0E6, stop:1 #9580FF);
    }

    #BottomNavBar QPushButton#BtnNext:disabled {
        background: #4e5058;
        color: #6d6f78;
    }

    #BottomNavBar QPushButton#BtnHome {
        background: transparent;
        color: #949ba4;
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
        font-size: 24px;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: 0.01em;
    }

    .StepPage QLabel#StepDescription {
        font-size: 14px;
        font-weight: 400;
        color: #949ba4;
        margin-bottom: 20px;
    }

    .StepPage QLabel#StatusSuccess {
        color: #00D4DB;
        font-weight: 600;
    }

    .StepPage QLabel#StatusWarning {
        color: #f0b232;
        font-weight: 600;
    }

    .StepPage QLabel#StatusError {
        color: #ed4245;
        font-weight: 600;
    }
"""
