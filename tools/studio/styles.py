# Project MUSE - styles.py
# Shared Stylesheet for Studio UI

STYLESHEET = """
    QMainWindow {
        background-color: #0A0A0A;
        font-family: Pretendard, Malgun Gothic, sans-serif;
    }
    QLabel {
        color: #D0D0D0;
        font-family: Pretendard, sans-serif;
    }
    QLabel#Title {
        font-size: 26px;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: 2px;
        padding: 12px;
    }
    QLabel#Subtitle {
        font-size: 13px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.5);
        letter-spacing: 0.5px;
        margin-bottom: 24px;
    }
    QPushButton.primary {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00D4DB, stop:1 #7B61FF);
        color: white;
        font-size: 15px;
        font-weight: 600;
        border-radius: 12px;
        padding: 16px;
        border: none;
        letter-spacing: 0.5px;
    }
    QPushButton.primary:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00E5EC, stop:1 #8B71FF);
    }
    QPushButton.primary:pressed {
        background: #00979E;
    }
    QPushButton.primary:disabled {
        background: #1A1A1A;
        color: #444444;
    }
    QPushButton.card {
        background-color: rgba(255, 255, 255, 0.02);
        color: #E0E0E0;
        font-size: 15px;
        font-weight: 500;
        text-align: left;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
    QPushButton.card:hover {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(0, 212, 219, 0.3);
    }
    QComboBox, QLineEdit {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 12px;
        color: white;
        font-size: 14px;
        font-weight: 500;
    }
    QComboBox:focus, QLineEdit:focus {
        border: 1px solid #00D4DB;
    }
    QComboBox::drop-down {
        border: none;
    }
    QProgressBar {
        border: none;
        background-color: rgba(255, 255, 255, 0.04);
        border-radius: 6px;
        height: 10px;
        text-align: center;
        color: transparent;
    }
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00D4DB, stop:1 #7B61FF);
        border-radius: 6px;
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
        background: rgba(255, 255, 255, 0.08);
        min-height: 30px;
        border-radius: 3px;
    }
    QScrollBar::handle:vertical:hover {
        background: #00D4DB;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QFrame {
        border: none;
    }
"""