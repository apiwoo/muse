# Project MUSE - styles.py
# Shared Stylesheet for Studio UI

STYLESHEET = """
    QMainWindow {
        background-color: #1A1A1A;
    }
    QLabel {
        color: #E0E0E0;
        font-family: 'Segoe UI', sans-serif;
    }
    /* Title Style */
    QLabel#Title {
        font-size: 28px;
        font-weight: bold;
        color: #FFFFFF;
        padding: 10px;
    }
    QLabel#Subtitle {
        font-size: 14px;
        color: #AAAAAA;
        margin-bottom: 20px;
    }
    /* Modern Button (Primary) */
    QPushButton.primary {
        background-color: #00ADB5;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
        border: none;
    }
    QPushButton.primary:hover {
        background-color: #00C4CC;
    }
    QPushButton.primary:pressed {
        background-color: #008C94;
    }
    QPushButton.primary:disabled {
        background-color: #333333;
        color: #666666;
    }
    /* Modern Button (Secondary/Card) */
    QPushButton.card {
        background-color: #2D2D2D;
        color: #EEEEEE;
        font-size: 16px;
        text-align: left;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3A3A3A;
    }
    QPushButton.card:hover {
        background-color: #383838;
        border: 1px solid #00ADB5;
    }
    /* Inputs */
    QComboBox, QLineEdit {
        background-color: #2D2D2D;
        border: 1px solid #444;
        border-radius: 6px;
        padding: 8px;
        color: white;
        font-size: 14px;
    }
    QComboBox::drop-down {
        border: none;
    }
    /* Progress Bar */
    QProgressBar {
        border: none;
        background-color: #2D2D2D;
        border-radius: 8px;
        height: 16px;
        text-align: center;
        color: transparent;
    }
    QProgressBar::chunk {
        background-color: #00ADB5;
        border-radius: 8px;
    }
    /* Scroll Area */
    QScrollArea {
        border: none;
        background-color: transparent;
    }
    QScrollBar:vertical {
        border: none;
        background: #1A1A1A;
        width: 8px;
        margin: 0;
    }
    QScrollBar::handle:vertical {
        background: #444;
        min-height: 20px;
        border-radius: 4px;
    }
"""