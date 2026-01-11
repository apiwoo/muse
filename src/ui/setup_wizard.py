# Project MUSE - setup_wizard.py
# First-run setup wizard dialog
# Shows TensorRT engine build progress on first run
# (C) 2025 MUSE Corp. All rights reserved.

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QWidget
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QFont


class BuildWorker(QThread):
    """
    Worker thread for building TensorRT engines.
    Runs the build process in background to keep UI responsive.
    """
    progress = Signal(int, str)  # (percent, message)
    finished = Signal(bool, str)  # (success, error_message)

    def __init__(self, builder):
        """
        Initialize the worker.

        Args:
            builder: FirstRunBuilder instance
        """
        super().__init__()
        self.builder = builder

    def run(self):
        """Execute the engine build process."""
        try:
            # Check GPU availability first
            gpu_ok, gpu_error = self.builder.check_gpu_available()
            if not gpu_ok:
                self.finished.emit(False, gpu_error)
                return

            self.progress.emit(5, "GPU detected. Starting build...")

            # Build all missing engines
            success = self.builder.build_all_missing(progress_callback=self._on_progress)

            if success:
                self.finished.emit(True, "")
            else:
                self.finished.emit(False, "One or more engines failed to build")

        except Exception as e:
            self.finished.emit(False, str(e))

    def _on_progress(self, percent, message):
        """Forward progress to the main thread."""
        self.progress.emit(int(percent), message)


class SetupWizardDialog(QDialog):
    """
    First-run setup wizard dialog.

    Displays progress while building TensorRT engines from ONNX models.
    Shows completion status and allows user to proceed or exit on error.
    """

    def __init__(self, builder, parent=None):
        """
        Initialize the setup wizard dialog.

        Args:
            builder: FirstRunBuilder instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.builder = builder
        self.worker = None
        self.build_success = False
        self.build_started = False

        self._init_ui()
        self._apply_style()

    def _init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("PROJECT MUSE - Initial Setup")
        self.setFixedSize(550, 350)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Title
        self.title_label = QLabel("PROJECT MUSE Initial Setup")
        self.title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)

        # Description
        self.desc_label = QLabel(
            "Optimizing AI models for your graphics card.\n"
            "This process runs only once on first launch."
        )
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)

        layout.addSpacing(10)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Status label (current operation)
        self.status_label = QLabel("Preparing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Warning label
        self.warning_label = QLabel("Estimated time: 5-10 minutes. Please do not close this window.")
        self.warning_label.setAlignment(Qt.AlignCenter)
        self.warning_label.setObjectName("warningLabel")
        layout.addWidget(self.warning_label)

        layout.addStretch()

        # Start button (becomes "Close" or "Start" depending on result)
        self.start_button = QPushButton("Please Wait...")
        self.start_button.setMinimumHeight(45)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._on_button_click)
        layout.addWidget(self.start_button)

    def _apply_style(self):
        """Apply dark theme styling consistent with the main app."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1f22;
                color: #dbdee1;
            }
            QLabel {
                color: #dbdee1;
                font-family: 'Inter', 'Pretendard', 'Segoe UI', sans-serif;
            }
            QLabel#warningLabel {
                color: #FFA500;
                font-size: 12px;
            }
            QProgressBar {
                background-color: #2b2d31;
                border: none;
                border-radius: 4px;
                text-align: center;
                color: #ffffff;
                font-weight: 600;
            }
            QProgressBar::chunk {
                background-color: #5865f2;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #5865f2;
                border: none;
                padding: 12px 24px;
                color: #ffffff;
                border-radius: 4px;
                font-weight: 600;
                font-size: 15px;
                font-family: 'Inter', 'Pretendard', 'Segoe UI', sans-serif;
            }
            QPushButton:hover {
                background-color: #4752c4;
            }
            QPushButton:pressed {
                background-color: #3c45a5;
            }
            QPushButton:disabled {
                background-color: #4e5058;
                color: #949ba4;
            }
        """)

    def showEvent(self, event):
        """Called when the dialog is shown."""
        super().showEvent(event)
        # Start the build process after a short delay
        if not self.build_started:
            self.build_started = True
            QTimer.singleShot(500, self._start_build)

    def _start_build(self):
        """Start the engine build process."""
        self.status_label.setText("Checking GPU availability...")

        self.worker = BuildWorker(self.builder)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, percent, message):
        """Handle progress updates from the worker."""
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_finished(self, success, error):
        """Handle build completion."""
        if success:
            self.build_success = True
            self.progress_bar.setValue(100)
            self.status_label.setText("Optimization Complete!")
            self.status_label.setStyleSheet("color: #23a55a; font-weight: 600;")
            self.warning_label.setText("You can now start using the program.")
            self.warning_label.setStyleSheet("color: #23a55a;")
            self.start_button.setText("Start PROJECT MUSE")
            self.start_button.setEnabled(True)
        else:
            self.build_success = False
            self.status_label.setText(f"Error: {error}")
            self.status_label.setStyleSheet("color: #da373c;")
            self.warning_label.setText("An NVIDIA GPU is required to run this program.")
            self.warning_label.setStyleSheet("color: #da373c;")
            self.start_button.setText("Close")
            self.start_button.setEnabled(True)
            self.start_button.setStyleSheet("""
                QPushButton {
                    background-color: #da373c;
                }
                QPushButton:hover {
                    background-color: #a12d2f;
                }
            """)

    def _on_button_click(self):
        """Handle button click."""
        if self.build_success:
            self.accept()
        else:
            self.reject()

    def closeEvent(self, event):
        """Handle window close event."""
        # Prevent closing while build is in progress
        if self.worker and self.worker.isRunning():
            event.ignore()
        else:
            event.accept()

    def keyPressEvent(self, event):
        """Handle key press events."""
        # Prevent Escape key from closing during build
        if event.key() == Qt.Key_Escape:
            if self.worker and self.worker.isRunning():
                event.ignore()
                return
        super().keyPressEvent(event)


def show_setup_wizard(builder):
    """
    Show the setup wizard dialog and wait for completion.

    Args:
        builder: FirstRunBuilder instance

    Returns:
        bool: True if setup completed successfully, False otherwise
    """
    from PySide6.QtWidgets import QApplication

    # Create application if not exists
    app = QApplication.instance()
    created_app = False
    if app is None:
        import sys
        app = QApplication(sys.argv)
        created_app = True

    dialog = SetupWizardDialog(builder)
    result = dialog.exec()

    if created_app:
        app.quit()

    return dialog.build_success


if __name__ == "__main__":
    # Test the dialog
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from PySide6.QtWidgets import QApplication
    from setup.first_run_builder import FirstRunBuilder

    app = QApplication(sys.argv)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    builder = FirstRunBuilder(project_root)

    dialog = SetupWizardDialog(builder)
    dialog.show()

    sys.exit(app.exec())
