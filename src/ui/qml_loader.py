"""
QmlLoader - Utility for loading QML files into QQuickWidget.

This module provides a simple interface for loading QML components
and connecting Python bridge objects to the QML context.
"""
import os
from pathlib import Path
from typing import Dict, Optional, Any

from PySide6.QtCore import QUrl, Qt
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QWidget

# Set QML style to Basic for full customization support
# Must be set before any QML engine is created
os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")

# QML root directory (relative to this file)
QML_ROOT = Path(__file__).parent.parent / "qml"


class QmlLoader:
    """Utility class for loading QML files into QQuickWidget."""

    @staticmethod
    def load(
        qml_filename: str,
        bridges: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
        transparent: bool = False
    ) -> QQuickWidget:
        """
        Load a QML file into a QQuickWidget.

        Args:
            qml_filename: Path to QML file relative to QML_ROOT
                         (e.g., "panels/BeautyPanelQml.qml")
            bridges: Dictionary of {name: object} to expose to QML context
            parent: Parent widget
            transparent: If True, make the widget background transparent

        Returns:
            QQuickWidget with the loaded QML content
        """
        widget = QQuickWidget(parent)
        widget.setResizeMode(QQuickWidget.SizeRootObjectToView)

        # Set transparent background if requested
        if transparent:
            widget.setClearColor(Qt.transparent)
            widget.setAttribute(Qt.WA_TranslucentBackground)

        # Add import path for custom QML modules (styles, components)
        engine = widget.engine()
        engine.addImportPath(str(QML_ROOT))

        # Register bridge objects with QML context
        if bridges:
            context = widget.rootContext()
            for name, obj in bridges.items():
                context.setContextProperty(name, obj)

        # Load the QML file
        qml_path = QML_ROOT / qml_filename
        if not qml_path.exists():
            raise FileNotFoundError(f"QML file not found: {qml_path}")

        widget.setSource(QUrl.fromLocalFile(str(qml_path)))

        # Check for errors
        if widget.status() == QQuickWidget.Error:
            errors = widget.errors()
            error_messages = [str(e.toString()) for e in errors]
            raise RuntimeError(f"QML load errors: {error_messages}")

        return widget

    @staticmethod
    def get_qml_path(qml_filename: str) -> Path:
        """Get the absolute path to a QML file."""
        return QML_ROOT / qml_filename

    @staticmethod
    def qml_root() -> Path:
        """Get the QML root directory path."""
        return QML_ROOT
