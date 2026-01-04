"""
TitleBarBridge - Bridge for Title Bar between Python and QML.

This bridge handles window control operations (minimize, maximize, close)
and provides window dragging functionality.
"""
from PySide6.QtCore import QObject, Signal, Property, Slot
from PySide6.QtWidgets import QWidget


class TitleBarBridge(QObject):
    """Bridge for managing window controls from QML."""

    # Signals
    titleChanged = Signal()
    isMaximizedChanged = Signal()

    def __init__(self, window: QWidget, parent=None):
        super().__init__(parent)
        self._window = window
        self._title = "PROJECT MUSE"
        self._drag_start_pos = None

    # Title Property
    @Property(str, notify=titleChanged)
    def title(self):
        return self._title

    @title.setter
    def title(self, value: str):
        if self._title != value:
            self._title = value
            self.titleChanged.emit()

    # Is Maximized Property
    @Property(bool, notify=isMaximizedChanged)
    def isMaximized(self):
        if self._window:
            return self._window.isMaximized()
        return False

    # =========================================================================
    # Window Control Slots
    # =========================================================================

    @Slot()
    def minimize(self):
        """Minimize the window."""
        if self._window:
            self._window.showMinimized()

    @Slot()
    def toggleMaximize(self):
        """Toggle between maximized and normal state."""
        if self._window:
            if self._window.isMaximized():
                self._window.showNormal()
            else:
                self._window.showMaximized()
            self.isMaximizedChanged.emit()

    @Slot()
    def close(self):
        """Close the window."""
        if self._window:
            self._window.close()

    # =========================================================================
    # Drag Handling Slots
    # =========================================================================

    @Slot(float, float)
    def startDrag(self, x: float, y: float):
        """Start window drag from given position."""
        if self._window:
            self._drag_start_pos = self._window.pos()

    @Slot(float, float)
    def moveDrag(self, dx: float, dy: float):
        """Move window by delta from drag start."""
        if self._window and self._drag_start_pos is not None:
            # If maximized, restore first
            if self._window.isMaximized():
                self._window.showNormal()
                self.isMaximizedChanged.emit()
                # Adjust drag start to new position
                self._drag_start_pos = self._window.pos()

            new_x = int(self._drag_start_pos.x() + dx)
            new_y = int(self._drag_start_pos.y() + dy)
            self._window.move(new_x, new_y)

    @Slot()
    def endDrag(self):
        """End window drag."""
        self._drag_start_pos = None

    # =========================================================================
    # Compatibility Methods
    # =========================================================================

    def set_title(self, title: str):
        """Set title (compatibility with TitleBar)."""
        self.title = title

    def update_maximized_state(self):
        """Update maximized state (call after window state change)."""
        self.isMaximizedChanged.emit()
