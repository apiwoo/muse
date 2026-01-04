"""
AppBridge - Global application state bridge for Python-QML communication.
"""
from PySide6.QtCore import QObject, Signal, Property, Slot


class AppBridge(QObject):
    """Bridge for managing global application state between Python and QML."""

    # Signals
    currentProfileChanged = Signal()
    isRecordingChanged = Signal()
    fpsChanged = Signal()
    statusMessageChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_profile = ""
        self._is_recording = False
        self._fps = 0.0
        self._status_message = ""

    # Current Profile
    @Property(str, notify=currentProfileChanged)
    def currentProfile(self):
        return self._current_profile

    @currentProfile.setter
    def currentProfile(self, value: str):
        if self._current_profile != value:
            self._current_profile = value
            self.currentProfileChanged.emit()

    # Recording State
    @Property(bool, notify=isRecordingChanged)
    def isRecording(self):
        return self._is_recording

    @isRecording.setter
    def isRecording(self, value: bool):
        if self._is_recording != value:
            self._is_recording = value
            self.isRecordingChanged.emit()

    # FPS
    @Property(float, notify=fpsChanged)
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value: float):
        if self._fps != value:
            self._fps = value
            self.fpsChanged.emit()

    # Status Message
    @Property(str, notify=statusMessageChanged)
    def statusMessage(self):
        return self._status_message

    @statusMessage.setter
    def statusMessage(self, value: str):
        if self._status_message != value:
            self._status_message = value
            self.statusMessageChanged.emit()

    @Slot(str)
    def setProfile(self, profile_name: str):
        """Slot for setting current profile from QML."""
        self.currentProfile = profile_name

    @Slot()
    def toggleRecording(self):
        """Slot for toggling recording state from QML."""
        self.isRecording = not self._is_recording
