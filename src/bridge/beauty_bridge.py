"""
BeautyBridge - Bridge for Beauty Panel parameters between Python and QML.

This bridge provides bi-directional communication for all beauty parameters
and maintains compatibility with the existing BeautyPanel interface.
"""
from PySide6.QtCore import QObject, Signal, Property, Slot


class BeautyBridge(QObject):
    """Bridge for managing beauty parameters between Python and QML."""

    # Parameter change signals
    faceVChanged = Signal()
    eyeScaleChanged = Signal()
    noseSlimChanged = Signal()
    headScaleChanged = Signal()
    shoulderNarrowChanged = Signal()
    ribcageSlimChanged = Signal()
    waistSlimChanged = Signal()
    hipWidenChanged = Signal()
    skinSmoothChanged = Signal()
    teethWhitenChanged = Signal()
    colorTemperatureChanged = Signal()
    showBodyDebugChanged = Signal()

    # State signals
    hasBackgroundChanged = Signal()
    profileNameChanged = Signal()

    # Compatibility signals (same interface as BeautyPanel)
    paramChanged = Signal(dict)
    bgCaptureRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize all parameters
        self._params = {
            'face_v': 0.0,
            'eye_scale': 0.0,
            'nose_slim': 0.0,
            'head_scale': 0.0,
            'shoulder_narrow': 0.0,
            'ribcage_slim': 0.0,
            'waist_slim': 0.0,
            'hip_widen': 0.0,
            'skin_smooth': 0.0,
            'teeth_whiten': 0.0,
            'color_temperature': 0.0,
            'show_body_debug': False,
        }

        self._has_background = False
        self._profile_name = ""
        self._block_signals = False

    # =========================================================================
    # Face Shape Properties
    # =========================================================================

    @Property(float, notify=faceVChanged)
    def faceV(self):
        return self._params['face_v']

    @faceV.setter
    def faceV(self, value: float):
        if self._params['face_v'] != value:
            self._params['face_v'] = value
            self.faceVChanged.emit()
            self._emit_param_changed()

    @Property(float, notify=eyeScaleChanged)
    def eyeScale(self):
        return self._params['eye_scale']

    @eyeScale.setter
    def eyeScale(self, value: float):
        if self._params['eye_scale'] != value:
            self._params['eye_scale'] = value
            self.eyeScaleChanged.emit()
            self._emit_param_changed()

    @Property(float, notify=noseSlimChanged)
    def noseSlim(self):
        return self._params['nose_slim']

    @noseSlim.setter
    def noseSlim(self, value: float):
        if self._params['nose_slim'] != value:
            self._params['nose_slim'] = value
            self.noseSlimChanged.emit()
            self._emit_param_changed()

    @Property(float, notify=headScaleChanged)
    def headScale(self):
        return self._params['head_scale']

    @headScale.setter
    def headScale(self, value: float):
        if self._params['head_scale'] != value:
            self._params['head_scale'] = value
            self.headScaleChanged.emit()
            self._emit_param_changed()

    # =========================================================================
    # Body Shape Properties
    # =========================================================================

    @Property(float, notify=shoulderNarrowChanged)
    def shoulderNarrow(self):
        return self._params['shoulder_narrow']

    @shoulderNarrow.setter
    def shoulderNarrow(self, value: float):
        if self._params['shoulder_narrow'] != value:
            self._params['shoulder_narrow'] = value
            self.shoulderNarrowChanged.emit()
            self._emit_param_changed()

    @Property(float, notify=ribcageSlimChanged)
    def ribcageSlim(self):
        return self._params['ribcage_slim']

    @ribcageSlim.setter
    def ribcageSlim(self, value: float):
        if self._params['ribcage_slim'] != value:
            self._params['ribcage_slim'] = value
            self.ribcageSlimChanged.emit()
            self._emit_param_changed()

    @Property(float, notify=waistSlimChanged)
    def waistSlim(self):
        return self._params['waist_slim']

    @waistSlim.setter
    def waistSlim(self, value: float):
        if self._params['waist_slim'] != value:
            self._params['waist_slim'] = value
            self.waistSlimChanged.emit()
            self._emit_param_changed()

    @Property(float, notify=hipWidenChanged)
    def hipWiden(self):
        return self._params['hip_widen']

    @hipWiden.setter
    def hipWiden(self, value: float):
        if self._params['hip_widen'] != value:
            self._params['hip_widen'] = value
            self.hipWidenChanged.emit()
            self._emit_param_changed()

    # =========================================================================
    # Skin Properties
    # =========================================================================

    @Property(float, notify=skinSmoothChanged)
    def skinSmooth(self):
        return self._params['skin_smooth']

    @skinSmooth.setter
    def skinSmooth(self, value: float):
        if self._params['skin_smooth'] != value:
            self._params['skin_smooth'] = value
            self.skinSmoothChanged.emit()
            self._emit_param_changed()

    @Property(float, notify=teethWhitenChanged)
    def teethWhiten(self):
        return self._params['teeth_whiten']

    @teethWhiten.setter
    def teethWhiten(self, value: float):
        if self._params['teeth_whiten'] != value:
            self._params['teeth_whiten'] = value
            self.teethWhitenChanged.emit()
            self._emit_param_changed()

    # =========================================================================
    # Color Properties
    # =========================================================================

    @Property(float, notify=colorTemperatureChanged)
    def colorTemperature(self):
        """UI value: 0.0 (cool) to 1.0 (warm), center is neutral"""
        # Convert internal -1~1 to UI 0~1
        return (self._params['color_temperature'] / 2.0) + 0.5

    @colorTemperature.setter
    def colorTemperature(self, value: float):
        # Convert UI 0~1 to internal -1~1
        internal_value = (value - 0.5) * 2.0
        if self._params['color_temperature'] != internal_value:
            self._params['color_temperature'] = internal_value
            self.colorTemperatureChanged.emit()
            self._emit_param_changed()

    # =========================================================================
    # Settings Properties
    # =========================================================================

    @Property(bool, notify=showBodyDebugChanged)
    def showBodyDebug(self):
        return self._params['show_body_debug']

    @showBodyDebug.setter
    def showBodyDebug(self, value: bool):
        if self._params['show_body_debug'] != value:
            self._params['show_body_debug'] = value
            self.showBodyDebugChanged.emit()
            self._emit_param_changed()

    # =========================================================================
    # State Properties
    # =========================================================================

    @Property(bool, notify=hasBackgroundChanged)
    def hasBackground(self):
        return self._has_background

    @hasBackground.setter
    def hasBackground(self, value: bool):
        if self._has_background != value:
            self._has_background = value
            self.hasBackgroundChanged.emit()

            # Reset warping params when background is removed
            if not value:
                self._reset_warping_params()

    @Property(str, notify=profileNameChanged)
    def profileName(self):
        return self._profile_name

    @profileName.setter
    def profileName(self, value: str):
        if self._profile_name != value:
            self._profile_name = value
            self.profileNameChanged.emit()

    # =========================================================================
    # Slots (callable from QML)
    # =========================================================================

    @Slot()
    def captureBackground(self):
        """Request background capture."""
        self.bgCaptureRequested.emit()

    @Slot()
    def resetAll(self):
        """Reset all parameters to default values."""
        self._block_signals = True

        for key in self._params:
            if key == 'show_body_debug':
                self._params[key] = False
            else:
                self._params[key] = 0.0

        self._block_signals = False

        # Emit all changed signals
        self.faceVChanged.emit()
        self.eyeScaleChanged.emit()
        self.noseSlimChanged.emit()
        self.headScaleChanged.emit()
        self.shoulderNarrowChanged.emit()
        self.ribcageSlimChanged.emit()
        self.waistSlimChanged.emit()
        self.hipWidenChanged.emit()
        self.skinSmoothChanged.emit()
        self.teethWhitenChanged.emit()
        self.colorTemperatureChanged.emit()
        self.showBodyDebugChanged.emit()

        self._emit_param_changed()

    # =========================================================================
    # Compatibility Methods (match BeautyPanel interface)
    # =========================================================================

    def set_profile_info(self, profile_name: str):
        """Set profile name (compatibility with BeautyPanel)."""
        self.profileName = profile_name

    def update_sliders_from_config(self, params: dict):
        """Load config values into properties (compatibility with BeautyPanel)."""
        self._block_signals = True

        if 'face_v' in params:
            self._params['face_v'] = params['face_v']
            self.faceVChanged.emit()
        if 'eye_scale' in params:
            self._params['eye_scale'] = params['eye_scale']
            self.eyeScaleChanged.emit()
        if 'nose_slim' in params:
            self._params['nose_slim'] = params['nose_slim']
            self.noseSlimChanged.emit()
        if 'head_scale' in params:
            self._params['head_scale'] = params['head_scale']
            self.headScaleChanged.emit()
        if 'shoulder_narrow' in params:
            self._params['shoulder_narrow'] = params['shoulder_narrow']
            self.shoulderNarrowChanged.emit()
        if 'ribcage_slim' in params:
            self._params['ribcage_slim'] = params['ribcage_slim']
            self.ribcageSlimChanged.emit()
        if 'waist_slim' in params:
            self._params['waist_slim'] = params['waist_slim']
            self.waistSlimChanged.emit()
        if 'hip_widen' in params:
            self._params['hip_widen'] = params['hip_widen']
            self.hipWidenChanged.emit()
        if 'skin_smooth' in params:
            self._params['skin_smooth'] = params['skin_smooth']
            self.skinSmoothChanged.emit()
        if 'teeth_whiten' in params:
            self._params['teeth_whiten'] = params['teeth_whiten']
            self.teethWhitenChanged.emit()
        if 'color_temperature' in params:
            self._params['color_temperature'] = params['color_temperature']
            self.colorTemperatureChanged.emit()
        if 'show_body_debug' in params:
            self._params['show_body_debug'] = bool(params['show_body_debug'])
            self.showBodyDebugChanged.emit()

        self._block_signals = False
        self._emit_param_changed()

    def get_current_params(self) -> dict:
        """Return current parameter dictionary (compatibility with BeautyPanel)."""
        return self._params.copy()

    def set_background_status(self, has_bg: bool):
        """Set background status (compatibility with BeautyPanel)."""
        self.hasBackground = has_bg

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _emit_param_changed(self):
        """Emit paramChanged signal with current params."""
        if not self._block_signals:
            self.paramChanged.emit(self._params.copy())

    def _reset_warping_params(self):
        """Reset all warping-related parameters to 0."""
        self._block_signals = True

        warping_keys = [
            'face_v', 'eye_scale', 'nose_slim', 'head_scale',
            'waist_slim', 'hip_widen', 'shoulder_narrow', 'ribcage_slim'
        ]

        for key in warping_keys:
            self._params[key] = 0.0

        self._block_signals = False

        # Emit signals
        self.faceVChanged.emit()
        self.eyeScaleChanged.emit()
        self.noseSlimChanged.emit()
        self.headScaleChanged.emit()
        self.waistSlimChanged.emit()
        self.hipWidenChanged.emit()
        self.shoulderNarrowChanged.emit()
        self.ribcageSlimChanged.emit()

        self._emit_param_changed()
