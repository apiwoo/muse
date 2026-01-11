# -*- mode: python ; coding: utf-8 -*-
# Project MUSE - PyInstaller Spec File
# Creates MUSE.exe distribution package
# (C) 2025 MUSE Corp. All rights reserved.

"""
PyInstaller Build Configuration for PROJECT MUSE

This spec file configures how PyInstaller bundles the application.
The resulting package includes:
- MUSE.exe (main executable)
- _internal/ (Python runtime and packages)
- libs/ (CUDA/TensorRT DLLs - copied separately)
- assets/ (models, shaders - copied separately)
- src/ (Python source code)

Build command:
    pyinstaller muse.spec --noconfirm

Or use:
    python tools/build_distribution.py
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Project root directory (where this spec file is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(SPEC))

# Hidden imports - modules that PyInstaller can't detect automatically
hidden_imports = [
    # PySide6 modules
    'PySide6.QtCore',
    'PySide6.QtWidgets',
    'PySide6.QtGui',
    'PySide6.QtOpenGL',
    'PySide6.QtOpenGLWidgets',

    # Deep learning frameworks
    'torch',
    'torch.cuda',
    'torchvision',
    'torchvision.transforms',
    'torchvision.models',

    # TensorRT
    'tensorrt',

    # Computer vision
    'cv2',
    'mediapipe',

    # Scientific computing
    'numpy',
    'cupy',
    'cupyx',
    'cupyx.scipy',
    'cupyx.scipy.ndimage',
    'fastrlock',
    'scipy',
    'scipy.ndimage',

    # ONNX
    'onnx',
    'onnxruntime',

    # UI/Theme
    'qdarktheme',

    # Virtual camera
    'pyvirtualcam',

    # Progress bar
    'tqdm',

    # Our own modules
    'setup',
    'setup.first_run_builder',
    'ui',
    'ui.setup_wizard',
    'utils',
    'utils.cuda_helper',
    'graphics',
    'core',
    'ai',
    'studio',
    'bridge',
]

# Collect all submodules for packages that have many internal imports
hidden_imports += collect_submodules('PySide6')
hidden_imports += collect_submodules('torch')
hidden_imports += collect_submodules('cv2')
hidden_imports += collect_submodules('cupy')
hidden_imports += collect_submodules('cupy_backends')
hidden_imports += collect_submodules('cupyx')
hidden_imports += collect_submodules('fastrlock')
hidden_imports += collect_submodules('scipy')
hidden_imports += collect_submodules('mediapipe')
hidden_imports += collect_submodules('qdarktheme')
hidden_imports += collect_submodules('tqdm')

# Data files to include
datas = [
    # Include src folder (Python source code)
    (os.path.join(PROJECT_ROOT, 'src'), 'src'),

    # Include tools folder (runtime scripts)
    (os.path.join(PROJECT_ROOT, 'tools', 'studio'), os.path.join('tools', 'studio')),
]

# Collect PySide6 data files (plugins, translations, etc.)
datas += collect_data_files('PySide6')

# Collect qdarktheme data files (theme stylesheets)
datas += collect_data_files('qdarktheme')

# Collect mediapipe data files (models)
datas += collect_data_files('mediapipe')

# Collect CuPy data files (include headers for JIT compilation)
datas += collect_data_files('cupy')

# Binary files (none explicitly - DLLs are in libs folder which is copied separately)
binaries = []

# Packages/modules to exclude (reduce size) - be conservative!
excludes = [
    # GUI frameworks we don't use
    'tkinter',
    '_tkinter',

    # Exclude PyQt to avoid conflict with PySide6
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.sip',

    # Jupyter/IPython (definitely not needed)
    'IPython',
    'jupyter',
    'jupyter_client',
    'jupyter_core',
    'ipykernel',
    'notebook',

    # Testing frameworks
    'pytest',

    # Development tools
    'sphinx',
    'docutils',
    'pylint',
    'mypy',
]

# Analysis
a = Analysis(
    # Entry point
    [os.path.join(PROJECT_ROOT, 'src', 'launcher_exe.py')],

    # Path extensions
    pathex=[
        PROJECT_ROOT,
        os.path.join(PROJECT_ROOT, 'src'),
    ],

    # Binary files
    binaries=binaries,

    # Data files
    datas=datas,

    # Hidden imports
    hiddenimports=hidden_imports,

    # Hook paths (none custom)
    hookspath=[],

    # Hook configurations
    hooksconfig={},

    # Runtime hooks (none)
    runtime_hooks=[],

    # Excluded modules
    excludes=excludes,

    # Windows options
    win_no_prefer_redirects=False,
    win_private_assemblies=False,

    # Encryption (none)
    cipher=block_cipher,

    # Don't create archive (faster debugging)
    noarchive=False,
)

# Create PYZ archive (compressed Python bytecode)
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    [],  # Don't include binaries in exe (use COLLECT instead)
    exclude_binaries=True,

    # Executable name
    name='MUSE',

    # Debug mode (set to True for debugging)
    debug=False,

    # Bootloader options
    bootloader_ignore_signals=False,

    # Strip symbols (reduces size but harder to debug)
    strip=False,

    # UPX compression (disabled for DLL compatibility)
    upx=False,

    # Console window (True for debugging, False for release)
    console=False,

    # Windows options
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,

    # Icon (optional)
    icon=os.path.join(PROJECT_ROOT, 'assets', 'icon.ico') if os.path.exists(os.path.join(PROJECT_ROOT, 'assets', 'icon.ico')) else None,
)

# Collect all files into distribution folder
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,

    # Strip symbols
    strip=False,

    # UPX compression
    upx=False,
    upx_exclude=[],

    # Output folder name
    name='PROJECT_MUSE',
)
