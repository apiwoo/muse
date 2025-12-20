# Project MUSE - ai/parsing/__init__.py
# AI Skin Parsing Module
# (C) 2025 MUSE Corp. All rights reserved.

"""
AI Skin Parsing module for high-precision skin mask generation.

This module provides:
- SkinParser: BiSeNet V2 TensorRT-based face parsing for pixel-level skin detection
"""

from .skin_parser import SkinParser

__all__ = ['SkinParser']
