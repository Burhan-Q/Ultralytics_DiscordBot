"""
Title: UltralyticsBot
Author: Burhan Qaddoumi
Date: 2023-10-12

Requires: discord.py, pyyaml, numpy, requests, opencv-python
"""

from pathlib import Path

ROOT = Path(__file__).parent
PROJ_ROOT = ROOT.parent.parent

__all__ = 'ROOT', 'PROJ_ROOT'
