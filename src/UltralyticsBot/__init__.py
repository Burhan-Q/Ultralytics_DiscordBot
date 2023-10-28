"""
Title: UltralyticsBot
Author: Burhan Qaddoumi
Date: 2023-10-12

Requires: discord.py, pyyaml, numpy, requests, opencv-python
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).parent
PROJ_ROOT = ROOT.parent.parent

SECRETS = yaml.safe_load((PROJ_ROOT / 'SECRETS/codes.yaml').read_text('utf-8'))
CMDS = yaml.safe_load((PROJ_ROOT / 'cfg/commands.yaml').read_text('utf-8'))
REQ_CFG = yaml.safe_load((PROJ_ROOT / 'cfg/req.yaml').read_text('utf-8'))
ASSETS = PROJ_ROOT / 'assets'

__all__ = 'ROOT', 'PROJ_ROOT', 'SECRETS', 'CMDS', 'REQ_CFG', 'ASSETS'
