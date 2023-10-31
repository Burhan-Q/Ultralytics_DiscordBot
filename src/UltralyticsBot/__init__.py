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
GH = "https://github.com/Burhan-Q/Ultralytics_DiscordBot"

SECRETS = yaml.safe_load((PROJ_ROOT / 'SECRETS/codes.yaml').read_text('utf-8'))
BOT_TOKEN = SECRETS['apikey']
BOT_ID = SECRETS['botID']
HUB_KEY = SECRETS['inferkey']
OWNER_ID = SECRETS['ownerID']
DEV_GUILD = SECRETS['devGuild']

CMDS = yaml.safe_load((PROJ_ROOT / 'cfg/commands.yaml').read_text('utf-8'))

REQ_CFG = yaml.safe_load((PROJ_ROOT / 'cfg/req.yaml').read_text('utf-8'))
DEFAULT_INFER = REQ_CFG['default']
REQ_ENDPOINT = REQ_CFG['endpoint']
REQ_LIM = REQ_CFG['limits']
RESPONSE_KEYS = tuple(REQ_CFG['response'])
MAX_REQ = REQ_CFG['max_req']

ASSETS = PROJ_ROOT / 'assets'

__all__ = 'ROOT', 'PROJ_ROOT', 'SECRETS', 'CMDS', 'REQ_CFG', 'ASSETS', 'BOT_TOKEN', 'BOT_ID', 'HUB_KEY', 'DEFAULT_INFER', 'REQ_ENDPOINT', 'REQ_LIM', 'RESPONSE_KEYS', 'GH'
