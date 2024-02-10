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

# Secrets config
SECRETS:dict = yaml.safe_load((PROJ_ROOT / 'SECRETS/codes.yaml').read_text('utf-8'))
BOT_TOKEN = SECRETS['apikey']
BOT_ID = SECRETS['botID']
HUB_KEY = SECRETS['inferkey']
OWNER_ID = SECRETS['ownerID']
DEV_GUILD = SECRETS['devGuild']
DEV_CH = SECRETS['devCh']

# Commands config
CMDS:dict = yaml.safe_load((PROJ_ROOT / 'cfg/commands.yaml').read_text('utf-8'))

# Inference request config
REQ_CFG:dict = yaml.safe_load((PROJ_ROOT / 'cfg/req.yaml').read_text('utf-8'))
DEFAULT_INFER = REQ_CFG['default']
REQ_ENDPOINT = REQ_CFG.get('endpoint') or SECRETS.get('endpoint')
REQ_LIM = REQ_CFG['limits']
RESPONSE_KEYS = tuple(REQ_CFG['response'])
MAX_REQ = REQ_CFG['max_req']
MODELS = REQ_CFG['models']

# Docker config
DOCKER_CFG = yaml.safe_load((PROJ_ROOT / 'compose.yaml').read_text('utf-8'))
REPO_DIR = DOCKER_CFG['services']['bot']['build']['args']['REPO_DIR'].strip().lower()

ASSETS = PROJ_ROOT / 'assets'

# Models Regex
YOLOv5_REGEX = r"^yolov5(n|s|m|l|x)(u|6u)$"
YOLOv8_REGEX = r"^yolov8(n|s|m|l|x)(-cls|-seg|-pose|-obb)?$"

__all__ = 'ROOT', 'PROJ_ROOT', 'SECRETS', 'CMDS', 'REQ_CFG', 'ASSETS', 'BOT_TOKEN', 'BOT_ID', 'HUB_KEY', 'DEFAULT_INFER', 'REQ_ENDPOINT', 'REQ_LIM', 'RESPONSE_KEYS', 'GH', 'YOLOv5_REGEX', 'YOLOv8_REGEX', 'MODELS'
