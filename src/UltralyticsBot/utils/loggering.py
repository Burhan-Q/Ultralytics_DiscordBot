"""
Title: utils/logging
Author: Burhan Qaddoumi
Date: 2023-10-12

Requires: 
"""

import logging
import logging.config
from pathlib import Path

import yaml

from UltralyticsBot import PROJ_ROOT

try:
    config_file = next((PROJ_ROOT / 'cfg').glob("Loggr.yaml"))
except StopIteration:
    config_file = list(Path(__file__).parent.parent.parent.rglob('Loggr.yaml'))

if config_file is not None and config_file != []:
    config_file = config_file[0] if isinstance(config_file, list) else config_file
    config = yaml.safe_load(config_file.read_text("utf-8"))

logging.config.dictConfig(config)
Loggr = logging.getLogger('Loggr')

