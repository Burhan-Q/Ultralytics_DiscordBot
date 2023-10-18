"""
Title: utils/plotting
Author: Burhan Qaddoumi
Date: 2023-10-11

Requires: pyyaml, numpy, opencv-python
"""
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml

from UltralyticsBot import PROJ_ROOT

COLORS_FILE = PROJ_ROOT / 'cfg/colors.yaml'

def hex2bgr(hexcolor) -> tuple[int,int,int]:
    # from /ultralytics/utils/plotting.py#L54
    """Conver HEX color strings to BGR color format."""
    return tuple(int(hexcolor[1 + i:1 + i + 2], 16) for i in (0, 2, 4))[::-1]

def get_colors() -> tuple[tuple[int,int,int]]:
    """Load colors for annotations from YAML file."""
    hex_colors = yaml.safe_load(Path(COLORS_FILE).read_text())['colors']
    return tuple(hex2bgr(col) for col in hex_colors)

COLORS = get_colors()

def select_color(n) -> int:
    """Cycle colors if there are less in colors.yaml than the number of class indices."""
    lim = len(COLORS)
    N = lim + 1
    
    if n <= lim:
        N = n

    elif n > lim:
        while N > lim:
            N = n % lim
            n = N
            if -lim < N < lim:
                break

    return int(N)

def rel_line_size(imH:int, imW:int):
    """Calculates line thickness size relative to largest image dimension."""
    return max(round(max(imH, imW) * 0.003), 2)

def xcycwh2xyxy(boxes:np.ndarray) -> np.ndarray:
    """Convert x-center, y-center, width, height bounding box to xmin, ymin, xmax, ymax bounding box."""
    xc, yc, w, h = np.split(boxes, 4, -1)
    x1, y1 = xc - (w / 2), yc - (h / 2)
    x2, y2 = x1 + w, y1 + h
    return np.hstack([x1,y1,x2,y2])

def nxy2xy(box:np.ndarray, imH:int, imW:int) -> np.ndarray:
    """Convert bounding box coordinates from normalized to pixel values, works for x1y1x2y2 and xywh."""
    x1, x2 = (box[..., ::2] * imW).astype(np.int_)
    y1, y2 = (box[..., 1::2] * imH).astype(np.int_)
    return np.hstack([x1, y1, x2, y2])

def drawbox(draw:np.ndarray, box:np.ndarray, line_size:int=3) -> np.ndarray:
    """Draws single bounding box on image, box format is x1y1x2y2."""
    # x1, y1, x2, y2, cl = [int(bn) for bn in box]
    x1, y1, x2, y2, cl = box.astype(int)
    iclr = select_color(cl)
    _ = cv.rectangle(draw, (x1,y1), (x2,y2), COLORS[iclr], line_size)

def draw_all_boxes(image:np.ndarray, boxes:np.ndarray, line_size:int=3):
    """Draws all bounding boxes on image, box format is x1y1x2y2."""
    _ = [drawbox(image, b.squeeze(), line_size) for b in boxes]
    return image

