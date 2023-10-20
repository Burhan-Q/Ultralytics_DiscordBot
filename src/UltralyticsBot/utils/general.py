"""
Title: bot.py
Author: Burhan Qaddoumi
Date: 2023-10-10

Requires: discord.py, pyyaml, numpy, requests, opencv-python
"""

import re
import base64

import requests

MODEL_RGX = r'((yolov)(5|8)(n|s|m|l|x))'
URL_RGX = r"^(http[s]?:\/\/)?([^:\/\s]+)(:([^\/]*))?(\/\w+\.)*([^#?\s]+)(\?([^#]*))?(#(.*))?$" # source regex101.com/r/lQ1nI3
IMG_EXT = ('.bmp', '.png', '.jpeg', '.jpg', '.tif', '.tiff', '.webp') # reference docs.ultralytics.com/modes/predict/#images, skipping (.mpo, .dng, .pfm)

def float_str(num:str) -> bool:
    """Checks if string is a valide float-like number. Returns `True` when all values around `.` are numeric and only one `.` is present, otherwise returns `False`."""
    return all([n.isnumeric() for n in num.split('.')]) and num.count('.') == 1

def req_values(data:dict) -> dict:
    """Converts string values in dictionary to either ``float`` or ``int`` where appropriate."""
    for k,v in data.items():
        if isinstance(v,str) and float_str(v):
            data[k] = float(v)
            # if None or is (str) but not float_str, do nothing
        elif isinstance(v,str) and v.isnumeric():
            data[k] = int(v)
    return data

def align_boxcoord(pxl_coords:list|tuple) -> str:
    """Creates string from pixel-space bounding box coordinates and right aligns coordinates."""
    return f'{str(tuple(str(coor).rjust(4) for coor in pxl_coords))}'.replace("'","")

def dec2str(num:float, round2:int=3, trail_zeros:bool=True) -> str:
    """Rounds decimal number to number of places provided, adds trailing zeros when `trail_zeros=True` (default)"""
    num = str(round(float(num), int(round2)))
    return num + ('0' * abs(len(num.split('.')[-1]) - int(round2)) if trail_zeros else '')

def is_link(text:str) -> bool:
    """Verify if string is a valid URL with regex"""
    # return any([text.lower().startswith(h) for h in ['http://','https://','www.']])
    return re.search(URL_RGX, text, re.IGNORECASE) is not None

def is_img_link(text:str) -> bool:
    """Verifies string is both valid URL and contains a supported image file extension."""
    return is_link(text) and any(tuple(re.search(rf'({e})', text, re.IGNORECASE) for e in IMG_EXT))

def model_chk(model_str:str) -> str:
    """Checks that model provided is conforms to standard string format, will default to YOLOv8 model if not valid version provided, and defaults to nano size if no valid model size provided."""
    if not is_link(model_str): # TODO add check for valid HUB link
        full = re.match(MODEL_RGX, model_str.lower(), re.IGNORECASE)
        if full:
            out = full.group().lower()
        elif not full:
            try:
                num = re.search(r'\d', model_str).group()
                num = num if num in ['5','8'] else '8'
            except AttributeError:
                num = '8'
            size = model_str[-1].lower() if model_str[-1].lower() in ['n','s','m','l','x'] else 'n'
            out = 'yolov' + num + size
    else:
        out = model_str
    
    return out

def gen_cmd(model:str,
            source:str,
            conf:float,
            iou:float,
            size:int,
            show:bool,
            py:bool=True,
            cli:bool=False
            ) -> str:
    """Generates command/code for inference with YOLOv5 or YOLOv8, generates YOLOv8 for python command when `py=True` and for command line interface when `cli=True`, YOLOv5 commands will always be generated for command line."""
    py = py if cli != py else not cli
    if not is_link(model):
        is_v8 = model.lower().startswith('yolov8')
        if is_v8 and py:
            cmd = '```py\n'
            cmd += 'from ultralytics import YOLO\n'
            cmd += f'model = YOLO("{model.lower()}.pt")\n'
            cmd += f'img = {source}\n'
            args = f'source=img, imgsz={size}, conf={dec2str(conf,2,False)}, iou={dec2str(iou,2,False)}, show={show}'
            cmd += f'model.predict({args})\n'
            # FUTURE additional stesp here as needed
            cmd += '```'
        
        elif is_v8 and cli:
            cmd = '```bash\n'
            args = f'source={source}, imgsz={size}, conf={dec2str(conf,2,False)}, iou={dec2str(iou,2,False)}, show={show}'
            cmd += f'yolo predict {args}\n'
            cmd += '```'
        
        elif not is_v8:
            cmd = '```bash\n'
            args = f'--weights {model.lower()}.pt --source {source} --imgsz {size} --conf-thres {dec2str(conf,2,False)} --iou-thres {dec2str(iou,2,False)}'
            cmd += f' python detect.py {args}\n'
            cmd += '```'
    
    return cmd

