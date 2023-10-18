"""
Title: bot.py
Author: Burhan Qaddoumi
Date: 2023-10-10

Requires: discord.py, pyyaml, numpy, requests, opencv-python
"""

import re

MODEL_RGX = r'((yolov)(5|8)(n|s|m|l|x))'

def dec2str(num:float, round2:int=3, trail_zeros:bool=True) -> str:
    """Rounds decimal number to number of places provided, adds trailing zeros when `trail_zeros=True` (default)"""
    num = str(round(float(num), int(round2)))
    return num + ('0' * abs(len(num.split('.')[-1]) - int(round2)) if trail_zeros else '')

def is_link(text:str) -> bool:
    """Verify if string starts with any of `http://`, `https://`, or `wwww.`"""
    return any([text.lower().startswith(h) for h in ['http://','https://','www.']])

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

