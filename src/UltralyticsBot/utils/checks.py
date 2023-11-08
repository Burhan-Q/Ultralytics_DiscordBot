"""
Title: checks.py
Author: Burhan Qaddoumi
Date: 2023-10-10

Requires: re
"""

import re
from urllib.parse import urlparse

MODEL_RGX = r'((yolov)(5|8)(n|s|m|l|x))'
URL_RGX = r"((http[s]?:\/\/)|(www))?[.]?([a-zA-Z0-9\-]+([.][a-zA-Z0-9\-]{2,63})+)([/]+[a-zA-Z0-9?$&;^~=+!,:@\-#._]*(%[0-9a-fA-F]{2})*[a-zA-Z0-9?$&;^~=+!,:@\-#._]*)*" # https://regex101.com/r/VzFmEN/2 NOTE captures most but not all URLs, anywhere in text
IMG_EXT = ('.bmp', '.png', '.jpeg', '.jpg', '.tif', '.tiff', '.webp') # reference docs.ultralytics.com/modes/predict/#images, skipping (.mpo, .dng, .pfm)

def is_link(text:str) -> bool:
    """Verify if string is a valid URL with regex and urlparse, loose-checker and could still fail."""
    return re.search(URL_RGX, text, re.IGNORECASE) is not None or urlparse(text).netloc != ''

def is_img_link(text:str,w_ext:bool=False) -> bool|tuple[bool,str|None]:
    """Verifies string is both valid URL and contains a supported image file extension. When `w_ext=True` will return ``tuple`` with check result and extension."""
    if w_ext:
        r = [ext.group() for ext in tuple(re.search(rf'({e})', text, re.IGNORECASE) for e in IMG_EXT) if ext is not None]
        return (is_link(text), r[0] if any(r) else None)
    else:
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
