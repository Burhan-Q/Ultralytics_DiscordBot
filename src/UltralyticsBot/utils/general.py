"""
Title: bot.py
Author: Burhan Qaddoumi
Date: 2023-10-10

Requires: discord.py, pyyaml, numpy, requests, opencv-python
"""

import re

import requests
import cv2 as cv
import numpy as np

from UltralyticsBot.utils.logging import Loggr

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

def is_img_link(text:str,w_ext:bool=False) -> bool|tuple[bool,str|None]:
    """Verifies string is both valid URL and contains a supported image file extension. When `w_ext=True` will return ``tuple`` with check result and extension."""
    if w_ext:
        r = [ext.group() for ext in tuple(re.search(rf'({e})', text, re.IGNORECASE) for e in IMG_EXT) if ext is not None]
        return (is_link(text), r[0] if any(r) else None)
    else:
        return is_link(text) and any(tuple(re.search(rf'({e})', text, re.IGNORECASE) for e in IMG_EXT))

def data_over_limit(data:bytes, lim:int=2.0) -> bool:
    """Checks if bytes data provided is larger than limit value, default limit is 2.0 MB (2097152 bytes)"""
    return (len(data) / (1024 ** 2)) > lim

def make_3ch_img(image:np.ndarray) -> np.ndarray:
    """Ensure image is always 3 channel BGR"""
    h, w, *c = image.shape
    if any(c) and c[0] > 3:
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR) # NOTE assumes that any open image with more than 3 channels will be BGRA
    if not any(c) and all((h, w)):
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR) # NOTE assuming graysacle image if no channel count, don't expect this to occur
    return image

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

class ReqImage:
    """
    Usage
    ---
    Class for inference request image. Generates image object from URL bytes data and has method to resize based on specified data inference size limit.

    Arguments
    ---
    img_url - ``str`` URL pointing to image file
    MB_lim - ``float`` | ``int`` value for inference limit in megabytes (MB), default value is 2.0 MB
    
    Methods
    ---
    process - retrieves URL image data, creates ``np.ndarray`` image, resizes image if too large for inference, sets `self.image_error=True` if issue present

    gen_image - creates ``np.ndarray`` image from URL data

    data_size - returns the size of image data in MB

    resize_img - when `self.image` is too large for inference, resizes image and stores in `self.infer_img`

    im2bytes - returns bytes data of input image or `self.infer_img` (default), must use original image extension (if none provided, retrieved from URL string)

    Attributes
    ---

    url_good - URL ``string`` passes regex check

    im_ext - original file extension from URL ``string``

    im_url - valid image URL ``string``

    imdata - ``bytes`` data retrieved from URL

    image - ``np.ndarray`` generated from `self.imdata`

    infer_img - ``np.ndarray`` to use for inference, if original `self.image` was too large, resized to below inference size threshold

    image_error - `True` when error encountered during data processing, means any of `self.im_url`, `self.imdata`, `self.image`, or `self.infer_img` are ``None``
    """
    def __init__(self, img_url:str, MB_lim:float|int=2.0) -> None:
        self.__MBsize_limit = MB_lim # inference request size limit, default is 2 MB (2097152 bytes)
        self.__source_url = img_url
        self.url_good, self.im_ext = is_img_link(img_url, True)
        self.im_url = img_url if is_img_link(img_url) or self.url_good else None
        self.imdata = self.image = self.infer_img = None
        self.image_error = False
        Loggr.debug(f"Invalid image URL submitted {self.__source_url}") if self.im_url is None else None
        
    def process(self):
        """Retrieve data from URL (if valid), generate image object, and resize image as needed for inference request."""
        self.imdata = requests.get(self.im_url).content if self.im_url is not None else None
        self.gen_image() if self.imdata is not None else None
        self.resize_img() if self.image is not None  else None
        self.image_error = False if self.imdata and self.image is not None and self.infer_img is not None else True
        # Loggr.debug(f"Invalid image URL submitted {self.__source_url}") if self.im_url is None else None
        Loggr.debug(f"Error retrieving data or image from {self.im_url}.") if self.image_error else Loggr.info(f"Data retrieved successfully.")
        
    def gen_image(self, data:bytes=None) -> None:
        """Create image from URL data."""
        data = self.imdata if data is None else data
        try:
            if self.im_url is not None and data is not None:
                self.image = make_3ch_img(cv.imdecode(np.frombuffer(data, np.uint8), -1))
            else:
                self.image_error = True
                Loggr.debug(f"Problem retrieving source image from data for URL {self.im_url}")
        
        except SyntaxError: # incorrect bytes string will raise this
            self.image_error = True
            Loggr.error(f"Syntax error for data retrieved from URL {self.im_url} when attempting to generate source image")
        
        except Exception as e: # all other error types
            self.image_error = True
            Loggr.error(f"Error {e} occurred when attempting to generate image from data for URL {self.im_url}")
            
    def data_size(self):
        """Returns the size (MB) of the retrieved data"""
        return (len(self.imdata) / (1024 ** 2))
    
    def resize_img(self, data:bytes=None):
        """Resize image dimensions to keep below inference threshold limit."""
        data = self.imdata if data is None else data
        need2shrink = data_over_limit(data, self.__MBsize_limit)
        r = min((self.__MBsize_limit / self.data_size()), 1.0)
        
        if need2shrink and isinstance(self.image, np.ndarray) and not self.image_error:
            self.infer_img = cv.resize(np.copy(self.image), None, (0,0), round(r,2), round(r,2))
        
        elif need2shrink and not isinstance(self.image, np.ndarray) and not self.image_error:
            try:
                _ = self.gen_image(self.imdata)
                self.infer_img = cv.resize(np.copy(self.image), None, (0,0), round(r,2), round(r,2))
            except cv.error:
                self.image_error = True
                Loggr.debug(f"Problem with source image for URL {self.im_url} unable to resize")
        
        elif not need2shrink and isinstance(self.image, np.ndarray) and not self.image_error:
            self.infer_img = np.copy(self.image)
        
        else:
            self.image_error = True
            Loggr.debug(f"Problem retriveving source image or resizing image for URL {self.im_url}")
    
    def im2bytes(self, image:np.ndarray=None, enc:str=''):
        """Either converts input image or `self.infer_img` from ``np.ndarray`` to ``bytes``."""
        enc = self.im_ext if enc in [None,''] else enc
        image = self.infer_img if image is None else image
        return cv.imencode(enc, image)[1].tobytes()
    
    
# Large test image "https://i.imgur.com/pDNOqoa.png"
# Normal test image "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"