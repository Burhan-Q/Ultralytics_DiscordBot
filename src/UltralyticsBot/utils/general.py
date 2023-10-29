"""
Title: general.py
Author: Burhan Qaddoumi
Date: 2023-10-10

Requires: discord.py, pyyaml, numpy, requests, opencv-python
"""

import re

import requests
import cv2 as cv
import numpy as np
import discord

from UltralyticsBot import BOT_ID
from UltralyticsBot.utils.logging import Loggr
from UltralyticsBot.utils.checks import is_img_link, is_link, URL_RGX

# MODEL_RGX = r'((yolov)(5|8)(n|s|m|l|x))'
# URL_RGX = r"(http[s]?:\/\/)?(www)?[.]?([a-zA-Z0-9\-]+([.][a-zA-Z0-9\-]{2,63})+)([/]+[a-zA-Z0-9?$&;^~=+!,:@\-#._]*(%[0-9a-fA-F]{2})*[a-zA-Z0-9?$&;^~=+!,:@\-#._]*)*" # https://regex101.com/r/VzFmEN/2 NOTE captures most but not all URLs, anywhere in text
# IMG_EXT = ('.bmp', '.png', '.jpeg', '.jpg', '.tif', '.tiff', '.webp') # reference docs.ultralytics.com/modes/predict/#images, skipping (.mpo, .dng, .pfm)

# def float_str(num:str) -> bool:
#     """Checks if string is a valide float-like number. Returns `True` when all values around `.` are numeric and only one `.` is present, otherwise returns `False`."""
#     return all([n.isnumeric() for n in num.split('.')]) and num.count('.') == 1

# def req_values(data:dict) -> dict:
#     """Converts string values in dictionary to either ``float`` or ``int`` where appropriate."""
#     for k,v in data.items():
#         if isinstance(v,str) and float_str(v):
#             data[k] = float(v)
#             # if None or is (str) but not float_str, do nothing
#         elif isinstance(v,str) and v.isnumeric():
#             data[k] = int(v)
#     return data

def align_boxcoord(pxl_coords:list|tuple) -> str:
    """Creates string from pixel-space bounding box coordinates and right aligns coordinates."""
    return f'{str(tuple(str(coor).rjust(4) for coor in pxl_coords))}'.replace("'","")

def dec2str(num:float, round2:int=3, trail_zeros:bool=True) -> str:
    """Rounds decimal number to number of places provided, adds trailing zeros when `trail_zeros=True` (default)"""
    num = str(round(float(num), int(round2)))
    return num + ('0' * abs(len(num.split('.')[-1]) - int(round2)) if trail_zeros else '')

# def is_link(text:str) -> bool:
#     """Verify if string is a valid URL with regex"""
#     # return any([text.lower().startswith(h) for h in ['http://','https://','www.']])
#     return re.search(URL_RGX, text, re.IGNORECASE) is not None

# def is_img_link(text:str,w_ext:bool=False) -> bool|tuple[bool,str|None]:
#     """Verifies string is both valid URL and contains a supported image file extension. When `w_ext=True` will return ``tuple`` with check result and extension."""
#     if w_ext:
#         r = [ext.group() for ext in tuple(re.search(rf'({e})', text, re.IGNORECASE) for e in IMG_EXT) if ext is not None]
#         return (is_link(text), r[0] if any(r) else None)
#     else:
#         return is_link(text) and any(tuple(re.search(rf'({e})', text, re.IGNORECASE) for e in IMG_EXT))

def data_over_limit(data:bytes, lim:int=2.0) -> bool:
    """Checks if bytes data provided is larger than limit value, default limit is 2.0 MB (2097152 bytes)"""
    return (len(data) / (1024 ** 2)) > lim

def image_oversize(img:np.ndarray=None, img_dims:tuple|list=(0,0), hLim:int=1280, wLim:int=1280): # NOTE replace with values imported from YAML
    assert img is not None or any(img_dims), Loggr.error(f"Neither image object or dimensions provided.")
    height, width = img_dims if any(img_dims) else img.shape[:2]
    return (height / hLim) > 1 or (width / wLim) > 1

def make_3ch_img(image:np.ndarray) -> np.ndarray:
    """Ensure image is always 3 channel BGR"""
    h, w, *c = image.shape
    if any(c) and c[0] > 3:
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR) # NOTE assumes that any open image with more than 3 channels will be BGRA
    if not any(c) and all((h, w)):
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR) # NOTE assuming graysacle image if no channel count, don't expect this to occur
    return image

# def model_chk(model_str:str) -> str:
#     """Checks that model provided is conforms to standard string format, will default to YOLOv8 model if not valid version provided, and defaults to nano size if no valid model size provided."""
#     if not is_link(model_str): # TODO add check for valid HUB link
#         full = re.match(MODEL_RGX, model_str.lower(), re.IGNORECASE)
#         if full:
#             out = full.group().lower()
#         elif not full:
#             try:
#                 num = re.search(r'\d', model_str).group()
#                 num = num if num in ['5','8'] else '8'
#             except AttributeError:
#                 num = '8'
#             size = model_str[-1].lower() if model_str[-1].lower() in ['n','s','m','l','x'] else 'n'
#             out = 'yolov' + num + size
#     else:
#         out = model_str
    
#     return out

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
    Attributes
    ---
    url_good - ``bool``
        When `True` URL appears good, otherwise `False`.

    im_url - ``str`` | ``None``
        String URL for image, when URL is good and URL appears to be for an image.

    imdata - ``bytes`` | ``None``
        Image from `self.image` as ``bytes`` data, or ``None`` if invalid.

    size - ``int`` | ``None``
        Size (MB) of `self.imdata` or value passed from `discord.Message` when image is attached.

    image - ``np.ndarry`` | ``None``
        When URL is valid, image data retrived from link.

    im_ext - ``str``
        Original file extension of image retrived.

    height - ``int`` | ``None``
        Value for `self.image` height.

    width - ``int`` | ``None``
        Value for `self.image` width.

    image_error - ``bool``
        If error occurs while retriving image, will be `True` otherwise `False`.

    Methods
    ---
    data_size() - Returns ``float`` of `self.imdata` in MB

    get_image() - Fetches data from provided URL, executed during `__init__`

    inference_img(infer_size, enc, Q) - Calculates and scales `self.image` dimensions for inference as needed, repopulates `self.size`, `self.height`, `self.width`, and `self.imdata` attributes if resized.

        - infer_size ``int`` - size for inference, default 640

        - enc ``str`` - file extension encoding for bytes data, default '.jpeg'

        - Q ``int`` - percentage to compress data
    """
    def __init__(self, img_url:str, MB_lim:float|int=2.0, **kwargs) -> None:
        self.__MBsize_limit = MB_lim # inference request size limit, default is 2 MB (2097152 bytes)
        self.__source_url = img_url
        self.__source_img = None
        self.url_good, self.im_ext = is_img_link(img_url, True)
        self.im_url = img_url if self.url_good or is_img_link(img_url) else None
        self.imdata = self.image = self.size = self.height = self.width = None
        self.image_error = False
        if any(kwargs):
            _ = [setattr(self, k, v) for k,v in kwargs.items()]
        self.get_image()
    
    def data_size(self):
        """Returns the size (MB) of the retrieved data"""
        return (len(self.imdata) / (1024 ** 2))
    
    def get_image(self):
        self.imdata = requests.get(self.im_url).content if self.im_url is not None else None
        try:
            if self.im_url is not None and self.imdata is not None:
                self.image = make_3ch_img(cv.imdecode(np.frombuffer(self.imdata, np.uint8), -1))
                self.__source_img = np.copy(self.image)
                self.height, self.width = self.image.shape[:2]
            else:
                self.image_error = True
                Loggr.debug(f"Problem retrieving source image from data for URL {self.im_url}")
        
        except SyntaxError: # incorrect bytes string will raise this
            self.image_error = True
            Loggr.error(f"Syntax error for data retrieved from URL {self.im_url} when attempting to generate source image")
    
        except Exception as e: # all other error types
            self.image_error = True
            Loggr.error(f"Error {e} occurred when attempting to generate image from data for URL {self.im_url}")
    
    def inference_img(self, infer_size:int=640, enc:str='.jpeg', Q:int=70) -> tuple[np.ndarray, bytes, float]:
        """Generates inference image by resizing and compressing data as required. Returns inference image, image bytes, and resized ratio."""
        R = 1.0
        need2resize = data_over_limit(self.imdata, self.__MBsize_limit) or image_oversize(img_dims=(self.height, self.width))
        self.size = self.data_size() if self.size is None else self.size
        if need2resize:
            R = round(min(((self.__MBsize_limit / self.size)), infer_size / self.height, infer_size / self.width, 1.0), 2)
            self.image = cv.resize(np.copy(self.image), None, None, R, R)
            enc_params = None if enc.lower() not in ['.jpeg', '.jpg'] else (cv.IMWRITE_JPEG_QUALITY, Q)
            self.imdata = cv.imencode(enc, self.image, enc_params)[1].tobytes()
            self.height, self.width = self.image.shape[:2]

        return self.image, self.imdata, R
    
# Large test image "https://i.imgur.com/pDNOqoa.png"
# Normal test image "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"

class ReqMessage:
    """
    Attributes
    ---
    msg - ``discord.Message``
        Discord message for inference request.

    url - ``str`` | ``None``
        URL string when `self.msg` contains valid URL, otherwise ``None``.

    author - ``discord.Message.author`` | ``None``
        Discord message author.

    mentions - ``list``
        List of mentioned users in `self.msg`.

    media - ``list[discord.Attachment]``
        List of media attached to `self.msg`.

    im_width - ``int`` | ``None``
        Width in pixels of image attched to `self.msg` if any, otherwise ``None``.

    im_height - ``int`` | ``None``
        Height in pixels of image attched to `self.msg` if any, otherwise ``None``.

    has_url - ``bool``
        Is `True` when text from `self.msg` contains what appears to be valid URL string, otherwise `False`.

    has_media - ``bool``
        Is `True` when `self.msg` contains any attachments, otherwise `False`.

    has_text - ``bool``
        Is `True` when `self.msg` contains text other than triggering-keywords and whitespace, otherwise `False`.

    has_img - ``bool``
        Is `True` when `self.msg` has attachment with content type 'image', otherwise `False`

    bot_mention - ``bool``
        Is `True` when `self.msg` contains bot-user mention/tag, otherwise `False`.

    Methods
    ---
    check_message() - Executes during `__init__` and populates attributes from `discord.Message`

    get_url() - If attributes not populated, executes `check_message()` then returns first URL as ``str`` found in `discord.Message.content`, otherwise returns ``None``.
    """
    commands = ['predict', 'help', 'about', 'hub'] # NOTE maybe pull from YAML instead
    
    def __init__(self, msg:discord.Message) -> None:
        self.msg = msg
        self.url = self.author = self.mentions = self.media = None
        self.im_height = self.im_width = self.attached_im = self.img_size = None
        self.has_url = self.has_media = self.has_text = self.has_img = self.bot_mention = False
        self.check_message()
        
    def check_message(self) -> None:
        self.has_text = isinstance(self.msg.content,str) and any(self.msg.content.replace("$predict","").strip())
        self.media = self.msg.attachments if any(self.msg.attachments) else []
        
        self.has_url = is_link(self.msg.content) if self.has_text else False
        self.has_img = (['image' in a.content_type for a in self.media]) if any(self.media) else False
        
        self.author = self.msg.author
        self.mentions = self.msg.mentions
        self.bot_mention = BOT_ID in [m.id for m in self.mentions]
        
        if self.has_text and self.has_url:
            self.url = re.search(URL_RGX, self.msg.content, re.IGNORECASE).group()
        
        elif self.has_img:
            self.attached_im = [a for a in self.media if 'image' in a.content_type][0] # only allow one image
            self.url = self.attached_im.url
            self.im_height, self.im_width = self.attached_im.height, self.attached_im.width
            self.img_size = self.attached_im.size / (1024 ** 2)
    
    def get_url(self) -> str:
        _ = self.check_message() if self.url is None else None
        return self.url
    
class ResponseMsg():
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        