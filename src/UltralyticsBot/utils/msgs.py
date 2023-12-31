"""
Title: msgs.py
Author: Burhan Qaddoumi
Date: 2023-10-29

Requires: requests
"""
import re
from typing import Callable

import discord
import requests

from UltralyticsBot import GH, BOT_ID
from UltralyticsBot.utils.general import dec2str, align_boxcoord
from UltralyticsBot.utils.checks import is_link, is_img_link, URL_RGX

NEWLINE = '\n' # use with f-strings
BOX_LJUST = 24 # Box coordinates will always be -> '(1234, 1234, 1234, 1234)'
# NOTE confidence values don't need justification, will always use 0.123
# NOTE class name will need to dynamically justify

IMG_ERR_MSG = f"Error occured when fetching image, check URL and try again. Open issue and include URL on [project repo]({GH}) if continued problems with working image URL."
API_ERR_MSG = "Error: API call failed with {} - {}" # response.status-code, response.reason
IMGSZ_MSG = '**__NOTE:__** Results are for image scaled by `{}` from original size, as required for inference.\n'
NOT_OWNER = f"This command is only for the Bot owner."

def longest(results:list[dict|str], _pad:int=2):
    """Finds the length of the longest class name string in results and adds padding spaces (2 by default)."""
    name_len = {len(n):n for n in set((r['name'] if isinstance(r, dict) else r) for r in results)}
    return sorted(name_len)[-1] + _pad

def gen_title(CL:int):
    """Generate title string for results. Requires padding length for 'class' which usually should be calculated dynamically."""
    return "{} {}   {}\n".format('class'.ljust(CL), 'conf'.ljust(4), 'x1y1x2y2'.ljust(BOX_LJUST))

def gen_line(cls_name:str, CL:int, conf:float, x1:int, y1:int, x2:int, y2:int):
    return '{} {}  {}\n'.format(cls_name.ljust(CL), dec2str(conf), align_boxcoord([x1,y1,x2,y2]).ljust(BOX_LJUST))

def get_args(args:list, chr:str=" ", n:int=1) -> list[str]:
    """Split string with character `chr` and return list values after `n`, defaults are `chr=' '` (space) and `n=1`"""
    return args.split(chr)[n:]

class ResponseMsg():
    def __init__(self, api_reply:requests.models.Response, plot:bool, txt:bool, ratio:float=1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.api_reply = api_reply
        self.plot = plot or not txt
        self.txt = txt
        self.ratio = ratio
        self.response()
        
    def response(self):
        """Processes API response information."""
        self.reply_dict = self.api_reply.json()
        self.reason = self.api_reply.reason
        self.code = self.api_reply.status_code
        for k in ['data', 'message', 'success']:
            setattr(self, k, self.reply_dict[k])
        
        self.msg = f'''{getattr(self, 'message')}\n'''
        self.cls_pad = 2 if not any(self.data) else longest(self.data)
    
    def start_msg(self, plt_fn:Callable, infer_ratio:float=1.0, highlight:str=''):
        self.ratio = infer_ratio if self.ratio == 1.0 else self.ratio
        
        if self.reason == 'OK' or self.code == 200:
            self.msg += IMGSZ_MSG.format(self.ratio) if self.ratio != 1.0 and self.txt else ''
            # self.cls_pad = 2 if not any(self.data) else longest(self.data)
            self.msg += '```{}\n'.format(highlight) if self.txt else ''
            self.msg += gen_title(self.cls_pad) if self.txt else ''
            
            self.anno_im, self.result_txt = plt_fn(predictions=self.data)
            self.msg += ((self.result_txt + '```') if self.txt else ('```' if self.txt and self.result_txt != '' else ''))
        else:
            self.anno_im = None
            self.msg = API_ERR_MSG.format(self.code, self.reason)
        
        return (self.anno_im, self.msg)

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
    
    def media_info(self) -> tuple[int|None,int|None,float|None]:
        """If image was attached, returns image height, width, and file-size."""
        return self.im_height, self.im_width, self.img_size