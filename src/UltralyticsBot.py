"""
Title: UltralyticsBot
Author: Burhan Qaddoumi
Date: 2023-10-10

Requires: discord.py, pyyaml, numpy, requests, opencv-python
"""

import base64
import json
import re
from pathlib import Path

import cv2 as cv
import discord
import numpy as np
import requests
import yaml
from discord import app_commands

from utils.plotting import nxy2xy, xcycwh2xyxy, draw_all_boxes, select_color

# References
# Discord slash-command: https://stackoverflow.com/questions/71165431/how-do-i-make-a-working-slash-command-in-discord-py
# Discord library docs: https://discordpy.readthedocs.io/en/stable/
# Dicord library examples: https://github.com/Rapptz/discord.py/tree/v2.3.2/examples

COLORS_FILE = Path('cfg/colors.yaml')
ASSETS = Path('assets') # NOTE aim is to use this as fallback when no image is provided
DEFAULT_INFER = {
                "confidence": "0.35",
                "iou": "0.45",
                "size": "640",
                "model": "yolov5n",
                "key": None,
                "image": None,
                }
REQ_ENDPOINT = "https://test.ultralytics.com/detect"
RESPONSE_KEYS = ('name', 'confidence', 'class', 'xcenter', 'ycenter', 'width', 'height')
TEMPFILE = 'detect_res.png'

# NOTE unverified bots in < 100 servers will be able to use message content intents, once above 100 servers, bot needs to be verified
class MyClient(discord.Client):
    """Class for Discord slash-commands, requires message content intents"""
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup(self):
        """Sync commands, could take upto an hour to show up when bot is in lots of servers"""
        await self.tree.sync()

def hex2bgr(hexcolor) -> tuple[int,int,int]: # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/plotting.py#L54
    """Conver HEX color strings to BGR color format"""
    return tuple(int(hexcolor[1 + i:1 + i + 2], 16) for i in (0, 2, 4))[::-1]

def get_colors() -> tuple[tuple[int,int,int]]:
    """Load colors for annotations"""
    hex_colors = yaml.safe_load(Path(COLORS_FILE).read_text())['colors']
    return tuple(hex2bgr(col) for col in hex_colors)

def get_values(data:dict) -> list:
    """Return values for known response keys"""
    return [data[k] for k in RESPONSE_KEYS]

def cleanup():
    """Deletes temporary image file after finished."""
    _ = Path(TEMPFILE).unlink(missing_ok=True)

def plot_result(imgbytes:bytes, predictions:list, include_msg:bool=False):
    # Load image
    img = cv.imdecode(np.frombuffer(imgbytes, np.uint8), -1)
    imH, imW = img.shape[:2]
    anno_img = np.copy(img)
    
    msg = f'''Detections:\n''' # TODO include @user
    pred_boxes = np.zeros((1,5)) # x-center, y-center, width, height, class
    # NOTE maybe better to convert bbox coordinates before generating message?
    for p in predictions:
        cls_name, conf, idx, *(x, y, w, h) = get_values(p)
        x1, y1, x2, y2 = tuple(nxy2xy(xcycwh2xyxy(np.array((x, y, w, h))), imH, imW)) # n-xcycwh -> x1y1x2y2
        pred_boxes = np.vstack([pred_boxes, np.array((x1, y1, x2, y2, idx))])
        msg += f'''class:  {cls_name} conf:   {round(conf,3)} index:  {idx} x1y1x2y2: {(x1, y1, x2, y2)}\n'''
        # clr_idx = select_color(idx)
        # _ = cv.rectangle(anno_img, (x1, y1), (x2, y2), COLORS[clr_idx], 3)

    anno_img = draw_all_boxes(anno_img, pred_boxes[1:])

    return (anno_img, None) if not include_msg else (anno_img, msg)

def reply_msg(response:requests.models.Response, plot:bool=False, req_img:bytes=None):
    plot = (req_img is not None or req_img != '') and plot

    pred, reply, success = response.json().values()
    msg = f'''{reply}\n''' # TODO include @user

    if (success or response.status_code == 200) and plot:
        result, msg = plot_result(req_img, pred, True)
        
        # NOTE figure out how to skip file save and directly upload from memory
        _ = cv.imwrite(TEMPFILE, result)
        box_img = discord.File(Path(TEMPFILE))
        out = (msg, box_img)

    elif success and not plot:
        for p in pred:
            cls_name, conf, idx, *(x, y, w, h) = get_values(p)
            # NOTE normalized (x,y,w,h) bounding boxes since image not loaded
            msg += f'''class:  {cls_name} conf:   {round(conf,3)} index:  {idx} xywh: {tuple(round(v,3) for v in (x, y, w, h))}\n'''
            out = (msg, None)

    else:
        out = ("Error: API call failed", None)
    
    return out

COLORS = get_colors()

def main(T,H):
    intents = discord.Intents.default()
    intents.message_content = True
    # client = discord.Client(intents=intents)
    client = MyClient(intents=intents)

    # TODO
    ## change to get message content URL (probably first one only)
    ## add formatting for returned message
    ## figure out how to limit requests per user, per day
    ## add argument for selecting YOLO model size, image size, confidence threshold, IOU threshold, etc.
    ## add means to select different weights
    ## add option for choosing other tasks (as they become available)
    ## add means to authenticate with HUB account

    @client.event
    async def on_ready():
        await client.tree.sync()
        print("Initialized sync")

    # TODO
    # change to use default inference with using $predict text in message, that way it's possible to attach file or use image link
    @client.event
    async def on_message(message):

        if message.content.startswith("$predict"):

            image_url = message.content.strip('$predict ') # assume only image URL is passed
            if image_url == '' and len(message.attachments) == 1:
                image_url = message.attachements[0].url

            request_url = REQ_ENDPOINT
            image_data = requests.get(image_url).content
            
            req_dict = DEFAULT_INFER.copy()
            req_dict['key'] = H
            req_dict['image'] = base64.b64encode(image_data).decode()

            response = requests.post(request_url, json=req_dict)

            if response.status_code == 200:
                preds, reply, success = response.json().values()
                result, _ = plot_result(image_data, preds, False)
                                
                _ = cv.imwrite(TEMPFILE, result)
                box_img = discord.File(Path(TEMPFILE))
                await message.reply("Detections", file=box_img)
                
                # cleanup()
            
            else:
                await message.channel.send("Error: API call failed")

    # TODO maybe include @user in response
    @client.tree.command(name='predict')
    @app_commands.describe(
        img_url="REQUIRED: Full URL to image",
        conf="OPTIONAL: Confidence threshold for object classification",
        iou="OPTIONAL: Intersection over union threshold for object detection",
        size="OPTIONAL: Largest image dimension, single number only",
        model="OPTIONAL: One of 'yolov5(n|m|l|x)' or 'yolov8(n|m|l|x)'; EXAMPLE: yolov5n",
    )
    async def im_predict(interaction:discord.Interaction, conf:float=0.3, iou:float=0.4, size:int=640, model:str='yolov8n', img_url:str='', show:bool=False):
        linklike = img_url.startswith('http://') or img_url.startswith('https://') or img_url.startswith('www.')
        if linklike:
            image_data = requests.get(img_url).content
        request_dict = {
                "confidence": str(conf),
                "iou": str(iou),
                "size": str(size),
                "model": str(model),
                "key": str(H),
                "image": base64.b64encode(image_data).decode(),
                }
        req = requests.post(REQ_ENDPOINT, json=request_dict)
        text, file = reply_msg(req, show, image_data)
        await (interaction.response.send_message(text, file=file) if file is not None else interaction.response.send_message(text))

    client.run(T)

if __name__ == '__main__':
    d = yaml.safe_load(list(Path(__file__).parent.parent.glob("SECRETS/codes.yaml"))[0].read_text())
    DISCORD_TOKEN = d['apikey']
    HUB_KEY = d['inferkey']
    main(DISCORD_TOKEN, HUB_KEY)
