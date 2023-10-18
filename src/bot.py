"""
Title: bot.py
Author: Burhan Qaddoumi
Date: 2023-10-10

Requires: discord.py, pyyaml, numpy, requests, opencv-python
"""
import re
import io
import base64
from pathlib import Path

import cv2 as cv
import discord
import numpy as np
import requests
import yaml
from discord import app_commands

from UltralyticsBot import PROJ_ROOT
from UltralyticsBot.utils.plotting import nxy2xy, xcycwh2xyxy, draw_all_boxes, select_color
from UltralyticsBot.utils.logging import Loggr
from UltralyticsBot.utils.general import dec2str, is_link, model_chk, gen_cmd

# References
# Discord slash-command: https://stackoverflow.com/questions/71165431/how-do-i-make-a-working-slash-command-in-discord-py
# Discord library docs: https://discordpy.readthedocs.io/en/stable/
# Dicord library examples: https://github.com/Rapptz/discord.py/tree/v2.3.2/examples

REQ_CFG = yaml.safe_load((PROJ_ROOT / 'cfg/req.yaml').read_text())

ASSETS = PROJ_ROOT /'assets' # NOTE future, use this as fallback when no image is provided
DEFAULT_INFER = REQ_CFG['default']
REQ_ENDPOINT = REQ_CFG['endpoint']
RESPONSE_KEYS = tuple(REQ_CFG['response'])
TEMPFILE = 'detect_res.png' # fallback

# NOTE unverified bots in < 100 servers will be able to use message content intents, once above 100 servers, bot needs to be verified
class MyClient(discord.Client):
    """Class for Discord slash-commands, requires message content intents"""
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup(self):
        """Sync commands, could take upto an hour to show up when bot is in lots of servers"""
        await self.tree.sync()

def get_values(data:dict) -> list:
    """Return values for known response keys"""
    return [data[k] for k in RESPONSE_KEYS]

def cleanup() -> None:
    """Deletes temporary image file after finished."""
    _ = Path(TEMPFILE).unlink(missing_ok=True)

def attach_file(img_data:np.ndarray, encode:str='.png', name:str='detections') -> discord.File:
    """Generates Discord message file attachment from numpy image array."""
    encode = encode if encode.startswith('.') else ('.' + encode)

    try:
        img_attachmnt = discord.File(io.BytesIO(cv.imencode(encode, img_data)[1]), f'{name}{encode}')
        Loggr.info("Attached from memory")
    
    except:
        _ = cv.imwrite(TEMPFILE, img_data)
        img_attachmnt = discord.File(Path(TEMPFILE))
        Loggr.info("Attached from disk")

    return img_attachmnt

def plot_result(imgbytes:bytes, predictions:list, include_msg:bool=False) -> tuple[np.ndarray, None|str]:
    # Load image
    img = cv.imdecode(np.frombuffer(imgbytes, np.uint8), -1)
    imH, imW = img.shape[:2]
    anno_img = np.copy(img)
    
    msg = f'Detections:\n'
    msg = '```' # start monospacing
    msg += f'''{'class'.ljust(10)} {'conf'.ljust(4)}   {'x1y1x2y2'.ljust(25)}\n''' # Title, intentional extra whitespace
    
    pred_boxes = np.zeros((1,5)) # x-center, y-center, width, height, class
    # NOTE maybe better to convert bbox coordinates before generating message?
    for p in predictions:
        cls_name, conf, idx, *(x, y, w, h) = get_values(p)
        x1, y1, x2, y2 = tuple(nxy2xy(xcycwh2xyxy(np.array((x, y, w, h))), imH, imW)) # n-xcycwh -> x1y1x2y2
        pred_boxes = np.vstack([pred_boxes, np.array((x1, y1, x2, y2, idx))])
        
        msg += f'{cls_name.ljust(10)} {dec2str(conf)} {str((x1, y1, x2, y2)).rjust(24)}\n'
    
    msg += '```' # end monospacing
    anno_img = draw_all_boxes(anno_img, pred_boxes[1:])

    return (anno_img, None) if not include_msg else (anno_img, msg)

def reply_msg(response:requests.models.Response, 
              plot:bool=False,
              req_img:bytes=None
              ) -> tuple[str | None, discord.File] | tuple[str, None]:
    
    plot = (req_img is not None or req_img != '') and plot

    pred, reply, success = response.json().values()
    msg = f'''{reply}\n'''

    if (success or response.status_code == 200) and plot:
        result, msg = plot_result(req_img, pred, True)
        box_img = attach_file(result)
        out = (msg, box_img)

    elif success and not plot:
        msg += '```'
        msg += f'''{'class'.ljust(10)} {'conf'.ljust(4)}   {'nxywh'.ljust(25)}\n'''
        for p in pred:
            cls_name, conf, idx, *(x, y, w, h) = get_values(p)
            # NOTE normalized (x,y,w,h) bounding boxes since image not loaded
            msg += f'''{cls_name.ljust(10)} {round(conf,3)} {str(tuple(conf(v) for v in (x, y, w, h))).rjust(24)}\n'''
        
        msg += '```'
        out = (msg, None)

    else:
        out = (f"Error: API call failed with {response.status_code} - {response.reason}", None)
    
    return out

def main(T,H):
    intents = discord.Intents.default()
    intents.message_content = True
    client = MyClient(intents=intents)

    @client.event
    async def on_ready():
        await client.tree.sync()
        print("Initialized sync")
        Loggr.info("Initialized client sync")

    # Message starting with `$predict` uses default inference settings
    @client.event
    async def on_message(message:discord.Message):
        
        if message.content.startswith("$predict"):
            image_url = message.content.strip('$predict ') # assume only image URL is passed
            if image_url == '' and len(message.attachments) > 0:
                Loggr.debug(message.attachments[0].url) # TESTING
                _ = [Loggr.debug(a.url) for a in message.attachments] # TESTING
                image_url = message.attachements[0].url # TODO figure why this isn't working
            request_url = REQ_ENDPOINT
            image_data = requests.get(image_url).content
            
            req_dict = DEFAULT_INFER.copy()
            req_dict['key'] = H
            req_dict['image'] = base64.b64encode(image_data).decode()

            try:
                response = requests.post(request_url, json=req_dict)
            except Exception as e:
                Loggr.error(f"Error during request: {e}")
                await message.channel.send("Error: API request failed")

            if response.status_code == 200:
                preds, reply, success = response.json().values()
                result, _ = plot_result(image_data, preds, False)
                
                box_img = attach_file(result)

                await message.reply("Detections", file=box_img)
                # await message.delete(delay=30) # FUTURE
            
            else:
                _, reply, success = response.json().values()
                Loggr.error(f"Response code {response.status_code} with reply {reply} and reason {response.reason}")
                await message.channel.send("Error: API request failed")

    # Slash-command for predict, allows for keyword parameters
    @client.tree.command(name='predict')
    @app_commands.describe(
        img_url="REQUIRED: Full URL to image",
        conf="OPTIONAL: Confidence threshold for object classification",
        iou="OPTIONAL: Intersection over union threshold for object detection",
        size="OPTIONAL: Largest image dimension, single number only",
        model="OPTIONAL: One of 'yolov5(n|m|l|x)' or 'yolov8(n|m|l|x)'; EXAMPLE: yolov5n",
        show="OPTIONAL: Display image results with annotations."
    )
    async def im_predict(interaction:discord.Interaction,
                         img_url:str='',
                         conf:float=0.3,
                         iou:float=0.4,
                         size:int=640,
                         model:str='yolov8n',
                         show:bool=False
                         ):
        await interaction.response.defer(thinking=True) # permits longer response time
        
        model = model_chk(model)
        linklike = is_link(img_url)
        size = size if isinstance(size, int) else 640
        conf = conf if isinstance(conf, float) and 0 < conf < 1.0 else 0.3
        iou = iou if isinstance(iou, float) and 0 < iou < 1.0 else 0.4

        if linklike:
            image_data = requests.get(img_url).content
            request_dict = {
                    "confidence": str(conf),
                    "iou": str(float(iou)),
                    "size": str(int(size)),
                    "model": str(model),
                    "key": str(H),
                    "image": base64.b64encode(image_data).decode(),
                    }
            try:
                req = requests.post(REQ_ENDPOINT, json=request_dict)
                
            except Exception as e:
                Loggr.error(f"Error during request: {e}")
                await interaction.response.send_message(f"Error: API request failed due to {e}")

            text, file = reply_msg(req, show, image_data)
            await (interaction.followup.send(content=text, file=file) if file is not None else interaction.followup.send(content=text))

        else:
            await interaction.followup.send("That link was..._weird_.")

    client.run(T)

if __name__ == '__main__':
    d = yaml.safe_load((PROJ_ROOT / 'SECRETS/codes.yaml').read_text())
    DISCORD_TOKEN = d['apikey']
    HUB_KEY = d['inferkey']

    main(DISCORD_TOKEN, HUB_KEY)
