"""
Title: bot.py
Author: Burhan Qaddoumi
Date: 2023-10-10

Requires: discord.py, pyyaml, numpy, requests, opencv-python
"""
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
from UltralyticsBot.utils.plotting import nxy2xy, xcycwh2xyxy, draw_all_boxes, select_color, rel_line_size
from UltralyticsBot.utils.logging import Loggr
from UltralyticsBot.utils.general import dec2str, is_link, model_chk, gen_cmd, req_values, float_str, align_boxcoord, ReqImage

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
GH = "https://github.com/Burhan-Q/Ultralytics_DiscordBot"

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

def inference_req(imgbytes:bytes, req2:str=REQ_ENDPOINT, **kwargs) -> requests.Response:
    """Constructs JSON (as dictionary) request using image-bytes data and endpoint, will update request JSON (dictionary) with any values from `kwargs` if keywords are found in `DEFAULT_INFER` dictionary."""
    req_dict = DEFAULT_INFER.copy()
    if any(kwargs):
        for k in kwargs:
            _ = req_dict.update({k:kwargs[k]}) if k in req_dict else None
    req_dict['key'] = HUB_KEY
    req_dict['image'] = base64.b64encode(imgbytes).decode()
    return requests.post(req2, json=req_dict)

def attach_file(img:np.ndarray, encode:str='.png', name:str='detections') -> discord.File:
    """Generates Discord message file attachment from numpy image array."""
    encode = encode if encode.startswith('.') else ('.' + encode)
    
    try:
        img_attachmnt = discord.File(io.BytesIO(cv.imencode(encode, img)[1]), f'{name}{encode}')
        Loggr.info("Attached from memory")
    
    except:
        _ = cv.imwrite(TEMPFILE, img)
        img_attachmnt = discord.File(Path(TEMPFILE))
        Loggr.info("Attached from disk")
        cleanup() # NOTE unsure if this will cause errors here
    
    return img_attachmnt

def plot_result(img:np.ndarray, predictions:list, include_msg:bool=False) -> tuple[np.ndarray, str]:
    # Load image
    imH, imW = img.shape[:2]
    anno_img = np.copy(img)
    line_size = rel_line_size(imH,imW)
    
    if include_msg:
        msg = '```\n' # start monospacing
        msg += f'''{'class'.ljust(10)} {'conf'.ljust(4)}   {'x1y1x2y2'.ljust(25)}\n''' # Title, intentional extra whitespace
    
    pred_boxes = np.zeros((1,5)) # x-center, y-center, width, height, class
    for p in predictions:
        cls_name, conf, idx, *(x, y, w, h) = get_values(p)
        x1, y1, x2, y2 = tuple(nxy2xy(xcycwh2xyxy(np.array((x, y, w, h))), imH, imW)) # n-xcycwh -> x1y1x2y2
        pred_boxes = np.vstack([pred_boxes, np.array((x1, y1, x2, y2, idx))])
        if include_msg:
            msg += f'{cls_name.ljust(10)} {dec2str(conf)}  {align_boxcoord([x1,y1,x2,y2]).ljust(24)}\n'
    
    anno_img = draw_all_boxes(anno_img, pred_boxes[1:], line_size)
    
    return (anno_img, '') if not include_msg else (anno_img, msg + '```')

def reply_msg(response:requests.models.Response, 
              plot:bool=False,
              txt_results:bool=False,
              req_img:ReqImage=None,
              debug:bool=False,
              ) -> tuple[str | None, discord.File] | tuple[str, None]:
    """Generate reply message from inference request."""
    plot = (plot or not txt_results) and isinstance(req_img.infer_img, np.ndarray)
    pred, reply, success = response.json().values() # NOTE this will raise ERROR if not enough values returned
    msg = f'''{reply}\n'''
    
    # Request good, plotting results with or without text
    if (success or response.status_code == 200) and (plot or not txt_results):
        result, text = plot_result(req_img.infer_img, pred, txt_results)
        msg += text
        box_img = attach_file(result) if not debug else 'DEBUGGING'
        out = (msg, box_img)
    
    # Request good, not plotting results
    elif success and (txt_results or not plot):
        msg += '```\n'
        msg += f'''{'class'.ljust(10)} {'conf'.ljust(4)}   {'nxywh'.ljust(25)}\n'''
        for p in pred:
            cls_name, conf, idx, *(x, y, w, h) = get_values(p)
            # NOTE normalized (x,y,w,h) bounding boxes since image not loaded
            msg += f'''{cls_name.ljust(10)} {dec2str(conf)}  {str(tuple(dec2str(v) for v in (x, y, w, h))).replace("'",'').ljust(24)}\n'''
        
        msg += '```'
        out = (msg, None)
    
    # Request failed
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
                Loggr.info("Inference using Discord message attachment.")
                image_url = message.attachments[0].url
            
            image = ReqImage(image_url)
            image.process()
            try:
                
                if not image.image_error:
                    image_data = image.im2bytes(image.infer_img)
                    req = inference_req(image_data, req2=REQ_ENDPOINT)
                    text, file = reply_msg(req, True, False, image)
                else:
                    req = file = None
                    text = f"Error occured when fetching image, check URL and try again. Open issue and include URL on [project repo]({GH}) if continued problems with working image URL."
                    Loggr.debug(f"Issue fetching image from URL {image_url}")
            
            except Exception as e:
                Loggr.error(f"Error during request: {e}")
                await message.channel.send(f"Error: API request failed due to {e}")
            
            # text, file = reply_msg(req, True, False, image_data)
            text = text if text is not None or text != '' else f"{req.json()['message']}\n"
            
            await message.reply(text, file=file)
    
    # Slash-command for predict, allows for keyword parameters
    @client.tree.command(name='predict', description="Runs inference on image link provided.")
    @app_commands.describe(
        img_url="REQUIRED: Full URL to image",
        conf="OPTIONAL: Confidence threshold for object classification",
        iou="OPTIONAL: Intersection over union threshold for object detection",
        size="OPTIONAL: Largest image dimension, single number only",
        model="OPTIONAL: One of 'yolov5(n|m|l|x)' or 'yolov8(n|m|l|x)'; EXAMPLE: yolov5n",
        show="OPTIONAL: Display image results with annotations."
    )
    async def im_predict(interaction:discord.Interaction,
                         img_url:str,
                         conf:app_commands.Range[float, 0.01, 1.0]=0.35,
                         iou:app_commands.Range[float, 0.1, 0.95]=0.45,
                         size:app_commands.Range[int, 32, 1280]=640,
                         model:str='yolov8n',
                         show:bool=False,
                         ):
        await interaction.response.defer(thinking=True) # permits longer response time
        
        model = model_chk(model)
        size = str(size) if (isinstance(size, int) or str(size).isnumeric()) else DEFAULT_INFER['size']
        conf = str(conf) if (isinstance(conf, float) or float_str(conf)) and 0 < float(conf) < 1.0 else DEFAULT_INFER['confidence']
        iou = str(iou) if (isinstance(iou, float) or float_str(iou)) and 0 < float(iou) < 1.0 else DEFAULT_INFER['iou']
        
        image = ReqImage(img_url)
        image.process()
        image_data = image.im2bytes(image.infer_img)
        
        try:
            if not image.image_error:
                image_data = image.im2bytes(image.infer_img)
                req = inference_req(image_data, req2=REQ_ENDPOINT, confidence=str(conf), iou=str(iou), size=str(size), model=str(model))
                text, file = reply_msg(req, show, True, image)
            else:
                req = file = None
                text = f"Error occured when fetching image, check URL and try again. Open issue and include URL on [project repo]({GH}) if continued problems with working image URL."
                Loggr.debug(f"Issue fetching image from URL {img_url}")
            
        except Exception as e:
            Loggr.error(f"Error during request: {e}")
            await interaction.response.send_message(f"Error: API request failed due to {e}")
            
        await (interaction.followup.send(content=text, file=file) if file is not None else interaction.followup.send(content=text))
    
    client.run(T)

if __name__ == '__main__':
    d = yaml.safe_load((PROJ_ROOT / 'SECRETS/codes.yaml').read_text())
    DISCORD_TOKEN = d['apikey']
    HUB_KEY = d['inferkey']

    main(DISCORD_TOKEN, HUB_KEY)
