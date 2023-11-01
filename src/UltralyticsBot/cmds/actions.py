"""
Title: commands.py
Author: Burhan Qaddoumi
Date: 2023-10-29

Requires: discord.py, pyyaml, numpy, requests, opencv-python
"""

import base64
from functools import partial

import discord
import requests
import numpy as np
from discord import app_commands

from UltralyticsBot import REQ_LIM, REQ_ENDPOINT, CMDS, RESPONSE_KEYS, HUB_KEY, DEFAULT_INFER, BOT_ID, OWNER_ID, GH, MAX_REQ
from UltralyticsBot.utils.checks import model_chk
from UltralyticsBot.utils.logging import Loggr
from UltralyticsBot.utils.general import ReqImage, ReqMessage, attach_file, dec2str, align_boxcoord
from UltralyticsBot.utils.plotting import nxy2xy, xcycwh2xyxy, rel_line_size, draw_all_boxes
from UltralyticsBot.utils.msgs import IMG_ERR_MSG, API_ERR_MSG, NOT_OWNER, gen_line, ResponseMsg
from UltralyticsBot.cmds.client import MyClient

TEMPFILE = 'detect_res.png' # fallback
LIMITS = {k:app_commands.Range[type(v['min']), v['min'], v['max']] for k,v in REQ_LIM.items()}
ACTIVITIES = {ki:k for ki,k in enumerate(['Reset', 'Playing', 'Streaming', 'Lisenting', 'Watching', 'Custom', 'Competing'],-1)}
iACTIVITIES = {k:ki for ki,k in enumerate(['unknown','game','stream','listen','watch','custom','competing'],-1)}

###-----SUPPORT FUNCTIONS-----###

def get_values(data:dict) -> list:
    """Return values for known response keys"""
    return [data[k] for k in RESPONSE_KEYS]

def inference_req(imgbytes:bytes, req2:str=REQ_ENDPOINT, **kwargs) -> requests.Response:
    """Constructs JSON (as dictionary) request using image-bytes data and endpoint, will update request JSON (dictionary) with any values from `kwargs` if keywords are found in `DEFAULT_INFER` dictionary."""
    req_dict = DEFAULT_INFER.copy()
    if any(kwargs):
        for k in kwargs:
            _ = req_dict.update({k:kwargs[k]}) if k in req_dict else None
    req_dict['key'] = HUB_KEY
    req_dict['image'] = base64.b64encode(imgbytes).decode()
    return requests.post(req2, json=req_dict)
    # return req_dict # NOTE might need to change in future

def process_result(img:np.ndarray, predictions:list, plot:bool, class_pad:int) -> tuple[np.ndarray, str]:
    # Load image
    imH, imW = img.shape[:2]
    anno_img = np.copy(img)
    line_size = rel_line_size(imH,imW)
    
    msg = ''
    pred_boxes = np.zeros((1,5)) # x-center, y-center, width, height, class
    for p in predictions:
        cls_name, conf, idx, *(x, y, w, h) = get_values(p)
        x1, y1, x2, y2 = tuple(nxy2xy(xcycwh2xyxy(np.array((x, y, w, h))), imH, imW)) # n-xcycwh -> x1y1x2y2
        pred_boxes = np.vstack([pred_boxes, np.array((x1, y1, x2, y2, idx))])
        msg += gen_line(cls_name, class_pad, conf, x1, y1, x2, y2)
    
    if plot:
        anno_img = draw_all_boxes(anno_img, pred_boxes[1:], line_size)
    
    return (anno_img, msg)

###-----GLOBAL COMMANDS-----###

# Message Command
async def msg_predict(message:discord.Message):
    
    if message.content.startswith("$predict") or (BOT_ID in [m.id for m in message.mentions]):
        text = file = None
        msg = ReqMessage(message)
        image_url = msg.get_url()
        imH, imW, imSize = msg.media_info()

        image = ReqImage(image_url, height=imH, width=imW, size=imSize)
        if not image.image_error:
        # infer_im, infer_data, infer_ratio = image.inference_img()

            try:
            # if not image.image_error:
                infer_im, infer_data, infer_ratio = image.inference_img()
                req = inference_req(infer_data, req2=REQ_ENDPOINT)
                # req_d = inference_req(infer_data, req2=REQ_ENDPOINT) # NOTE may need to check request size
                # req_sz = len(str(req_d).encode('utf-8'))
                # if req_sz >= MAX_REQ:
                #     reduce = (MAX_REQ - 100) / req_sz
                #     infer_im, infer_data, infer_ratio = image.inference_img(Q=reduce)
                req.raise_for_status()
                Reply = ResponseMsg(req, True, False, infer_ratio)
                file, text = Reply.start_msg(partial(process_result, img=infer_im, plot=True, class_pad=Reply.cls_pad), infer_ratio=infer_ratio)
                file = attach_file(file)
        
            except requests.HTTPError:
                Loggr.error(API_ERR_MSG.format(req.status_code, req.reason))
                await message.reply(API_ERR_MSG.format(req.status_code, req.reason))

            except Exception as e:
                Loggr.error(f"Error during request: {e} with response {req.status_code} - {req.reason}")
                await message.reply(API_ERR_MSG.format(e, ' '.join(['and', req.status_code, req.reason])))
        
        else:
            req = file = None
            text = IMG_ERR_MSG
            Loggr.debug(f"Issue fetching image from URL {image_url}")

        text = text if text is not None or text != '' else f"{Reply.message}\n"
        
        await message.reply(text, file=file)

###-----Slash Commands-----###

async def im_predict(interaction:discord.Interaction,
                         img_url:str,
                         show:bool=True,
                         conf:LIMITS['conf']=0.35,
                         iou:LIMITS['iou']=0.45,
                         size:LIMITS['size']=640,
                         model:str='yolov8n',
                         ):
        await interaction.response.defer(thinking=True) # permits longer response time
        
        model = model_chk(model)
        image = ReqImage(img_url)
        # infer_im, infer_data, infer_ratio = image.inference_img(int(size))
        if not image.image_error:
        
            try:
            # if not image.image_error:
                infer_im, infer_data, infer_ratio = image.inference_img(int(size))
                req = inference_req(infer_data, req2=REQ_ENDPOINT, confidence=str(conf), iou=str(iou), size=str(size), model=str(model))
                req.raise_for_status()
                Reply = ResponseMsg(req, show, True, infer_ratio)
                file, text = Reply.start_msg(partial(process_result, img=infer_im, plot=show, class_pad=Reply.cls_pad), infer_ratio=infer_ratio)
                file = attach_file(file) if show else None
        
            except requests.HTTPError:
                Loggr.error(f"Error during request: {e} with response {req.status_code} - {req.reason}")
                await interaction.followup.send(API_ERR_MSG.format(req.status_code, req.reason))

            except Exception as e:
                Loggr.error(f"Error during request: {e} with response {req.status_code} - {req.reason}")
                await interaction.followup.send(API_ERR_MSG.format(e, ' '.join(['and', req.status_code, req.reason])))
        
        else:
            req = file = None
            text = IMG_ERR_MSG
            Loggr.debug(f"Issue fetching image from URL {img_url}")
        
        if file is None:
            await interaction.followup.send(content=text)
        else:
            await interaction.followup.send(content=text, file=file)

async def about(interaction:discord.Interaction):
    msg = CMDS['Global']['about']['content']
    await interaction.response.send_message(content=msg, suppress_embeds=True)

async def commands(interaction:discord.Interaction):
    msg = CMDS['Global']['commands']['content']
    await interaction.response.send_message(content=msg, suppress_embeds=True)

async def help(interaction:discord.Interaction):
    msg = CMDS['Global']['help']['content']
    await interaction.response.send_message(content=msg, suppress_embeds=True)
    
async def slash_example(interaction:discord.Interaction):
    msg = CMDS['Global']['slashexample']['content']
    await interaction.response.send_message(content=msg, suppress_embeds=True)

async def msgexample(interaction:discord.Interaction):
    msg = CMDS['Global']['msgexample']['content']
    await interaction.response.send_message(content=msg, suppress_embeds=True)

def fetch_embed(embeds:dict, topic:str, sub_topic:str) -> discord.Embed:
    """Simply returns value for keys provided."""
    return embeds[topic][sub_topic]


###-----OWNER COMMANDS-----###

# Change Activity Status

def chng_status(message:discord.Message):
    if message.author.id == OWNER_ID and message.content.startswith("$newstatus"):
        *_, activity, name = message.content.split(',')
        new_activity = iACTIVITIES[activity.strip().lower()] # integer
        status = discord.Activity(type=new_activity, name=name.strip())

        msg = f"Bot is now set to {ACTIVITIES[new_activity]} {name.capitalize()}."
        Loggr.info(f"Bot status updated. Now {activity.capitalize()} {name}")
    
    elif message.author.id != OWNER_ID:
        status = -2
        msg = f"This command is only for the Bot owner."
    
    return (status, msg)

# 

###-------------------------------------------------------------------------------------------------###