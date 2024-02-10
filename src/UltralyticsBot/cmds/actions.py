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
from UltralyticsBot.utils.general import ReqImage, attach_file, files_age
from UltralyticsBot.utils.plotting import nxy2xy, xcycwh2xyxy, rel_line_size, draw_all_boxes
from UltralyticsBot.utils.msgs import IMG_ERR_MSG, API_ERR_MSG, NOT_OWNER, gen_line, ReqMessage, ResponseMsg, NEWLINE
from UltralyticsBot.cmds.client import MyClient

TEMPFILE = 'detect_res.png' # fallback
LIMITS = {k:app_commands.Range[type(v['min']), v['min'], v['max']] for k,v in REQ_LIM.items()}
ACTIVITIES = {ki:k for ki,k in enumerate(['Reset', 'Playing', 'Streaming', 'Listening', 'Watching', 'Custom', 'Competing'],-1)}
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

def fetch_embed(embeds:dict, topic:str, sub_topic:str) -> discord.Embed:
    """Simply returns value for keys provided."""
    return embeds[topic][sub_topic]

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
@app_commands.describe(
        img_url='Valid HTTP/S link to a supported image type.',
        show="Enable/disable showing annotated result image.",
        conf="Confidence threshold for class predictions.",
        iou="Intersection over union threshold for detections.",
        size="Inference image size (single dimension only).",
        model="One of YOLOv(5|8)(n|s|m|l|x) models to use for inference."
)
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
                if req.status_code != 200: # Catch all other non-good return codes and make sure to reply
                    Loggr.debug(f"{API_ERR_MSG.format(req.status_code, req.reason)}")
                    await interaction.followup.send(API_ERR_MSG.format(req.status_code, req.reason))
                
                else:
                    Reply = ResponseMsg(req, show, True, infer_ratio)
                    file, text = Reply.start_msg(partial(process_result, img=infer_im, plot=show, class_pad=Reply.cls_pad), infer_ratio=infer_ratio)
                    file = attach_file(file) if show else None
        
            except requests.HTTPError as e:
                Loggr.debug(f"Error during request: {e} with response {req.status_code} - {req.reason}")
                await interaction.followup.send(API_ERR_MSG.format(req.status_code, req.reason))

            except Exception as e:
                Loggr.debug(f"Error during request: {e} with response {req.status_code} - {req.reason}")
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

###-----DEV COMMANDS-----###

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

# DEVMsgs class
## TODO use to replace if/elif logic in bot.py

class DEVMsgs:
    def __init__(self, devID:int) -> None:
        self.devID = devID
        self.is_dev = False
    
    def init_cmds(self, cmd_list:list[str]=CMDS['DevMsgs']) -> None:
        self.cmds = [c for c in cmd_list if c.startswith('$')]
    
    def verify(self) -> str|None:
        self.is_dev = self.devID == self.mesg.author.id
        return self.is_dev

    async def fire_cmd(self, cmd:str, client:MyClient, msg:discord.Message, *args, **kwargs):
        """Calls class method with same name as `cmd` argument, expected to use `$` as first character."""
        self.mesg = msg
        if self.verify():
            await getattr(self, cmd.lower().strip('$'))(client, *args, **kwargs) if cmd.lower() in self.cmds else None
        else:
            await self.mesg.reply(f"## That command is for bot owner only.")
    
    async def newstatus(self, client:MyClient, *args, **kwargs):
        """Updates bot displayed activity."""
        status, response = chng_status(self.mesg)
        await client.change_presence(activity=status)
        await self.mesg.reply(response)

    async def cmd_sync(self, client:MyClient, *args, **kwargs):
        """Attempts to sync commands either to server command was sent from or to all servers."""
        target, *_ = self.mesg.guild.name if not any(args) else list(args)
        guild = self.mesg.guild if not any(args) else None
        Loggr.info(f"Commands sync for {target} server(s).")
        try:
            syncd = await client.tree.sync(guild=guild)
            await self.mesg.reply(f"Synced commands {(NEWLINE+'- ').join([c.name for c in syncd])}")
        except Exception as e:
            Loggr.error(f"Exception {e} while syncing commands for {target}{f' with ID {guild.id}' if guild else ''}.")
            await self.mesg.reply(f"Error {e} occured while syncing, please open Issue at {GH} and include your Server-ID.")

    async def rm_cmd(self, client:MyClient, *args, **kwargs):
        """Attempts to remove command from server command was sent from or from all servers if `None`."""
        arg, *_ = args if any(args) else ''
        guild = self.mesg.guild
        removed = await client.tree.remove_command(command=arg.lower(), guild=guild)
        response = f"Removed ${removed.name} from {guild.name}" if removed else f"No command with name {arg.lower()}"
        await self.mesg.reply(response)
    
    async def add_cmd(self, client:MyClient, *args, **kwargs):
        """Attempts to add command from server command was sent from, and must be `synced` for it to update."""
        arg, *_ = args if any(args) else ''
        guild = self.mesg.guild
        try:
            await client.tree.add_command(command=arg.lower(), guild=guild)
            response = f"Added ${arg.lower()} to {guild.name}."
        except app_commands.CommandAlreadyRegistered:
            response = f"Command ${arg.lower()} registered in {guild.name} - {guild.id} already."
        except Exception as e:
            Loggr.error(f"Error {e} encountered when attempting to add ${arg.lower()} to {guild.name} - {guild.id}.")
            response = f"Command ${arg.lower()} is either incorrect or {guild.name} - {guild.id} is at command limit."
        finally:
            await self.mesg.reply(response)

###-------------------------------------------------------------------------------------------------###