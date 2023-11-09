"""
Title: client.py
Author: Burhan Qaddoumi
Date: 2023-10-29

Requires: discord.py
"""
import datetime

import discord
from discord import app_commands
from discord.ext import tasks

from UltralyticsBot import CMDS, DEV_CH
from UltralyticsBot.utils.logging import Loggr
from UltralyticsBot.utils.docs_data import docs_choices, load_docs_cache

RUN_AT = datetime.time(hour=0, minute=0, second=0, tzinfo=datetime.timezone.utc) # time to refresh repo and docs

class MyClient(discord.Client):
    """Class for Discord application/bot with slash-commands, requires message content intents"""
    def __init__(self, *, intents:discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.docs_choices, self.docs_embeds = load_docs_cache()
        self.cmd_pop()
    
    async def setup(self):
        """Sync commands, could take upto an hour to show up when bot is in lots of servers"""
        Loggr.info(f"Intitating client sync.")
        await self.tree.sync()
    
    async def setup_hook(self) -> None:
        # return await super().setup_hook()
        self.docs_update.start()
    
    @tasks.loop(time=RUN_AT)
    async def docs_update(self):
        """Task loop to update Documentation commands."""
        notice_ch = self.get_channel(DEV_CH)
        Loggr.info(f"Running scheduled docs command.")
        self.docs_choices, self.docs_embeds = docs_choices()
        await notice_ch.send(content=f"Docs update task completed.")

    @docs_update.before_loop
    async def before_my_task(self):
        await self.wait_until_ready()
    
    def cmd_pop(self, cmds:dict=CMDS):
        """Populate client with commands from YAML file."""
        Loggr.info(f"Populating commands to client.")
        for k,v in cmds['Global'].items():
            setattr(self, 'GLOBAL_'+k, self.tree.command(name=k, description=v['description']))

        for k,v in cmds['Dev'].items():
            setattr(self, 'DEV_'+k, self.tree.command(name=k, description=v['description']))
        
        Loggr.info(f"All commands populated to client.")

# intnt = discord.Intents.default()
# intnt.message_content = True
# client = MyClient(intents=intnt)
# client.cmd_pop()
