"""
Title: client.py
Author: Burhan Qaddoumi
Date: 2023-10-29

Requires: discord.py
"""

import discord
from discord import app_commands

from UltralyticsBot import CMDS

class MyClient(discord.Client):
    """Class for Discord application/bot with slash-commands, requires message content intents"""
    def __init__(self, *, intents:discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
    
    async def setup(self):
        """Sync commands, could take upto an hour to show up when bot is in lots of servers"""
        await self.tree.sync()
    
    def cmd_pop(self, cmds:dict=CMDS):
        """Populate client with commands from YAML file."""
        for k,v in cmds['Global'].items():
            setattr(self, 'GLOBAL_'+k, self.tree.command(name=k, description=v['description']))
        
        for k,v in cmds['Dev'].items():
            setattr(self, 'DEV_'+k, self.tree.command(name=k, description=v['description']))

intnt = discord.Intents.default()
intnt.message_content = True
client = MyClient(intents=intnt)
client.cmd_pop()