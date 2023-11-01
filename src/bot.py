"""
Title: bot.py
Author: Burhan Qaddoumi
Date: 2023-10-10

Requires: discord.py, pyyaml, numpy, requests, opencv-python
"""
import discord
from discord import app_commands

from UltralyticsBot import BOT_TOKEN, OWNER_ID, DEV_GUILD
from UltralyticsBot.cmds.client import MyClient
from UltralyticsBot.cmds.actions import msg_predict, im_predict, chng_status, ACTIVITIES, about, commands, help, slash_example, msgexample, fetch_embed
from UltralyticsBot.utils.logging import Loggr
from UltralyticsBot.utils.msgs import NOT_OWNER
from UltralyticsBot.utils.docs_data import docs_choices

def main():
    doc_choices, doc_embeds = docs_choices()
    Loggr.info("Finished fetching docs.")

    intent = discord.Intents.default()
    intent.message_content = True
    client = MyClient(intents=intent)
    client.cmd_pop()

    # @client.event
    # async def on_interaction(interaction:discord.Interaction):
    #     Loggr.info(f"Commands synced for {interaction.guild.name} on event.")
    #     await client.tree.sync(guild=interaction.guild)
    
    @client.event
    async def on_ready():
        Loggr.info("Initialized client sync")
        await client.tree.sync()

    @client.event # NOTE
    async def on_message(message:discord.Message):
        
        # DEV Command
        if message.author.id == OWNER_ID and message.content.startswith("$newstatus"): 
            status, msg = chng_status(message)
            await client.change_presence(activity=status)
            await message.reply(msg)
        
        # GLOBAL Command
        elif message.content.startswith("$predict"):
            await msg_predict(message)
    
    # Slash-Commands
    client.GLOBAL_predict(im_predict)

    client.GLOBAL_about(about)

    client.GLOBAL_commands(commands)

    client.GLOBAL_help(help)

    client.GLOBAL_slashexample(slash_example)

    client.GLOBAL_msgexample(msgexample)
    
    @client.GLOBAL_docs_tasks
    @app_commands.choices(sub_section=doc_choices['Tasks'])
    @app_commands.describe(
        sub_section="Task Documentation Subsection to generate embedding for.",
        user="Username who should be mentioned in the response with embed.",
        )
    async def docs_tasks(interaction:discord.Interaction, sub_section:app_commands.Choice[str],user:str=None):
        # await interaction.response.defer(thinking=True) # NOTE if additional time needed to respond
        
        # doc_embed:discord.Embed = doc_embeds['Tasks'][sub_section.name]
        doc_embed = fetch_embed(doc_embeds, 'Tasks', sub_section.name)
        mention = user if user is not None else ''
        await interaction.response.send_message(content=mention, embed=doc_embed)
        # await interaction.followup.send(content=mention, embed=doc_embed)

    @client.GLOBAL_docs_modes
    @app_commands.choices(sub_section=doc_choices['Modes'])
    @app_commands.describe(
        sub_section="Modes Documentation Subsection to generate embedding for.",
        user="Username who should be mentioned in the response with embed.",
        )
    async def docs_modes(interaction:discord.Interaction,
                         sub_section:app_commands.Choice[str],
                         user:str=None,
                         ):
        # await interaction.response.defer(thinking=True) # NOTE if additional time needed to respond
        
        # doc_embed:discord.Embed = doc_embeds['Modes'][sub_section.name]
        doc_embed = fetch_embed(doc_embeds, 'Modes', sub_section.name)
        mention = user if user is not None else ''
        await interaction.response.send_message(content=mention, embed=doc_embed)
        # await interaction.followup.send(content=mention, embed=doc_embed)

    @client.GLOBAL_docs_models
    @app_commands.choices(sub_section=doc_choices['Models'])
    @app_commands.describe(
        sub_section="Models Documentation Subsection to generate embedding for.",
        user="Username who should be mentioned in the response with embed.",
        )
    async def docs_models(interaction:discord.Interaction, sub_section:app_commands.Choice[str],user:str=None):
        # await interaction.response.defer(thinking=True) # NOTE if additional time needed to respond
        
        # doc_embed:discord.Embed = doc_embeds['Models'][sub_section.name]
        doc_embed = fetch_embed(doc_embeds, 'Models', sub_section.name)
        mention = user if user is not None else ''
        await interaction.response.send_message(content=mention, embed=doc_embed)
        # await interaction.followup.send(content=mention, embed=doc_embed)
    
    @client.GLOBAL_docs_datasets
    @app_commands.choices(sub_section=doc_choices['Datasets'])
    @app_commands.describe(
        sub_section="Datasets Documentation Subsection to generate embedding for.",
        user="Username who should be mentioned in the response with embed.",
        )
    async def docs_datasets(interaction:discord.Interaction, sub_section:app_commands.Choice[str],user:str=None):
        # await interaction.response.defer(thinking=True) # NOTE if additional time needed to respond
        
        # doc_embed:discord.Embed = doc_embeds['Datasets'][sub_section.name]
        doc_embed = fetch_embed(doc_embeds, 'Datasets', sub_section.name)
        mention = user if user is not None else ''
        await interaction.response.send_message(content=mention, embed=doc_embed)
        # await interaction.followup.send(content=mention, embed=doc_embed)

    @client.GLOBAL_docs_guides
    @app_commands.choices(sub_section=doc_choices['Guides'])
    @app_commands.describe(
        sub_section="Guides Documentation Subsection to generate embedding for.",
        user="Username who should be mentioned in the response with embed.",
        )
    async def docs_guides(interaction:discord.Interaction, sub_section:app_commands.Choice[str],user:str=None):
        # await interaction.response.defer(thinking=True) # NOTE if additional time needed to respond
        
        # doc_embed:discord.Embed = doc_embeds['Guides'][sub_section.name]
        doc_embed = fetch_embed(doc_embeds, 'Guides', sub_section.name)
        mention = user if user is not None else ''
        await interaction.response.send_message(content=mention, embed=doc_embed)
        # await interaction.followup.send(content=mention, embed=doc_embed)
    
    @client.GLOBAL_docs_integrations
    @app_commands.choices(sub_section=doc_choices['Integrations'])
    @app_commands.describe(
        sub_section="Integrations Documentation Subsection to generate embedding for.",
        user="Username who should be mentioned in the response with embed.",
        )
    async def docs_integrations(interaction:discord.Interaction, sub_section:app_commands.Choice[str],user:str=None):
        # await interaction.response.defer(thinking=True) # NOTE if additional time needed to respond
        
        # doc_embed:discord.Embed = doc_embeds['Integrations'][sub_section.name]
        doc_embed = fetch_embed(doc_embeds, 'Integrations', sub_section.name)
        mention = user if user is not None else ''
        await interaction.response.send_message(content=mention, embed=doc_embed)
        # await interaction.followup.send(content=mention, embed=doc_embed)

    @client.DEV_status_change
    @app_commands.choices(
        category=[
            app_commands.Choice(**discord.ActivityType.playing._asdict()),
            app_commands.Choice(**discord.ActivityType.streaming._asdict()),
            app_commands.Choice(**discord.ActivityType.watching._asdict()),
            app_commands.Choice(**discord.ActivityType.listening._asdict()),
            app_commands.Choice(**discord.ActivityType.custom._asdict()),
            app_commands.Choice(**discord.ActivityType.competing._asdict()),
        ]
    )
    @app_commands.describe(
            category="One of: 'game', 'stream', 'custom', 'watch', or 'listen'.",
            name="Name of Game/Content for activity.",
    )
    async def new_activity(interaction:discord.Interaction,
                        category:app_commands.Choice[int],
                        name:str):
        await interaction.response.defer(thinking=True)

        if interaction.user.id == OWNER_ID:
            activity = discord.Activity(type=category.value, name=name)
            msg = f"Now {ACTIVITIES[category.value]} {name}."
            Loggr.info(f"Updating bot status. {msg}")

        else:
            activity = None
            msg = NOT_OWNER

        if activity:
            await client.change_presence(activity=activity)
        await interaction.followup.send(content=msg)

    client.run(BOT_TOKEN)

if __name__ == '__main__':
    main()