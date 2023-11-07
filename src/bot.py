"""
Title: bot.py
Author: Burhan Qaddoumi
Date: 2023-10-10

Requires: discord.py, pyyaml, numpy, requests, opencv-python
"""
import discord
from discord import app_commands

from UltralyticsBot import BOT_TOKEN, OWNER_ID, DEV_GUILD, BOT_ID
from UltralyticsBot.cmds.client import MyClient
from UltralyticsBot.cmds.actions import msg_predict, im_predict, chng_status, ACTIVITIES, about, commands, help, slash_example, msgexample, fetch_embed
from UltralyticsBot.utils.logging import Loggr
from UltralyticsBot.utils.msgs import NOT_OWNER, NEWLINE, get_args
from UltralyticsBot.utils.docs_data import docs_choices

def main():
    doc_choices, doc_embeds = docs_choices()
    Loggr.info("Finished fetching docs.")

    intent = discord.Intents.default()
    intent.message_content = True
    client = MyClient(intents=intent)
    # client.setup() # NOTE lets to Rate Limiting (especially when testing)
    # client.cmd_pop() # NOTE included with class init method
    
    @client.event
    async def on_ready():
        Loggr.info("Client is ready, sync must be run manually.")
        # Loggr.info("Initialized client sync")
        # await client.tree.sync() # NOTE lets to Rate Limiting (especially when testing)

    @client.event
    async def on_message(message:discord.Message):
        author, guild, content, mentions = [getattr(message, a) for a in ['author', 'guild', 'content', 'mentions']]
        is_owner = author.id == OWNER_ID
        bot_mention = any([b.id == BOT_ID for b in mentions])
        args = get_args(content)

        # DEV Command
        ## TODO use actions.DEVMsgs to replace message logic here.
        if is_owner and content.startswith("$newstatus"): 
            status, msg = chng_status(message)
            await client.change_presence(activity=status)
            await message.reply(msg)

        elif is_owner and content.startswith("$cmd_sync"):
            if not any(args):
                Loggr.info(f"Syncing the client for {guild}.")
                try:
                    # [client.tree.add_command(c, guild=guild) for c in client.tree.get_commands()]
                    syncd_cmds = await client.tree.sync(guild=guild)
                    Loggr.info(f"Commands synced {[c.name for c in syncd_cmds]}")
                    await message.reply(f"Commands synced {(NEWLINE + '- ').join([c.name for c in syncd_cmds])} for this server.")
                except Exception as e:
                    Loggr.error(f"Syncing exception {e}")
            
            elif 'all' in [a.lower() for a in args]:
                Loggr.info(f"Executing sync/setup for commands across all guilds.")
                try:
                    syncd_cmds = await client.tree.sync()
                    Loggr.info(f"Syncing commands: {[c.name for c in syncd_cmds]} to all servers.")
                    await message.reply(f"The following commands were fetched: {[c.name for c in syncd_cmds]} for all servers.")
                except Exception as e:
                    Loggr.error(f"Syncing exception {e}")

        elif is_owner and content.startswith("$rm_cmd"):
            cmd = await client.tree.remove_command(command=args[0].lower(), guild=guild)
            await message.reply(f"Removed the {cmd.name} commands from server {guild.name}") if cmd is not None else await message.reply(f"No command with name {args[0].lower()}.")

        elif is_owner and content.startswith("$add_cmd"):
            await client.tree.add_command(command=args[0].lower(), guild=guild)
            await message.reply(f"Removed the {args} command from server {guild.name}")
        
        # GLOBAL Command
        elif content.startswith("$predict") or bot_mention:
            await msg_predict(message)

        elif content.startswith("$docs"):
            section = doc_embeds.get(args[0], None)
            sub_sect = section.get(args[1]) if section is not None else None
            res = sub_sect if section and sub_sect else None
            await message.reply(embed=res) if isinstance(res, discord.Embed) else await message.reply("Couldn't find that!")
    
    # Slash-Commands
    client.GLOBAL_predict(im_predict)

    client.GLOBAL_about(about)

    client.GLOBAL_commands(commands)

    client.GLOBAL_help(help)

    client.GLOBAL_slashexample(slash_example)

    client.GLOBAL_msgexample(msgexample)

    # TODO decide if this should stay
    # @client.tree.command(name="getcommands", description="Gets list of commands")
    # async def getcommands(inter:discord.Interaction):
    #     await inter.response.send_message('\n'.join([c.name for c in client.tree.get_commands(guild=inter.guild)]))
    
    @client.GLOBAL_docs_tasks
    @app_commands.choices(sub_section=list(doc_choices['Tasks']))
    @app_commands.describe(sub_section="Task Documentation Subsection to generate embedding for.",user="Username who should be mentioned in the response with embed.")
    async def docs_tasks(interaction:discord.Interaction, sub_section:app_commands.Choice[str],user:str=None):
        doc_embed = doc_embeds['Tasks'][sub_section.name]
        mention = user if user is not None else ''
        await interaction.response.send_message(content=mention, embed=doc_embed)
 
    @client.GLOBAL_docs_modes
    @app_commands.choices(sub_section=doc_choices['Modes'])
    @app_commands.describe(sub_section="Modes Documentation Subsection to generate embedding for.",user="Username who should be mentioned in the response with embed.")
    async def docs_modes(interaction:discord.Interaction,
                         sub_section:app_commands.Choice[str],
                         user:str=None,
                         ):
        doc_embed = doc_embeds['Modes'][sub_section.name]
        mention = user if user is not None else ''
        await interaction.response.send_message(content=mention, embed=doc_embed)

    @client.GLOBAL_docs_models
    @app_commands.choices(sub_section=doc_choices['Models'])
    @app_commands.describe(sub_section="Models Documentation Subsection to generate embedding for.",user="Username who should be mentioned in the response with embed.",)
    async def docs_models(interaction:discord.Interaction, sub_section:app_commands.Choice[str],user:str=None):
        doc_embed:discord.Embed = doc_embeds['Models'][sub_section.name]
        mention = user if user is not None else ''
        await interaction.response.send_message(content=mention, embed=doc_embed)
    
    @client.GLOBAL_docs_datasets
    @app_commands.choices(sub_section=doc_choices['Datasets'])
    @app_commands.describe(sub_section="Datasets Documentation Subsection to generate embedding for.",user="Username who should be mentioned in the response with embed.",)
    async def docs_datasets(interaction:discord.Interaction, sub_section:app_commands.Choice[str],user:str=None):
        doc_embed:discord.Embed = doc_embeds['Datasets'][sub_section.name]
        mention = user if user is not None else ''
        await interaction.response.send_message(content=mention, embed=doc_embed)

    @client.GLOBAL_docs_guides
    @app_commands.choices(sub_section=doc_choices['Guides'])
    @app_commands.describe(sub_section="Guides Documentation Subsection to generate embedding for.",user="Username who should be mentioned in the response with embed.",)
    async def docs_guides(interaction:discord.Interaction, sub_section:app_commands.Choice[str],user:str=None):
       doc_embed:discord.Embed = doc_embeds['Guides'][sub_section.name]
       mention = user if user is not None else ''
       await interaction.response.send_message(content=mention, embed=doc_embed)
    
    @client.GLOBAL_docs_integrations
    @app_commands.choices(sub_section=doc_choices['Integrations'])
    @app_commands.describe(sub_section="Integrations Documentation Subsection to generate embedding for.",user="Username who should be mentioned in the response with embed.",)
    async def docs_integrations(interaction:discord.Interaction, sub_section:app_commands.Choice[str],user:str=None):
        doc_embed:discord.Embed = doc_embeds['Integrations'][sub_section.name]
        mention = user if user is not None else ''
        await interaction.response.send_message(content=mention, embed=doc_embed)

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
    
    # @client.DEV_cmd_sync
    # async def cmd_sync(interaction:discord.Interaction):
    #     Loggr.info(f"Syncing commands to {interaction.guild.name} on demand.")
    #     await client.tree.sync(guild=interaction.guild)

    discord_Loggr = Loggr.handlers[0] # StreamHandler
    client.run(BOT_TOKEN, log_formatter=discord_Loggr.formatter, log_handler=discord_Loggr, log_level=10) # logging.DEBUG=10

if __name__ == '__main__':
    main()