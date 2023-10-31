
import discord
from discord import Intents, app_commands

from UltralyticsBot import BOT_TOKEN, OWNER_ID, DEV_GUILD
from UltralyticsBot.cmds.client import MyClient
from UltralyticsBot.cmds.actions import msg_predict, im_predict, chng_status, ACTIVITIES, about, commands, help, slash_example, msgexample
from UltralyticsBot.utils.logging import Loggr
from UltralyticsBot.utils.msgs import NOT_OWNER

def main():
    intent = discord.Intents.default()
    intent.message_content = True
    client = MyClient(intents=intent)
    client.cmd_pop()

    @client.event
    async def on_ready():
        await client.tree.sync()
        # print("Initialized sync")
        Loggr.info("Initialized client sync")
    
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