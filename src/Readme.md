# Running UltralyticsBot as a service on Linux

# Intro

Following the steps below will run the `bot.py` code as a (background) service for Linux. This should be configured and run _after_ verifying that [bot.py](/src/bot.py) is configured and working correctly. Make sure to propagate all changes for your specific configuration throughout the process. This was adapted from [this gist](https://gist.github.com/emxsys/a507f3cad928e66f6410e7ac28e2990f) by @emxsys and verified to work on RPi3b running Debian 11 (bullseye).

## 1. Create Service File

### Navigate to

`cd /lib/systemd/system/`

### Run command

`sudo nano discordbot.service`

- Use your text editor of choice

### Add below into discordbot.service

- This example will create a file named `discordbot.service`

    - You may rename as you see fit, `{PREFERED_NAME}.service` but ensure to replace all references in subsequent steps with the name you choose

```toml
[Unit]
Description=Ultralytics Discord Bot
After=multi-user.target

[Service]
Type=simple

# Replace "/path/to/.env/" and "/path/to/.../src/bot.py" with appropriate directories
ExecStart=/path/to/.env/bin/python /path/to/Ultralytics_DiscordBot/src/bot.py
Restart=on-abort

[Install]
WantedBy=multi-user.target
```

- Save changes and close text editor

## 2. Run commands

### Update permissions

```bash
# Owner [read/write], Group [read], Public [read] Permissions
>>> sudo chmod 644 /lib/systemd/system/discordbot.service
```

### Make script executible

```bash
>>> chmod +x /path/to/Ultralytics_DiscordBot/src/bot.py # use same location from .service file
```

### Reload daemon, enable service, start

```bash
>>> sudo systemctl daemon-reload # run after any changes made to project code
>>> sudo systemctl enable discordbot.service
>>> sudo systemctl start discordbot.service
```

## Service Interactions

### Check bot-service status
`sudo systemctl status discordbot.service`

### Start bot-service
`sudo systemctl start discordbot.service`

### Stop bot-service
`sudo systemctl stop discordbot.service`

### Check bot-service log
`sudo journalctl -f -u discordbot.service`


