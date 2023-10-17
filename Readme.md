# UltralyticsBot for Discord

# Table of Contents

1. [Intro](#introduction)

1. [Repo Structure](#repository-layout)

1. [Setup](#setup-self-host)

    - [Bot setup](#discord-bot-setup)

    - [Ultralytics HUB setup](#ultralytics-hub)

1. [Commands](#commands)

    - [Message Comand Prediction](#message)

    - [Slash Command Prediction](#slash-command)

1. [Known Issues and Limits]()

## Introduction

A Discord bot for object detection inference using [Ultralytics HUB][hub] web API (beta). Initially this bot will only be available on the official [Ultralytics Discord Server][server]. The `UltralyticsBot` will also be self-hosted (running on a device I own), so it may not high-availability to start. This is the first Discord bot I've written and I'll do my best to support feature requests and issues.

<p align="center">
    <img src="/assets/readme/bot_slash-predict_results.png" alt="drawing" width="340"/>
</p>

## Repository layout

```sh
├───cfg
│    └─── colors.yaml # hex color codes for bounding box annotations
│    └─── Loggr.yaml # UltralyticsBot logger config
│    └─── req.yaml # API request information
│
├───SECRETS
│    └─── codes.yaml # Private Ultralytics HUB API key and Bot Token
│
└───src
    └─── bot.py # bot application
    └───UltralyticsBot
        └─── utils
                └─── logging.py
                └─── plotting.py
```

## Setup (self-host)

At present this Discord Bot is only configured to run on a local computer (self-hosted). Interface with Discord is accomplished using [discord.py](https://discordpy.readthedocs.io/en/stable/) for python 3.10.

### Discord Bot Setup

1. Login to the [Discord Developer Portal](https://discord.com/developers/) and select `Applications`.

    !['Discord dev portal sidebar'](/assets/readme/discord_devportal_sidebar.png)

1. On the `Applications` page, select `New Application`.

    !['New Applications'](/assets/readme/discord_devportal_new_app.png)

1. Enter a name for the bot/application and agree to the [Discord API Terms of Service][discord dev tos] and [Developer Policy][discord dev policy].

    !['Naming new application'](/assets/readme/discord_devportal_new_app_name.png)

1. Select the applicaiton/bot and then navigate to the `Bot Settings` section

    !['Bot Sidebar'](/assets/readme/discord_dev_bot_sidebar.png)

1. Enable the following `Privileged Gateway Intents`:

    - MESSAGE CONTENT INTENT

    **NOTE:** Once a bot has joined 100 servers, it will require verification and approval to access these settings, [additional information](https://support.discord.com/hc/en-us/articles/360040720412)

    - Currently `UltralyticsBot` is **not** a Public Bot and does not require OAUTH2 Code Grant.

        - In the future `UltralyticsBot` may become a Public Bot that can be invited to other Discord Servers.

    !['Bot Settings'](/assets/readme/discord_dev_bot_settings.png)

1. Copy the `Bot Token` and add this to [SECRETS/codes.yaml](/SECRETS/codesEXAMPLE.yaml) as the value for `apikey`

    ## ⚠ **WARNING** ⚠
    
    Your `Bot Token` should be protected like a password, if you accidentally share or publish this, make sure to return to the Discord Developer Portal and reset your `Bot Token` as soon as possible.

1. Navigate to the `OAuth2 Settings` and select the sub-section `URL Generator`

    !['OAuth2 URL Generator'](/assets/readme/discord_dev_bot_oauth2_url_gen.png)

1. Select the following `Scopes`

    - `bot`

    !['Bot Scopes'](/assets/readme/discord_devportal_scopes.png)

1. A new section for `Bot Permissions` will open below, select the following `Permissions`:

    - `Send Messages`

    - `Embed Links`

    - `Attach Files`

    - `Read Message History`

    - `Use Slash Commands`

    !['Bot Permissions'](/assets/readme/discord_devportal_bot_permissions.png)

1. Under the `Bot Permissions` section, copy the `Generated URL`, then paste and go-to the link in your browser, which presents a page where you can select the Discord Server you wish to have the bot join.

    - **NOTE:** Your user account for the server must have adequate permissions to join a bot to the server.

### See the [discord.py docs setup instructions](https://discordpy.readthedocs.io/en/stable/discord.html) as additional reference

### Ultralytics HUB

1. Login (or sign up [here][hub]) for an Ultralytics HUB account.

1. TBD

## Commands

`UltralyticsBot` supports object detection via two methods:

### Message

Send a message starting with `$predict` followed by a URL for an image, and `UltralyticsBot` will use the default inference request settings [found here](/cfg/req.yaml) and returns **only** an annotated image result.

#### MESSAGE EXAMPLE:

!['Message-predict preview'](/assets/readme/bot_message-predict.png)

#### RESULT EXAMPLE:

!['Message-predict results'](/assets/readme/bot_message-predict_results.png)

### Slash Command

In the Discord message box, type `/predict` and follow the prompts for adding additional arguments. The only required argument is the `image_url` but the other arguments allow for you to adjust the inference settings.

| Argument |      Description     |          Values         | Required | Notes                                                             |
| :------: | :------------------: | :---------------------: | :------: | :---------------------------------------------------------------- |
|  img_url |   valid image link   |   string URL to image   |    YES   | [Some image links have been found not to work](#image-links)      |
|   conf   | confidence threshold |    0.0 < `conf` < 1.0   | OPTIONAL | default = 0.35, [small values may cause error](#many-predictions) |
|   iou    |    iou threshold     |    0.0 < `iou` < 1.0    | OPTIONAL | default = 0.45, [small values may cause error](#many-predictions) |
|   size   | inference image size |   single integer >= 1   | OPTIONAL | default = 640, [large values may cause errors](#inference-size)   |
|   model  |   inference model    | yolov(5\|8)(n\|m\|l\|x) | OPTIONAL | default = yolov8n, pretrained on [COCO2017][coco dataset]         |
|   show   | show annotated image |      True \| False      | OPTIONAL | default = False (only text results)                               |

#### COMMAND EXAMPLE:

**NOTE** order of arguments has been updated to put `img_url` first

!['Slash-predict preview'](/assets/readme/bot_slash-predict.png)

!['Slash-predict running'](/assets/readme/bot_slash-predict_sending.png)

#### RESULTS EXAMPLE:

!['Slash-predict results'](/assets/readme/bot_slash-predict_results.png)

#### Text results formatting

```
Detections:
class:  {1} conf:   {2} index:  {3} x1y1x2y2: ({4}, {5}, {6}, {7})
```

|  n  | Name     | Value                                    |
| :-: | :------- | :--------------------------------------- |
|  1  | class    | class name/label                         |
|  2  | conf     | prediction confidence                    |
|  3  | index    | class label index                        |
|  4  | x1y1x2y2 | bounding box xmin                        |
|  5  | x1y1x2y2 | bounding box ymin                        |
|  6  | x1y1x2y2 | bounding box xmax                        |
|  7  | x1y1x2y2 | bounding box ymax                        |

---

## Known Limitations and Issues

### Image Links

During testing, _some_ links to images that appeared to be valid did not work correctly with `UltralyticsBot`. This seems to be an issue with either the content provider of the image, or the method which the bot attempts to fetch the image. 

### Many Predictions

Images with a large quantity of objects to detect, especially if using small values for `iou` or `conf` arguments with `/predict`, will generate too much text for a single message. This error may return a message from the bot `Error: API request failed` or may fail silently.

### Inference Size

Specifying a large image size during inference may result in longer inference times. If the inference time is too long, the bot may timeout and not respond at all. Recommendation is to keep the default argument value `size=640` and only increase the inference size if/when absolutely required.

---

[hub]: https://hub.ultralytics.com/signup?utm_source=GitHub&utm_medium=BotReadme
[server]: https://ultralytics.com/discord
[discord dev tos]: https://discord.com/developers/docs/policies-and-agreements/developer-terms-of-service
[discord dev policy]: https://discord.com/developers/docs/policies-and-agreements/developer-policy
[ultra docs]: https://docs.ultralytics.com/
[coco dataset]: https://docs.ultralytics.com/datasets/detect/coco/