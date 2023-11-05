"""
Title: articles.py
Author: Burhan Qaddoumi
Date: 2023-11-04

Requires: discord.py, pyyaml, requests
"""

import datetime
from pathlib import Path
from html.parser import HTMLParser
from html.entities import name2codepoint
from xml.etree import ElementTree as et

import requests
import yaml

class MyHTMLParser(HTMLParser):
    def __init__(self, *, convert_charrefs: bool = True) -> None:
        super().__init__(convert_charrefs=convert_charrefs)
        self.data = list()
        self.fig = list()

    def handle_starttag(self, tag, attrs):
        # print("Start tag:", tag)
        for attr in attrs:
            attr = [a for a in attr if any([a.startswith(r) for r in ['http://', 'https://', 'www.']])]
            _ = self.fig.extend(attr)
            # print("     attr:", attr)

    def handle_endtag(self, tag):
        ...
        # print("End tag  :", tag)

    def handle_data(self, data):
        _ = self.data.append(data)
        # print("Data     :", data)

    def handle_comment(self, data):
        ...
        # print("Comment  :", data)

    def handle_entityref(self, name):
        ...
        # c = chr(name2codepoint[name])
        # print("Named ent:", c)

    def handle_charref(self, name):
        ...
        # if name.startswith('x'):
        #     c = chr(int(name[1:], 16))
        # else:
        #     c = chr(int(name))
        # print("Num ent  :", c)

    def handle_decl(self, data):
        ...
        # print("Decl     :", data)

def get_intro(text:list) -> str:
    """Keeps first few sentences of content as preview, removes any 'Figure' description sentences and undesirable characters."""
    return '. '.join([s for s in ' '.join(text[:2]).split('. ') if 'Fig' not in s]).replace(FILTER, ' ')

def new_discord_post(new_data:dict):
    """When new article data found, generate and send a new Discord post."""
    ...

thrsh = 5
articles = dict()
run_update = False
TODAY = datetime.datetime.now().date()
SAVE_FORMAT = "{}_Medium_Articles.yaml"
SAVE_PATH = Path.home() / 'Medium_Publications'
URL = "https://ultralytics.medium.com/feed" # RSS like feed
FILTER = u'\xa0' # non-breaking space, replace with ' ' character

form = dict(msg=None, title=None, description=None, url=None, image_url=None) # similar to discordEmbed

tags = {'item':
        {'msg':'title', # Use as 'content' for discord.Embed
         'url':'link',
         'author':'{http://purl.org/dc/elements/1.1/}creator',
         'title':'pubDate', # Use publish data for 'title' of discord.Embed
         'content':'{http://purl.org/rss/1.0/modules/content/}encoded'
         }
        }

for iy,y in enumerate(SAVE_PATH.glob(f"*.yaml")):
    last_chk = datetime.date.fromisoformat(y.stem.split('_')[0])
    dT = TODAY - last_chk
    run_update = dT.days >= thrsh

    if run_update:
        previous = yaml.safe_load(y.read_text('utf-8'))
        break

if run_update:
    save_file = (Path.home() / SAVE_FORMAT.format(last_chk.isoformat()))
    
    data = requests.get(URL)
    
    xml = et.XML(data.text)
    CHANNEL = next(xml.iterfind('channel'))

    for n,i in enumerate(CHANNEL.findall('item')):
        parser = MyHTMLParser()
        n_article = form.copy()

        content = {t:next(i.iterfind(v)).text for t,v in tags['item'].items()}
        parser.feed(content['content'])
        
        n_article.update({k:content[k] for k in form.keys() if k in content})
        n_article['description'] = get_intro(parser.data)
        n_article['image_url'] = parser.fig[0]

        articles.update({n:n_article})

        new = articles == previous
        if new:
            f"No new articles found on {TODAY.isoformat()}" if new else f"New content found {TODAY.isoformat()}."
            
            new_discord_post(articles) # TODO finish writing this function

else:
    f"Only {dT.days} passed since previous update cycle, skipping."

