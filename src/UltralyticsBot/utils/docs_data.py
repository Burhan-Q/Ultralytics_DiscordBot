"""
Title: docs_data.py
Author: Burhan Qaddoumi
Date: 2023-11-01

Requires: discord.py, pyyaml, xml.etree.ElementTree
"""
import re
import string
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml
import discord
import requests
from discord import app_commands

from UltralyticsBot import BOT_ID, REPO_DIR
from UltralyticsBot.utils.logging import Loggr

MD_LINK_RGX = r"\#+\W\[\w+\]\((h|H)ttp(s)?://.*\)" # For headers specifically
YOLOvN_RGX = r'(yolo)(v)?\d?' # include re.IGNORECASE
YOLO_RGX = r'(yolo)'

DOCS_URL = "https://docs.ultralytics.com/"
GH_REPO = "https://github.com/ultralytics/ultralytics.git"
ULTRA_LICENSING = "https://www.ultralytics.com/license"
LICENSE = "AGPL-3.0"

DOCS_DIR = "docs"
DOCS_LOC = "en" # english locale
DOCS_IDX = "mkdocs" # mkdocs.yml
YAML_EXT = ['.yaml', '.yml']
LOCAL_DOCS = REPO_DIR if any(REPO_DIR) else "repo_data" # Directory name for local documentation files
BRAND = {'hub':'HUB', 'yolo':'YOLO', 'ultralytics':'Ultralytics'}

LOGO_ICON = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics-logomark-color.png"
INTGR8_BANNER = "https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-integrations.png"
BGRD_LOGO = "https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-ultralytics-github.png"
FULL_LOGO = "https://github.com/Burhan-Q/Ultralytics_DiscordBot/assets/62214284/ec6ef857-72b1-407b-b078-b2c3e8e34df0"
YOLO_LOGO = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/discord/emote-Ultralytics_YOLO_Logomark.png"

CATEGORIES = ['Modes', 'Tasks', 'Models', 'Datasets', 'Guides', 'YOLOv5', 'HUB', 'Integrations', 'Help'] # 'NEW 🚀 Explorer'
ALL_CAPS = ['YOLO', 'CLI', 'JSON', 'YAML', 'HUB', 'API', 'URL', 'OBB', 'TCP', 'RTSP', 'ONNX', 'TF.JS', 'TF', 'NCNN', 'CNN', 'COCO']

def brand_format(text:str) -> str:
    """Ensures correct text formatting of Ultralytics Branding."""
    txt_parts = [i.span() for i in [re.search(rf'({k})', text, re.IGNORECASE) for k in BRAND] if i is not None]
    txt_out = text
    for w in txt_parts:
        txt_out = txt_out.replace(text[w[0]:w[1]], BRAND[text[w[0]:w[1]].lower()])
    return txt_out

def allcapwords(text:str) -> str:
    """Converts words that should be shown with all caps from title-case to all-caps."""
    for a in ALL_CAPS:
        text = text.replace(a.title(), a)
    return text

def md_index_2link(mdtxt:str, base_link:str=DOCS_URL) -> str:
    """Constructs links from markdown header sections and base URL string."""
    base_link = base_link if base_link.endswith('/') else base_link + '/'
    return base_link + '#' + ''.join([c for c in mdtxt.strip('# ').lower() if c not in string.punctuation]).replace(' ','-')

def delist_dict(in_obj:list, out:dict=None) -> dict:
    """Creates nested dictionaries if dictionaries contain list of dictionaries."""
    out = out if out is not None else dict()
    if isinstance(in_obj, list):
        _ = [out.update(delist_dict(x)) for x in in_obj]
    elif isinstance(in_obj, str):
        pass
    elif isinstance(in_obj, dict):
        for k,v in in_obj.items():
            out.update({k:delist_dict(v)} if isinstance(v, list) else {k:v})
    return out

def get_subcat_files(cat_path:Path) -> list[Path]:
    """Fetch sub-category doc-files, these are expected to be found at a depth of one (1)."""
    return [f for f in cat_path.rglob("*.md") if f.stem != 'index']

def get_dataset_files(ds_path:Path) -> list[Path]:
    """Fetch Dataset doc-files, these are nested inside directories and should be `index.md` files."""
    tasks = [p for p in ds_path.iterdir() if p.is_dir()]
    return [next(task.glob("index.md")) for task in tasks]

def no_header_links(md_header:str) -> str:
    """Removes Markdown Header links and only returns header text."""
    return md_header.split(']')[0].replace('[', '') if re.search(MD_LINK_RGX, md_header) else md_header

def get_md_headers(md_content:list) -> list[str]:
    """Gets Markdown headers text, ignoring code-block comment lines"""
    headers = {k:v for k,v in enumerate(md_content) if v.startswith('#')}
    codeblcks = [k for k,v in enumerate(md_content) if v.startswith('```')]
    code_idx = list(zip(codeblcks[::2],codeblcks[1::2]))
    return [no_header_links(ht) for h,ht in headers.items() if not any([c[0] < h < c[1] for c in code_idx])]

def fetch_sitemap(sitemap_url:str="http://docs.ultralytics.com/sitemap.xml") -> list[str]:
    """Fetches and parses the sitemap XML, returning a list of URLs."""
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()  # Check that the request was successful
        sitemap_xml = response.content
        root = ET.fromstring(sitemap_xml)
        namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [url.text for url in root.findall("sitemap:url/sitemap:loc", namespace)]
        return urls
    except requests.RequestException as e:
        Loggr.error(f"Error fetching sitemap: {e}")
        return []

def docs_choices(to_file:bool=False) -> tuple[dict, dict]|None:
    """Fetches data from sitemap and crawls the Docs pages for generating links to pages of the Docs as Discord Embeds."""
    urls = fetch_sitemap()
    # Filter URLs to include only desired pages
    filtered_urls = [url for url in urls if "docs" in url and not any(exclude in url for exclude in ["api", "models"])]
    options = {C:{} for C in CATEGORIES}
    for url in filtered_urls:
        # Extract category and subcategory from URL
        match = re.search(r"https?://docs\.ultralytics\.com/(.+?)/(.+?)/?", url)
        if match:
            category, subcategory = match.groups()
            if category.capitalize() in CATEGORIES:
                # Generate embed for the URL
                embed = discord.Embed(title=subcategory.replace("-", " ").title(), url=url, color=0x00ff00)
                embed.set_author(name="Ultralytics Documentation", url=DOCS_URL, icon_url=LOGO_ICON)
                embed.set_footer(text="Ultralytics", icon_url=LOGO_ICON)
                # Ensure title length is less than 25 characters
                if len(subcategory) > 25:
                    subcategory = subcategory[:22] + "..."
                options[category.capitalize()][subcategory] = embed

    # Output to file
    if to_file:
        for k,v in options.items():
            embeds_file = Path(f'{k}.yaml')
            _ = embeds_file.write_text(yaml.safe_dump({kk:vv.to_dict() for kk,vv in v.items()}, allow_unicode=True),encoding='utf-8')
    else:
        opts_d = dict()
        for k,v in options.items():
            opts_d.update({k:[app_commands.Choice(name=kk, value=kk) for kk in v]})
        return opts_d, options

# Validate URLs in the sitemap
def validate_sitemap_urls(urls: list[str]) -> bool:
    """Validates that all URLs in the sitemap are reachable and return a 200 status code."""
    for url in urls:
        try:
            response = requests.head(url)
            if response.status_code != 200:
                Loggr.error(f"URL validation failed for {url}: Status code {response.status_code}")
                return False
        except requests.RequestException as e:
            Loggr.error(f"Error validating URL {url}: {e}")
            return False
    return True

if __name__ == '__main__':
    urls = fetch_sitemap()
    if validate_sitemap_urls(urls):
        Loggr.info("All sitemap URLs are valid.")
    else:
        Loggr.error("Some sitemap URLs are invalid.")
