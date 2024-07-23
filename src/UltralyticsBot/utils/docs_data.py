"""
Title: docs_data.py
Author: Burhan Qaddoumi
Date: 2023-11-01

Requires: discord.py, pyyaml
"""
import re
import string
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
# from typing import Any, Coroutine

import yaml
import discord
import requests
from discord import app_commands

from UltralyticsBot import BOT_ID, REPO_DIR
from UltralyticsBot.utils.logging import Loggr
from UltralyticsBot.utils.config import (
    DOCS_CFG,
    DOCS_URL,
    GH_REPO,
    ULTRA_LICENSING,
    LICENSE,
    DOCS_DIR,
    DOCS_LOC,
    DOCS_IDX,
    YAML_EXT,
    BRAND,
    LOGO_ICON,
    INTGR8_BANNER,
    BGRD_LOGO,
    FULL_LOGO,
    YOLO_LOGO,
    CATEGORIES,
    ALL_CAPS,
    IGNORE,
)

MD_LINK_RGX = re.compile("\#+\W\[\w+\]\((h|H)ttp(s)?://.*\)") # For headers specifically
YOLOvN_RGX = re.compile('(yolo)(v)?\d?') # include re.IGNORECASE
YOLO_RGX = re.compile('(yolo)')

LOCAL_DOCS = REPO_DIR if any(REPO_DIR) else "repo_data" # Directory name for local documentation files

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


def fetch_sitemap(sitemap_url:str="http://docs.ultralytics.com/sitemap.xml") -> list[str]:
    """Fetches and parses the sitemap XML, returning a list of URLs."""
    web_urls = []
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()  # Check that the request was successful
        sitemap_xml = response.content
        root = ET.fromstring(sitemap_xml)
        namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        web_urls = [url.text for url in root.findall("sitemap:url/sitemap:loc", namespace)]
    except requests.RequestException as e:
        Loggr.error(f"Error fetching sitemap: {e}")
    return web_urls


def get_md_headers(md_content:list) -> list[str]:
    """Gets Markdown headers text, ignoring code-block comment lines"""
    headers = {k:v for k,v in enumerate(md_content) if v.startswith('#')}
    codeblcks = [k for k,v in enumerate(md_content) if v.startswith('```')]
    code_idx = list(zip(codeblcks[::2],codeblcks[1::2]))
    return [no_header_links(ht) for h,ht in headers.items() if not any([c[0] < h < c[1] for c in code_idx])]


def fetch_gh_docs(repo:str=GH_REPO, local_docs:str=LOCAL_DOCS) -> tuple[Path, subprocess.CompletedProcess]:
    """Fetch docs from repo; defaults are Ultralytics Repo and `Path.home() / repo_data` respectively."""
    save_path = Path.home() / local_docs
    # into_path = Path.home() / 'python_proj/yolo3.9/ultralytics' # NOTE TESTING ONLY
    save_path.mkdir() if not save_path.exists() else None
    repo_name = repo.strip('.git').split("/")[-1]
    if (save_path / repo_name).exists():
        cmd = ['git', 'pull']
        save_path = save_path / repo_name
    else:
        cmd = ['git', 'clone', repo]
    proc_run = subprocess.call(cmd, cwd=save_path.as_posix(), text=True) # blocking
    save_path = save_path / repo_name if save_path.name != repo_name else save_path # update for output
    return save_path, proc_run


def yaml_2_embeds(file:str|Path) -> tuple[dict,dict]:
    """Reads YAML file and generates `discord.Embeds` and `discord.app_choices.Choice` objects. Output order is `choices, embeds` both as dictionaries. If YAML file doesn't have correct name, will raise a generic `Exception`."""
    file = Path(file)
    category = brand_format(file.stem.capitalize()) if brand_format(file.stem.capitalize()) in CATEGORIES else None

    if category:
        data = yaml.safe_load(file.read_text('utf-8'))
        options = list()
        embeds, opts = dict(), dict()
        for k,v in data.items():
            embeds.update({k:discord.Embed.from_dict(v)})
            options.append(app_commands.Choice(name=k, value=k))
        
        embeds_out = {category:embeds}
        opts = {category:options}

        return opts, embeds_out
    
    elif category is None:
        raise Exception(f"No Docs category named matching {file.as_posix()}")


def load_docs_cache(docs_path:Path=(Path.home() / LOCAL_DOCS)) -> tuple[dict,dict]:
    """Loads data from the path where local repo is cloned and assumes YAML cache has been created."""
    choices, embeds = {c:{} for c in CATEGORIES}, {c:{} for c in CATEGORIES}
    for yfile in docs_path.glob("*.yaml"):
        o_c, o_e = yaml_2_embeds(yfile)
        _ = choices.update(o_c)
        _ = embeds.update(o_e)
        
    return choices, embeds


def sub_guides(d:dict) -> dict:
    """Extract Tutorial and Guides under the primary Guides key."""
    keep = {}
    for k,v in d.items():
        if k.lower() in ["guides", "real-world projects", "tutorials",]:
            keep.update(**v)
        elif k.lower() in ["yolov5"]:
            keep.update(**v["Tutorials"])
    return keep


def top_level(all_urls:list[str]) -> set[str]:
    """Returns the top level of all URLs."""
    ignore = set([e.lower() for e in IGNORE])
    return set([urlparse(url).path.split('/')[1] for url in all_urls]).difference(ignore)


def walk_path(path:Path) -> dict[str,Path]:
    """Walks the path and returns all files and directories."""
    paths = {}
    for f in path.iterdir():
        if f.is_dir():
            paths.update(walk_path(f))
        else:
            paths.update({f.parent.name:[f]}) if not paths.get(f.parent.name) else paths.get(f.parent.name).append(f)
    return paths


def yolov5_tutorials(d:dict) -> dict:
    """Extract YOLOv5 Tutorials from the dictionary."""
    return d.get("YOLOv5").get("Tutorials")


def docs_choices(to_file:bool=False) -> tuple[dict, dict]|None:
    """Fetches data from repo and crawls the Docs files for generating links to pages+sections of the Docs as Discord Embeds. First dictionary are the `discord.app_command.Choices` and the second include the `discord.Embed` objects."""
    global CATEGORIES
    Loggr.info(f"Fetching data from {GH_REPO} for documentation.")
    into_path, run_result = fetch_gh_docs()
    urls = fetch_sitemap()
    CATEGORIES = ({brand_format(t.capitalize()) for t in top_level(urls)} | set(CATEGORIES)) - set(IGNORE)
    #TODO raise run_result.check_returncode() # Raises CalledProcessError
    # REFERENCE https://docs.python.org/3.9/library/subprocess.html#subprocess.CalledProcessError

    # Read MKDOCS index
    # docs_idx = [f for f in [(into_path / DOCS_DIR / DOCS_IDX).with_suffix(y) for y in YAML_EXT] if f.exists()]
    docs_idx = [f for f in [(into_path / DOCS_IDX).with_suffix(y) for y in YAML_EXT] if f.exists()]
    Loggr.info(f"Searching for documentation index file in {into_path.as_posix()}")
    assert any(docs_idx), f"Unable to locate mkdocs index file in {into_path.as_posix()} repo directory."
    
    text_data = docs_idx[0].read_text('utf-8').splitlines()
    text_data = [s for s in text_data if '!!' not in s]
    
    docs_layout = yaml.safe_load('\n'.join(text_data))['nav'] # list
    docs = delist_dict(docs_layout)
    _ = [docs.update({k:docs.get("Guides").pop(k)}) for k in {"YOLOv5", "Tutorials"} if docs.get("Guides").get(k) is not None]
    Loggr.info(f"Documentation sections found are: {[k for k in docs]} and kept only {CATEGORIES} for populating commands.")
    
    # Try using custom embeds instead
    options = {C:{} for C in CATEGORIES}
    
    for k,v in docs.items():
        
        if k in CATEGORIES:
            category_path = (into_path / DOCS_DIR / DOCS_LOC / k.lower())
            files = get_subcat_files(category_path) if k.lower() != 'datasets' else get_dataset_files(category_path)
            
            for f in files[:25]:  # limit to 25 entries (max for choices)
                SUB_CAT = f.as_posix().partition(k.lower())[-1].replace('.md','')
                SUB_CAT = '/' + [s for s in SUB_CAT.split('/') if s != ''][0] # formatting
                base_URL = DOCS_URL + k.lower() + SUB_CAT.lower()
                
                # if entry not in options:
                if brand_format(SUB_CAT.strip(string.punctuation).capitalize()) not in options[k]:
                    # Get subsections
                    TITLE, *TOC = get_md_headers(f.read_text('utf-8').splitlines())
                    stop = TOC.index("## FAQ") if "## FAQ" in TOC else None  # avoid FAQ section
                    TOC = TOC[:stop]
                    TITLE = brand_format(TITLE.strip('# '))
                    
                    embed = discord.Embed(title=TITLE,
                                        colour=15665350, # pink-ish, looked okay
                                        url=base_URL,)
                    _ = embed.set_image(url=FULL_LOGO)
                    _ = embed.set_thumbnail(url=LOGO_ICON)
                    
                    for si,section in enumerate(TOC, 1):
                        section_name = allcapwords(section.strip('# ').title().replace("’S", "'s"))
                        section_link = md_index_2link(section, base_URL)
                        _ = embed.set_author(name="UltralyticsBot")
                        _ = embed.add_field(name=section_name, value=f"[Go to section]({section_link})", inline=False) # NOTE inline fields get smooshed and look bad, don't use
                        _ = embed.set_footer(text=f"{LICENSE} or Ultralytics Enterprise Licensing {ULTRA_LICENSING}\n", icon_url=YOLO_LOGO)
                    
                    _ = options[k].update({brand_format(SUB_CAT.strip(string.punctuation).capitalize()):embed})
    
    # Output to file
    ## NOTE app_command.choices dictionary is made when loading YAML file with yaml_2_embeds()
    if to_file:
        for k,v in options.items():
            embeds_file = into_path.parent / f'{k}.yaml'
            _ = embeds_file.write_text(yaml.safe_dump({kk:vv.to_dict() for kk,vv in v.items()}, allow_unicode=True), encoding='utf-8')
    
    # Generate dictionary for use with app_commands.choices
    else:
        opts_d = dict()
        for k,v in options.items():
            opts_d.update({k:[app_commands.Choice(name=kk, value=kk) for kk in v]})
    
        return opts_d, options
    # app_commands.choices(**opts_d) # NOTE this might work as @decorator

if __name__ == '__main__()':
    docs_choices()

# ###------REFERENCE------###
# embed = discord.Embed(
#     title=...,
#     description=...,
#     timestamp=...,
#     colour=...,
#     url=...,
# )
# embed.add_field(name=..., value=..., inline=False)
# # await bot.say(embed=embed)
# dict(
#     title=...,
#     description=...,
#     url=...,
#     color=...,
#     timestamp=...,
#     footer=dict(text=...,icon_url=...),
#     fields=[
#         dict(name=..., value=...),
#         dict(name=...,value=...,inline=...,),
#         ]
# )
# discord.Colour()

# ###------------------------###
