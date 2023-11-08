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

import yaml
import discord
from discord import app_commands

from UltralyticsBot import BOT_ID, REPO_DIR
from UltralyticsBot.utils.logging import Loggr

MD_LINK_RGX = r"\#+\W\[\w+\]\((h|H)ttp(s)?://.*\)" # For headers specifically

DOCS_URL = "https://docs.ultralytics.com/"
GH_REPO = "https://github.com/ultralytics/ultralytics.git"
ULTRA_LICENSING = "https://www.ultralytics.com/license"
LICENSE = "AGPL-3.0"

DOCS_DIR = "docs"
DOCS_IDX = "mkdocs" # mkdocs.yml
YAML_EXT = ['.yaml', '.yml']
LOCAL_DOCS = REPO_DIR if any(REPO_DIR) else "repo_data" # Directory name for local documentation files

LOGO_ICON = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics-logomark-color.png"
INTGR8_BANNER = "https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-integrations.png"
BGRD_LOGO = "https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-ultralytics-github.png"
FULL_LOGO = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics-logotype-color.png"
YOLO_LOGO = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/discord/emote-Ultralytics_YOLO_Logomark.png"

CATEGORIES = ['Modes', 'Tasks', 'Models', 'Datasets', 'Guides', 'Integrations']

def md_index_2link(mdtxt:str, base_link:str=DOCS_URL):
    """Constructs links from markdown header sections and base URL string."""
    base_link = base_link if base_link.endswith('/') else base_link + '/'
    return base_link + '#' + ''.join([c for c in mdtxt.strip('# ').lower() if c not in string.punctuation]).replace(' ','-')

def delist_dict(in_obj:list, out:dict=None):
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

def get_subcat_files(cat_path:Path):
    """Fetch sub-category doc-files, these are expected to be found at a depth of one (1)."""
    return [f for f in cat_path.rglob("*.md") if f.stem != 'index']

def get_dataset_files(ds_path:Path):
    """Fetch Dataset doc-files, these are nested inside directories and should be `index.md` files."""
    tasks = [p for p in ds_path.iterdir() if p.is_dir()]
    return [next(task.glob("index.md")) for task in tasks]

def no_header_links(md_header:str):
    """Removes Markdown Header links and only returns header text."""
    return md_header.split(']')[0].replace('[', '') if re.search(MD_LINK_RGX, md_header) else md_header

def get_md_headers(md_content:list):
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
        cmd = ['git', 'merge']
    else:
        cmd = ['git', 'clone', repo]
    # proc_run = subprocess.run(cmd, cwd=save_path, capture_output=True, text=True) # "Cloning into 'ultralytics'...\n", from `.stderr`, not certain how to capture more; `returncode == 0` should be successful
    proc_run = subprocess.call(cmd, cwd=save_path, text=True) # blocking
    save_path = save_path / repo_name # update for output
    return save_path, proc_run

def yaml_2_embeds(file:str|Path) -> tuple[dict,dict]:
    """Reads YAML file and generates `discord.Embeds` and `discord.app_choices.Choice` objects. Output order is `choices, embeds` both as dictionaries. If YAML file doesn't have correct name, will raise a generic `Exception`."""
    file = Path(file)
    category = file.stem.capitalize() if file.stem.capitalize() in CATEGORIES else None

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

def docs_choices(to_file:bool=False) -> tuple[dict, dict]|None:
    """Fetches data from repo and crawls the Docs files for generating links to pages+sections of the Docs as Discord Embeds. First dictionary are the `discord.app_command.Choices` and the second include the `discord.Embed` objects."""
    Loggr.info(f"Fetching data from {GH_REPO} for documentation.")
    into_path, run_result = fetch_gh_docs()
    #TODO raise run_result.check_returncode() # Raises CalledProcessError
    # REFERENCE https://docs.python.org/3.9/library/subprocess.html#subprocess.CalledProcessError

    # Read MKDOCS index
    docs_idx = [f for f in [(into_path / DOCS_IDX).with_suffix(y) for y in YAML_EXT] if f.exists()]
    Loggr.info(f"Searching for documentation index file in {into_path.as_posix()}")
    assert any(docs_idx), f"Unable to locate mkdocs index file in {into_path.as_posix()} repo directory."
    
    text_data = docs_idx[0].read_text('utf-8').splitlines()
    text_data = [s for s in text_data if '!!' not in s]
    
    docs_layout = yaml.safe_load('\n'.join(text_data))['nav'] # list
    docs = delist_dict(docs_layout)
    Loggr.info(f"Documentation sections found are: {[k for k in docs]} and kept only {CATEGORIES} for populating commands.")
    
    # Try using custom embeds instead
    options = {C:{} for C in CATEGORIES}
    
    for k,v in docs.items():
        
        if k in CATEGORIES:
            category_path = (into_path / DOCS_DIR / k.lower())
            files = get_subcat_files(category_path) if k.lower() != 'datasets' else get_dataset_files(category_path)
            
            for f in files:
                SUB_CAT = f.as_posix().partition(k.lower())[-1].replace('.md','')
                SUB_CAT = '/' + [s for s in SUB_CAT.split('/') if s != ''][0] # formatting
                base_URL = DOCS_URL + k.lower() + SUB_CAT.lower()
                
                # if entry not in options:
                if SUB_CAT.strip(string.punctuation).capitalize() not in options[k]:
                    # Get subsections
                    TITLE, *TOC = get_md_headers(f.read_text('utf-8').splitlines())
                    TITLE = TITLE.strip('# ').title()
                    
                    embed = discord.Embed(title=TITLE,
                                        colour=15665350, # pink-ish, looked okay
                                        url=base_URL,)
                    _ = embed.set_image(url=FULL_LOGO)
                    _ = embed.set_thumbnail(url=LOGO_ICON)
                    
                    for si,section in enumerate(TOC, 1):
                        section_name = section.strip('# ').title()
                        section_link = md_index_2link(section, base_URL)
                        _ = embed.set_author(name="UltralyticsBot")
                        _ = embed.add_field(name=section_name, value=f"[Go to section]({section_link})", inline=False) # NOTE inline fields get smooshed and look bad, don't use
                        _ = embed.set_footer(text=f"{LICENSE} or Ultralytics Enterprise Licensing {ULTRA_LICENSING}\n", icon_url=YOLO_LOGO)
                    
                    _ = options[k].update({SUB_CAT.strip(string.punctuation).capitalize():embed})
    
    # Output to file
    ## NOTE app_command.choices dictionary is made when loading YAML file with yaml_2_embeds()
    if to_file:
        for k,v in options.items():
            embeds_file = into_path.parent / f'{k}.yaml'
            _ = embeds_file.write_text(yaml.safe_dump({kk:vv.to_dict() for kk,vv in v.items()}),encoding='utf-8')
    
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
