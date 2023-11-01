"""
Title: docs_data.py
Author: Burhan Qaddoumi
Date: 2023-11-01

Requires: discord.py, pyyaml
"""

import string
# import enum
import subprocess
from pathlib import Path
# from dataclasses import dataclass
# from typing import Literal

import yaml
import discord
# from discord import app_commands

from UltralyticsBot.utils.logging import Loggr

DOCS_URL = "https://docs.ultralytics.com/"
GH_REPO = "https://github.com/ultralytics/ultralytics.git"

DOCS_DIR = "docs"
DOCS_IDX = "mkdocs" # mkdocs.yml
YAML_EXT = ['.yaml', '.yml']
LOCAL_DOCS = "repo_data" # Directory name for local documentation files

LOGO_ICON = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics-logomark-color.png"
INTGR8_BANNER = "https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-integrations.png"
BGRD_LOGO = "https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-ultralytics-github.png"
FULL_LOGO = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics-logotype-color.png"

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

def get_md_headers(md_content:list):
    """Gets Markdown headers text, ignoring code-block comment lines"""
    headers = {k:v for k,v in enumerate(md_content) if v.startswith('#')}
    codeblcks = [k for k,v in enumerate(md_content) if v.startswith('```')]
    code_idx = list(zip(codeblcks[::2],codeblcks[1::2]))
    return [ht for h,ht in headers.items() if not any([c[0] < h < c[1] for c in code_idx])]

def fetch_gh_docs(repo:str=GH_REPO, local_docs:str=LOCAL_DOCS) -> tuple[Path, subprocess.CompletedProcess]:
    """Fetch docs from repo; defaults are Ultralytics Repo and `Path.home() / repo_data` respectively."""
    save_path = Path.home() / local_docs
    # into_path = Path.home() / 'python_proj/yolo3.9/ultralytics' # NOTE TESTING ONLY
    save_path.mkdir() if not save_path.exists() else None
    repo_name = repo.strip('.git').split("/")[-1]
    if (save_path / repo_name).exists():
        save_path = save_path / repo_name
        # cmd = ['git', 'fetch', repo]
        cmd = ['git', 'fetch']
    else:
        cmd = ['git', 'clone', repo]
    proc_run = subprocess.run(cmd, cwd=save_path, capture_output=True, text=True) # "Cloning into 'ultralytics'...\n", from `.stderr`, not certain how to capture more; `returncode == 0` should be successful
    return save_path, proc_run

def docs_choices() -> dict:
    """Fetches data from repo and crawls the Docs files for generating links to pages+sections of the Docs as Discord Embeds."""
    Loggr.info(f"Fetching data from {GH_REPO} for documentation.")
    into_path, run_result = fetch_gh_docs()
    #TODO raise run_result.check_returncode() # Raises CalledProcessError
    # REFERENCE https://docs.python.org/3.9/library/subprocess.html#subprocess.CalledProcessError

    # Read MKDOCS index
    docs_idx = [f for f in [(into_path / DOCS_IDX).with_suffix(y) for y in YAML_EXT] if f.exists()]
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
                                        url=base_URL)
                    _ = embed.set_image(url=FULL_LOGO)
                    _ = embed.set_thumbnail(url=LOGO_ICON)
                    
                    for section in TOC:
                        section_name = section.strip('# ').title()
                        section_link = md_index_2link(section, base_URL)
                        
                        _ = embed.add_field(name=section_name, value=f"[Go to section]({section_link})", inline=True)
                    
                    _ = options[k].update({SUB_CAT.strip(string.punctuation).capitalize():embed})
    
    # Generate dictionary for use with app_commands.choices                
    opts_d = dict()
    for k,v in options.items():
        opts_d.update({k:[dict(name=kk, value=kk) for kk in v]})
    
    return opts_d
    # app_commands.choices(**opts_d) # NOTE this might work as @decorator

if __name__ == '__main__()':
    docs_choices()

# NOTE these were other options attempted for use with app_commands.choices 
# @dataclass
# class Options:
#     ...
# opt = Options()
# for k,v in options.items():
#     setattr(opt, k, [dict(name=kk, value=vv) for kk,vv in v.items()])
#     ...

# opts = list()
# for k,v in options.items():
#     opts.append({k:[dict(name=kk, value=kk) for kk in v]})


# SECTIONS = [o for o in options]
# SUB_SECTION = [k for o in options for k in options[o]]
# for o in opts:
#     app_commands.choices(**o)

# class Sections(enum.Enum):
#     def __init__(self, **kwargs):
#         _ = [setattr(self, k, v) for k,v in kwargs.items()]


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
