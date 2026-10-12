"""Where a system prompt can come from.

Two kinds of source, kept distinguishable in the dropdown:

* **This package's ``prompts/`` folder** — ships with the pack, identical for
  everyone, and works on a machine that has never run LM Studio.
* **LM Studio's own presets**, listed with an ``[LM]`` prefix — the prompt you
  already curated in the GUI, so there is one place to edit it and no second copy
  to drift.

LM Studio keeps presets in **two** places, and both are scanned:

    ~/.lmstudio/config-presets/<Name>.preset.json       made in the GUI
    ~/.lmstudio/hub/presets/<owner>/<name>/preset.json  downloaded from the Hub

Only presets that actually carry a system prompt are listed. This dropdown picks a
system prompt, and a preset holding nothing but sampling settings has nothing to
offer it.

There is no endpoint for either location (``/api/v1/presets`` returns 404), so the
files are read directly. That has one structural consequence: ComfyUI builds a
node's dropdown **once**, when it asks for the node definition, so the *list* of
names is a startup snapshot — while the *content* is re-read on every run, so
editing an existing preset takes effect on the next queue without restarting
anything. Only a brand-new preset needs the node's refresh button.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
from typing import Dict, List, Optional

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
LM_HOME = os.path.join(os.path.expanduser("~"), ".lmstudio")
CONFIG_PRESETS_DIR = os.path.join(LM_HOME, "config-presets")
HUB_PRESETS_DIR = os.path.join(LM_HOME, "hub", "presets")

LM_PREFIX = "[LM] "
CUSTOM_PRESET = "(custom)"

_SUFFIX = ".preset.json"
_HUB_FILE = "preset.json"
_SYSTEM_PROMPT_KEY = "llm.prediction.systemPrompt"


def list_local() -> List[str]:
    """The ``.txt`` contracts that ship with this package."""
    try:
        return sorted(f for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt"))
    except OSError:
        return []


def read_local(name: str) -> str:
    """Read one bundled contract; ``name`` is reduced to its basename first."""
    path = os.path.join(PROMPTS_DIR, os.path.basename(name))
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def lmstudio_presets() -> Dict[str, str]:
    """``{display name: file path}`` for every preset LM Studio has.

    Hub presets are named ``<owner>/<name>`` so they cannot collide with a local
    preset of the same name.
    """
    found: Dict[str, str] = {}

    for path in glob.glob(
        os.path.join(CONFIG_PRESETS_DIR, "**", "*" + _SUFFIX), recursive=True
    ):
        found[os.path.basename(path)[: -len(_SUFFIX)]] = path

    for path in glob.glob(os.path.join(HUB_PRESETS_DIR, "*", "*", _HUB_FILE)):
        parts = pathlib.PurePath(path).parts
        if len(parts) >= 3:
            found[f"{parts[-3]}/{parts[-2]}"] = path

    return dict(sorted(found.items()))


def read_lmstudio(name: str) -> Optional[str]:
    """The system prompt stored in one preset, or ``None`` if it has none."""
    path = lmstudio_presets().get(name)
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None

    for field in ((data.get("operation") or {}).get("fields") or []):
        if isinstance(field, dict) and field.get("key") == _SYSTEM_PROMPT_KEY:
            value = field.get("value")
            if isinstance(value, str) and value.strip():
                return value
    return None


def list_lmstudio() -> List[str]:
    """Names of the LM Studio presets that carry a system prompt."""
    return sorted(n for n in lmstudio_presets() if read_lmstudio(n) is not None)


def list_presets() -> List[str]:
    """Everything the ``system_prompt`` dropdown offers, plus the escape hatch."""
    return list_local() + [LM_PREFIX + n for n in list_lmstudio()] + [CUSTOM_PRESET]


def read_preset(value: str) -> str:
    """Resolve a dropdown value to system prompt text; raises ``OSError`` if gone."""
    if value.startswith(LM_PREFIX):
        name = value[len(LM_PREFIX) :]
        text = read_lmstudio(name)
        if text is None:
            raise OSError(f"LM Studio 预设 '{name}' 读不到，或它没有 system prompt")
        return text
    return read_local(value)
