"""Turn a PE rewriter's JSON hint into a diffusion prompt plus pixel dimensions.

The contract asks for::

    {"rewritten_prompt": "...", "wh_ratio": "16:9", "ratio_follow": "<image1>"}

``wh_ratio`` and ``ratio_follow`` are mutually exclusive; ``ratio_follow`` wins
when both appear, because it yields an exact pixel size rather than one derived
from a megapixel budget. The ratio maths mirrors ComfyUI's own Resolution
Selector (``comfy_extras/nodes_resolution.py``): a 1 MP budget is 1024x1024,
linear scale is ``sqrt(area / (w_ratio * h_ratio))``, and each side snaps to the
nearest multiple of ``multiple``.

``ratio_follow`` names a reference image by its 1-based position, so the same
``images_pipe`` that feeds the rewriter and the encoder also answers "how big is
<image1>?" — no second image connection required.

Size resolution is *best effort*: a hint that cannot produce dimensions comes
back as 0x0 with an explanation in ``report``, because the prompt itself is
still perfectly usable when the workflow takes its size from somewhere else.
Only an unusable hint (no prompt text at all) raises.

Parsing is equally forgiving. A model that fences its answer in ```` ```json ````
or writes "Here you go:" first has still answered correctly; strict
``json.loads`` would fail the whole run over punctuation. The repairs are tried
in order of how much they alter the text, and whichever one worked is reported,
so a habitually sloppy model stays visible instead of being silently tolerated.
"""

from __future__ import annotations

import json
import math
import re

_FOLLOW_RE = re.compile(r"^<image(\d+)>$")
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.S)
_MIN_SIZE = 8

# Keys the un-contracted fine-tune tends to emit instead of "rewritten_prompt".
_PROMPT_FALLBACKS = ("rewritten_instruction", "rewritten", "prompt", "instruction")


def _strip_fences(text: str) -> str:
    """The contents of the first ``` block, if the answer arrived in one."""
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text


def _match_brace(text: str, start: int) -> int | None:
    """Index of the ``}`` closing the ``{`` at ``start``, ignoring string contents."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _balanced_objects(text: str):
    """Every ``{...}`` in the text that braces balance, in order of appearance.

    A balanced pair is not necessarily the JSON. Prose like ``use {braces} like
    this:`` balances perfectly and means nothing, so the search must continue
    past it rather than stopping at the first match — and it resumes *after* each
    candidate, so a nested object is never mistaken for the answer.
    """
    position = text.find("{")
    while position != -1:
        end = _match_brace(text, position)
        if end is None:
            position = text.find("{", position + 1)
            continue
        yield text[position : end + 1]
        position = text.find("{", end + 1)


def _strip_trailing_commas(text: str) -> str:
    """Drop ``,`` before ``}`` or ``]``, but only outside strings."""
    out: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == ",":
            look = index + 1
            while look < len(text) and text[look] in " \t\r\n":
                look += 1
            if look < len(text) and text[look] in "}]":
                index += 1  # drop it
                continue
        out.append(char)
        index += 1
    return "".join(out)


def extract_json_object(text: str) -> tuple[dict, str]:
    """Parse the JSON object out of whatever the model actually produced.

    Models wrap their answer in ``` fences, preface it with "Here you go:", or
    leave a trailing comma. Strict ``json.loads`` rejects all of that, which
    turns a perfectly usable answer into a failed run. Each repair is tried in
    order of how much it changes the text, and the one that worked is reported so
    a habitually-sloppy model is visible rather than silently tolerated.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("hint 是空的 —— 把 LLM Chat 的 text 输出接过来。")

    def variants(source: str):
        yield source, "直接解析"
        yield _strip_trailing_commas(source), "去掉尾逗号"
        # Only when the answer has no double quotes at all, so a genuine
        # apostrophe in prose can never be turned into a delimiter.
        if '"' not in source:
            yield source.replace("'", '"'), "单引号改双引号"

    seen: set[str] = set()
    attempts: list[tuple[str, str]] = []
    stripped = _strip_fences(raw)
    candidates = [raw, stripped]
    candidates.extend(_balanced_objects(raw))
    if stripped != raw:
        candidates.extend(_balanced_objects(stripped))
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            attempts.extend(variants(candidate))

    last_error = ""
    for candidate, strategy in attempts:
        try:
            data = json.loads(candidate)
        except ValueError as exc:
            last_error = str(exc)
            continue
        if isinstance(data, dict):
            return data, strategy
        last_error = "顶层不是 JSON 对象"

    raise ValueError(
        f"从输出里提不出 JSON 对象（{last_error}）。\n开头: {raw[:200]}"
    )


def _get_ci(data: dict, key: str):
    """Case-insensitive lookup, falling back to an exact match."""
    if key in data:
        return data[key], key
    lowered = key.lower()
    for k in data:
        if isinstance(k, str) and k.lower() == lowered:
            return data[k], k
    return None, None


def _as_str(value) -> str:
    return "" if value is None else str(value).strip()


def parse_ratio(value) -> tuple[float, float] | None:
    """Parse ``W:H`` into positive floats; ``None`` when empty."""
    s = _as_str(value)
    if not s:
        return None
    if ":" not in s:
        raise ValueError(f"wh_ratio 必须用 ':' 分隔（如 '16:9'），得到 '{s}'")
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"wh_ratio 必须正好两段，得到 '{s}'")
    w, h = float(parts[0].strip()), float(parts[1].strip())
    if not (w > 0 and h > 0):
        raise ValueError(f"wh_ratio 必须为正数，得到 '{s}'")
    return w, h


def dims_from_ratio(ratio_w: float, ratio_h: float, megapixels: float, multiple: int) -> tuple[int, int]:
    total = float(megapixels) * 1024 * 1024
    scale = math.sqrt(total / (ratio_w * ratio_h))

    def snap(n: float) -> int:
        return max(_MIN_SIZE, round(n / multiple) * multiple)

    return snap(ratio_w * scale), snap(ratio_h * scale)


class PEHintResolve:
    CATEGORY = "Qwen PE"
    FUNCTION = "resolve"
    DESCRIPTION = (
        "Strip the JSON wrapper off an LLM hint and resolve wh_ratio / ratio_follow "
        "into exact pixel dimensions. An unresolvable size comes back as 0x0 with an "
        "explanation, never as an error."
    )

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("prompt", "width", "height", "report")
    OUTPUT_TOOLTIPS = (
        "The rewritten prompt, with the JSON wrapper stripped.",
        "Target width in pixels; 0 when it could not be determined.",
        "Target height in pixels; 0 when it could not be determined.",
        "What was found and how the size was decided.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hint": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "dynamicPrompts": False,
                        "tooltip": "The JSON text from LLM Chat.",
                    },
                ),
            },
            "optional": {
                "images_pipe": (
                    "MIL_IMAGES",
                    {"tooltip": "Needed only when the hint says ratio_follow, e.g. '<image1>'."},
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": "Only used when the hint gives wh_ratio; ignored for ratio_follow.",
                    },
                ),
                "multiple": (
                    "INT",
                    {"default": 8, "min": 8, "max": 128, "step": 4},
                ),
            },
        }

    def resolve(self, hint, images_pipe=None, megapixels=1.0, multiple=8):
        notes: list[str] = []

        text = _as_str(hint)
        if not text:
            raise ValueError("hint 是空的 —— 把 LLM Chat 的 text 输出接过来。")

        data, strategy = extract_json_object(text)
        if strategy != "直接解析":
            notes.append(f"⚠️ 输出不是干净 JSON，靠「{strategy}」救回来的")

        # ---- prompt ------------------------------------------------------
        prompt, key = _get_ci(data, "rewritten_prompt")
        prompt = _as_str(prompt)
        if not prompt:
            for fallback in _PROMPT_FALLBACKS:
                value, found = _get_ci(data, fallback)
                value = _as_str(value)
                if value:
                    prompt = value
                    notes.append(
                        f"⚠️ 没有 rewritten_prompt，退回用 '{found}' —— "
                        "通常说明合约（system prompt）没生效。"
                    )
                    break
        if not prompt:
            raise ValueError(f"JSON 里找不到提示词。实际字段: {sorted(map(str, data))}")
        if key and key != "rewritten_prompt":
            notes.append(f"字段名大小写为 '{key}'")

        # ---- size --------------------------------------------------------
        width = height = 0
        wh_raw, _ = _get_ci(data, "wh_ratio")
        follow_raw, _ = _get_ci(data, "ratio_follow")
        wh = _as_str(wh_raw)
        follow = _as_str(follow_raw)

        if follow:
            match = _FOLLOW_RE.match(follow.lower())
            if not match:
                notes.append(f"⚠️ ratio_follow 格式应为 '<imageN>'，得到 '{follow}' → 尺寸留 0")
            else:
                index = int(match.group(1))
                images = [t for t in (images_pipe or []) if t is not None]
                if len(images) < index:
                    notes.append(
                        f"⚠️ ratio_follow 指向 <image{index}>，但只提供了 {len(images)} 张 "
                        "→ 尺寸留 0（把 images_pipe 也接到这个节点）"
                    )
                else:
                    tensor = images[index - 1]
                    height, width = int(tensor.shape[-3]), int(tensor.shape[-2])
                    notes.append(f"尺寸跟随 <image{index}>：{width}×{height}")
        elif wh:
            try:
                ratio = parse_ratio(wh)
                width, height = dims_from_ratio(ratio[0], ratio[1], float(megapixels), int(multiple))
                notes.append(f"按 wh_ratio '{wh}' × {float(megapixels)}MP → {width}×{height}")
            except Exception as exc:
                notes.append(f"⚠️ wh_ratio 解析失败（{exc}）→ 尺寸留 0")
        else:
            notes.append(
                "⚠️ JSON 里既没有 wh_ratio 也没有 ratio_follow → 尺寸留 0，"
                "请改用别的尺寸来源（如 Get Image Size）。"
            )

        report = "\n".join(notes)
        # Printed as well as returned, so the reasoning is visible without a wire.
        print(f"[PEHintResolve]\n{report}")
        return {
            "ui": {"text": [report]},
            "result": (prompt, int(width), int(height), report),
        }


NODE_CLASS_MAPPINGS = {"PEHintResolve": PEHintResolve}
NODE_DISPLAY_NAME_MAPPINGS = {"PEHintResolve": "PE Hint Resolve"}
