"""Chat with any OpenAI-compatible LLM, carrying multi-image input at native size.

Works with LM Studio, Ollama, llama.cpp, OpenRouter, OpenAI — anything serving
``/v1/chat/completions``.

The interesting part is ``images_pipe``. The usual way to send several images to
a vision model is a batch, and a batch forces every image to one common size,
which distorts them — which matters a lot when the model is being asked to
describe or edit them. A plain ComfyUI list is no better: a node without
``INPUT_IS_LIST`` then runs **once per image** instead of once with all of them.
``MultiImageLoader.images_pipe`` sidesteps both: it carries the whole set as one
opaque value, and every image is encoded at its own resolution.

The system prompt comes from one of two places, both listed in the same dropdown
(see ``presets.py``): the ``.txt`` files in this package's ``prompts/`` folder,
and the presets already saved in LM Studio, shown with an ``[LM]`` prefix. The LM
Studio ones are read fresh on every run, so editing a prompt in the LM Studio GUI
takes effect on the next queue. ``(custom)`` falls back to the
``system_prompt_text`` box, which also works as a linkable override — the text
box wins whenever it is non-empty.

Each call is a fresh single-turn exchange: system message plus one user message.
No history is kept, which is what stateless rewriters and captioners expect.
"""

from __future__ import annotations

import base64
import io
import json
from typing import List

import numpy as np
import requests

from .presets import CUSTOM_PRESET, list_presets, read_preset


def _default_model() -> str:
    try:
        from .nodes_lmstudio import DEFAULT_BASE, list_model_keys

        keys = list_model_keys(DEFAULT_BASE)
        if keys:
            return keys[0]
    except Exception:
        pass
    return ""


def _chat_url(base: str) -> str:
    """Accept an origin, a ``/v1`` base, or a full endpoint."""
    u = (base or "").strip().rstrip("/") or "http://127.0.0.1:1234"
    if u.endswith("/chat/completions"):
        return u
    if u.endswith("/v1"):
        return u + "/chat/completions"
    return u + "/v1/chat/completions"


def parse_stop(text) -> List[str]:
    """Accept a JSON array of strings or one stop sequence per line."""
    raw = (text or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    return [line for line in raw.splitlines() if line.strip()]


def _unload_models(api_base_url: str, model: str, mode: str) -> str:
    """Free VRAM after the answer arrived; returns a line for the status report.

    A prompt enhancer holds several GB that downstream image generation needs, so
    the interesting moment to evict it is right here — before the sampling graph
    starts allocating. The eviction goes through LM Studio's native API, and the
    instances are read back afterwards, because LM Studio frees VRAM
    asynchronously and "requested" is not the same as "gone".
    """
    from .nodes_lmstudio import _instances, _origin, _unload

    origin = _origin(api_base_url)
    try:
        before = _instances(origin)
    except Exception as exc:
        return f"⚠️ 卸载失败，连不上 LM Studio ({origin}): {exc}"

    if mode == "all":
        targets = before
    else:
        targets = [
            x
            for x in before
            if x["key"] == model or x["id"] == model or x["id"].split(":")[0] == model
        ]

    if not targets:
        return "卸载：没有需要卸载的实例"

    done: list[str] = []
    failed: list[str] = []
    for inst in targets:
        try:
            _unload(origin, inst["id"])
            done.append(inst["id"])
        except Exception as exc:
            failed.append(f"{inst['id']}（{exc}）")

    lines = []
    if done:
        lines.append(f"已请求卸载 {len(done)} 个实例：" + "、".join(done))
    if failed:
        lines.append("⚠️ 卸载失败：" + "；".join(failed))

    try:
        left = _instances(origin)
        if left:
            lines.append("⚠️ 仍残留：" + "、".join(x["id"] for x in left))
        else:
            lines.append("✅ 显存已释放，LM Studio 现在没有已加载模型")
    except Exception as exc:
        lines.append(f"⚠️ 回读失败: {exc}")

    return "\n".join(lines)


def _to_data_url(tensor) -> str:
    """Encode one ``(1, H, W, C)`` image tensor as a PNG data URL, unresized."""
    from PIL import Image

    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"期望 [H,W,C] 图像张量，得到 {arr.shape}")

    frame = np.clip(arr, 0.0, 1.0)
    frame = (frame * 255.0).round().astype("uint8")
    channels = frame.shape[-1]
    if channels == 1:
        pil = Image.fromarray(frame[..., 0], mode="L")
    elif channels == 3:
        pil = Image.fromarray(frame, mode="RGB")
    elif channels == 4:
        pil = Image.fromarray(frame, mode="RGBA")
    else:
        raise ValueError(f"期望 1/3/4 通道，得到 {channels}")

    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class LLMChat:
    CATEGORY = "Qwen PE"
    FUNCTION = "chat"
    DESCRIPTION = (
        "Chat with any OpenAI-compatible LLM. Sends every image from images_pipe "
        "at its native resolution, and takes its system prompt from a dropdown of "
        "the .txt files in prompts/."
    )

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "reasoning", "status")
    OUTPUT_TOOLTIPS = (
        "The model's answer.",
        "The thinking block, when the server returns it separately.",
        "What was sent and what came back (usage, applied parameters).",
    )

    @classmethod
    def INPUT_TYPES(cls):
        presets = list_presets()
        return {
            "required": {
                "api_base_url": (
                    "STRING",
                    {"default": "http://127.0.0.1:1234/v1", "multiline": False},
                ),
                "model": (
                    "STRING",
                    {
                        "default": _default_model(),
                        "multiline": False,
                        "tooltip": "Model key. Link it from LM Studio Load Model "
                        "(right-click -> Convert widget to input) so load parameters "
                        "are applied first.",
                    },
                ),
                "system_prompt": (
                    presets,
                    {
                        "default": presets[0],
                        "tooltip": "Two sources: .txt files in this pack's prompts/ folder, "
                        "and presets saved in LM Studio (shown with an [LM] prefix). The LM "
                        "Studio ones are re-read every run, so editing one in the GUI takes "
                        "effect on the next queue. Click refresh after adding a new preset.",
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "dynamicPrompts": False},
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 6144,
                        "min": 64,
                        "max": 131072,
                        "step": 64,
                        "tooltip": "Prompt rewriters think before answering, so this must "
                        "cover the thinking block AND the answer. A value near 1024 "
                        "truncates the JSON mid-object.",
                    },
                ),
            },
            "optional": {
                "images_pipe": (
                    "MIL_IMAGES",
                    {
                        "tooltip": "Images from Multi Image Loader, sent at native "
                        "resolution. Needed by vision models and edit contracts.",
                    },
                ),
                "system_prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "dynamicPrompts": False,
                        "tooltip": "Overrides the dropdown whenever it is non-empty. Also "
                        "works as a link, so a system prompt can come from elsewhere.",
                    },
                ),
                "image_max_count": ("INT", {"default": 16, "min": 1, "max": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "thinking": (
                    ["default", "on", "off"],
                    {
                        "default": "default",
                        "tooltip": "default sends nothing. The PE models only support "
                        "on/off and ignore effort levels.",
                    },
                ),
                "json_mode": ("BOOLEAN", {"default": False}),
                "stop": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "One stop sequence per line, or a JSON array of strings.",
                    },
                ),
                "unload_after": (
                    ["off", "this", "all"],
                    {
                        "default": "off",
                        "tooltip": "Free VRAM once the answer is in, so downstream image "
                        "generation has room. 'this' unloads the model just used, 'all' "
                        "unloads everything. Goes through LM Studio's native API.",
                    },
                ),
                "timeout_sec": ("INT", {"default": 300, "min": 10, "max": 3600}),
            },
        }

    def chat(
        self,
        api_base_url,
        model,
        system_prompt,
        user_prompt,
        max_tokens,
        images_pipe=None,
        system_prompt_text="",
        image_max_count=16,
        temperature=0.7,
        top_p=1.0,
        seed=-1,
        thinking="default",
        json_mode=False,
        stop="",
        unload_after="off",
        timeout_sec=300,
    ):
        model = (model or "").strip()
        if not model:
            raise ValueError(
                "model 为空 —— 从「LM Studio Load Model」的 model 输出连过来，或直接填写。"
            )

        override = (system_prompt_text or "").strip()
        if override:
            prompt_text = override
            source = f"system_prompt_text（{len(override)} 字符，覆盖下拉）"
        elif system_prompt == CUSTOM_PRESET:
            raise ValueError("选了「自定义」，但 system_prompt_text 是空的。")
        else:
            try:
                prompt_text = read_preset(system_prompt)
            except OSError as exc:
                raise ValueError(f"读不到预设 '{system_prompt}': {exc}") from exc
            source = f"{system_prompt}（{len(prompt_text)} 字符）"

        images = [t for t in (images_pipe or []) if t is not None][: int(image_max_count)]

        if images:
            content: list = [{"type": "text", "text": user_prompt}]
            for tensor in images:
                content.append({"type": "image_url", "image_url": {"url": _to_data_url(tensor)}})
            user_content = content
        else:
            # No images: keep content a plain string, the shape every
            # OpenAI-compatible server accepts.
            user_content = user_prompt

        messages: list = []
        if prompt_text:
            messages.append({"role": "system", "content": prompt_text})
        messages.append({"role": "user", "content": user_content})

        body = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if int(seed) >= 0:
            body["seed"] = int(seed)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if thinking == "on":
            body["reasoning_effort"] = "medium"
        elif thinking == "off":
            body["reasoning_effort"] = "none"
        stop_values = parse_stop(stop)
        if stop_values:
            body["stop"] = stop_values

        url = _chat_url(api_base_url)
        try:
            response = requests.post(url, json=body, timeout=int(timeout_sec))
        except Exception as exc:
            raise ValueError(f"请求 {url} 失败: {exc}") from exc

        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code}: {response.text[:300]}")

        payload = response.json()
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""

        usage = payload.get("usage") or {}
        status = (
            f"model={model}\n"
            f"system_prompt={source}\n"
            f"images={len(images)}  max_tokens={int(max_tokens)}  "
            f"temperature={float(temperature)}  thinking={thinking}\n"
            f"finish_reason={choice.get('finish_reason')}\n"
            f"usage={json.dumps(usage, ensure_ascii=False)}\n"
            f"text={len(text)} 字符  reasoning={len(reasoning)} 字符"
        )
        if choice.get("finish_reason") == "length":
            status += "\n⚠️ 输出被 max_tokens 截断 —— 若期待 JSON，它很可能不完整，调大 max_tokens。"

        if unload_after != "off":
            status += "\n" + _unload_models(api_base_url, model, unload_after)

        # Printed as well as returned: a node's text output is invisible unless
        # something consumes it, and this is exactly the text you need when a run
        # goes wrong.
        print(f"[LLMChat]\n{status}")
        return {"ui": {"text": [status]}, "result": (text, reasoning, status)}


NODE_CLASS_MAPPINGS = {"LLMChat": LLMChat}
NODE_DISPLAY_NAME_MAPPINGS = {"LLMChat": "LLM Chat (Multi-Image / Presets)"}


def _register_routes() -> None:
    """Expose the live preset list so a new LM Studio preset needs no restart."""
    from aiohttp import web

    from server import PromptServer

    @PromptServer.instance.routes.get("/qwen_pe/presets")
    async def _presets_route(request):
        try:
            return web.json_response({"presets": list_presets()})
        except Exception as exc:
            return web.json_response({"presets": [], "error": str(exc)})


try:
    _register_routes()
except Exception as exc:  # headless imports / registry scanners
    print(f"[QwenPE] 未注册预设路由（{exc}）")
