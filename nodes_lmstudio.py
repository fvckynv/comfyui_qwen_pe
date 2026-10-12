"""Load an LM Studio model with an explicit load configuration.

The OpenAI-compatible endpoint ComfyUI normally talks to cannot touch *load*
parameters. Context length, flash attention, KV-cache placement and evaluation
batch size are fixed the moment the model is loaded — long before a
``/v1/chat/completions`` request exists — so they are simply not part of that
API. LM Studio exposes them on its native API instead, which is what this node
calls:

    POST {origin}/api/v1/models/load
    { "model": "...", "context_length": 16384, "flash_attention": true, ... }

Wire the ``model`` output into ``PERewriter.model`` (right-click that widget ->
"Convert widget to input"). The link creates a data dependency, so the model is
guaranteed to be loaded with the requested config before the request is sent.
Without the link the rewriter still runs, but LM Studio will just-in-time load
the model with whatever config it happens to have stored — which is exactly the
value this node exists to override.

LM Studio lets one model key hold several loaded instances at once, named
``key``, ``key:2``, ``key:3``, ... That makes "is it already loaded?" a question
about *every* instance of the key, not just the first one found — otherwise a
stale instance with a different context length silently wins the read-back.

The ``model`` dropdown is filled from LM Studio when ComfyUI asks for the node
definition, which only happens at startup, so ``web/refresh_models.py`` adds a
refresh button backed by ``GET /lmstudio_loader/models``.

``vram_budget_gib`` turns this node into a gate. LM Studio's load request always
succeeds if the model fits *at all* — the failure that actually bites is subtler:
the prompt enhancer loads happily, and then the diffusion model has nowhere to
go. Only the graph knows how much the next stage needs, so the budget is declared
here, checked against LM Studio's own estimate before anything is loaded, and a
configuration that would overshoot is refused rather than loaded.
"""

from __future__ import annotations

import json
from typing import List

import requests

from .lms_cli import estimate_load, format_estimate

DEFAULT_BASE = "http://127.0.0.1:1234"
PLACEHOLDER = "⚠ LM Studio 未运行 —— 启动后点「刷新模型」"

# The load parameters this node controls, and therefore compares. These are
# exactly the keys LM Studio's /api/v1/models/load accepts for a llama.cpp GGUF
# (it rejects anything else with an "unrecognized_keys" 400), minus the
# speculative-decoding group: enabling MTP on a model that does not support it
# fails the whole load with a 500, so it is not worth a naive toggle.
_CONFIG_KEYS = (
    "context_length",
    "eval_batch_size",
    "physical_batch_size",
    "flash_attention",
    "offload_kv_cache_to_gpu",
    "parallel",
    "context_checkpoints",
)

# Short: this runs while ComfyUI is building /object_info at startup, and must
# never make startup hang on a sleeping LM Studio.
_PROBE_TIMEOUT = 4


def _origin(url: str) -> str:
    """Reduce any LM Studio URL (base, /v1, or a full endpoint) to its origin."""
    u = (url or "").strip().rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1/responses", "/v1", "/api/v1/models/load"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
            break
    return u.rstrip("/") or DEFAULT_BASE


def _entries(origin: str) -> list:
    r = requests.get(origin + "/api/v1/models", timeout=_PROBE_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, list):
        return payload
    return payload.get("models") or payload.get("data") or []


def list_model_keys(origin: str) -> List[str]:
    """Every downloaded LLM key LM Studio knows about, sorted."""
    keys = []
    for entry in _entries(origin):
        if not isinstance(entry, dict):
            continue
        if entry.get("type") not in (None, "llm", "LLM"):
            continue
        key = entry.get("key") or entry.get("id")
        if key:
            keys.append(key)
    return sorted(set(keys))


def _instances(origin: str) -> list[dict]:
    """Every loaded instance as ``{"key", "id", "config"}``.

    ``id`` may carry an instance suffix (``key:2``); ``key`` is the plain model
    key both are filed under.
    """
    out: list[dict] = []
    for entry in _entries(origin):
        if not isinstance(entry, dict):
            continue
        key = entry.get("key") or entry.get("id")
        for inst in entry.get("loaded_instances") or []:
            if isinstance(inst, dict) and inst.get("id"):
                out.append(
                    {"key": key, "id": inst["id"], "config": inst.get("config") or {}}
                )
    return out


def _unload(origin: str, instance_id: str) -> None:
    r = requests.post(
        origin + "/api/v1/models/unload",
        json={"instance_id": instance_id},
        timeout=180,
    )
    r.raise_for_status()


def _config_matches(config: dict, want: dict) -> bool:
    return all(config.get(k) == v for k, v in want.items())


def _applied(config: dict) -> dict:
    return {k: config.get(k) for k in _CONFIG_KEYS}


def _model_options() -> List[str]:
    """Options for the model dropdown; never raises."""
    try:
        keys = list_model_keys(DEFAULT_BASE)
    except Exception:
        keys = []
    return keys or [PLACEHOLDER]


def _report(model: str, lines: list[str]):
    """Print for the console, hand the text to the node UI, and return both outputs.

    The status is reference material for a human, so it must not need a wire to
    be readable — a node's text output is invisible unless something consumes it.
    """
    text = "\n".join(lines)
    print(f"[LMStudioLoadModel]\n{text}")
    return {"ui": {"text": [text]}, "result": (model, text)}


class LMStudioLoadModel:
    CATEGORY = "Qwen PE"
    FUNCTION = "load"
    DESCRIPTION = (
        "Load an LM Studio model with explicit context length, flash attention, "
        "KV-cache placement and batch size. The OpenAI-compatible API cannot set "
        "these, so this node calls LM Studio's native API instead."
    )

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model", "status")
    OUTPUT_TOOLTIPS = (
        "The model key, passed through so it can drive the rewriter node.",
        "Human-readable report of what happened.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        options = _model_options()
        return {
            "required": {
                "model": (
                    options,
                    {
                        "default": options[0],
                        "tooltip": "LM Studio model key (quantization suffix included). "
                        "Click 刷新模型 after downloading a new model.",
                    },
                ),
                "api_base_url": ("STRING", {"default": DEFAULT_BASE, "multiline": False}),
                "context_length": (
                    "INT",
                    {"default": 16384, "min": 512, "max": 262144, "step": 512},
                ),
            },
            "optional": {
                "flash_attention": ("BOOLEAN", {"default": True}),
                "offload_kv_cache_to_gpu": ("BOOLEAN", {"default": True}),
                "eval_batch_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "physical_batch_size": ("INT", {"default": 512, "min": 32, "max": 8192, "step": 32}),
                "parallel": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "tooltip": "Prediction slots the model can serve at once. A ComfyUI "
                        "graph sends one request at a time, so extra slots buy nothing and "
                        "cost memory. Context length is per slot, not shared.",
                    },
                ),
                "context_checkpoints": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 128,
                        "tooltip": "How many prompt checkpoints to keep, so a re-used prefix "
                        "can be reprocessed cheaply. 0 disables it.",
                    },
                ),
                "unload_other_models": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Unload every other loaded instance (including stale extra "
                        "instances of this same model) before loading, so exactly one clean "
                        "instance ends up resident.",
                    },
                ),
                "vram_budget_gib": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 512.0,
                        "step": 0.5,
                        "tooltip": "0 disables the check. Above 0 this node asks LM Studio "
                        "what the configuration would cost and refuses to load when it "
                        "overshoots, so the prompt enhancer cannot quietly take the VRAM "
                        "the diffusion model needs next. Set it to what is left of your "
                        "GPU after the diffusion model — LM Studio cannot know that, only "
                        "you can.",
                    },
                ),
                "timeout_sec": ("INT", {"default": 300, "min": 10, "max": 1800}),
            },
        }

    def load(
        self,
        model,
        api_base_url,
        context_length,
        flash_attention=True,
        offload_kv_cache_to_gpu=True,
        eval_batch_size=512,
        physical_batch_size=512,
        parallel=1,
        context_checkpoints=32,
        unload_other_models=True,
        vram_budget_gib=0.0,
        timeout_sec=300,
    ):
        model = (model or "").strip()
        if not model or model == PLACEHOLDER:
            raise ValueError(
                "还没选模型。请先启动 LM Studio，然后在节点上点「刷新模型」再选。"
            )

        origin = _origin(api_base_url)
        want = {
            "context_length": int(context_length),
            "flash_attention": bool(flash_attention),
            "offload_kv_cache_to_gpu": bool(offload_kv_cache_to_gpu),
            "eval_batch_size": int(eval_batch_size),
            "physical_batch_size": int(physical_batch_size),
            "parallel": int(parallel),
            "context_checkpoints": int(context_checkpoints),
        }
        notes: list[str] = []

        try:
            before = _instances(origin)
        except Exception as exc:
            raise ValueError(f"连不上 LM Studio ({origin}): {exc}") from exc

        def mine_of(insts):
            return [
                x
                for x in insts
                if x["key"] == model or x["id"] == model or x["id"].split(":")[0] == model
            ]

        mine = mine_of(before)

        # Exactly one instance and it already carries the requested config: done.
        if len(mine) == 1 and _config_matches(mine[0]["config"], want):
            return _report(
                model,
                [
                    "✅ 已加载且配置一致，跳过重载",
                    json.dumps(_applied(mine[0]["config"]), ensure_ascii=False),
                ],
            )
        if len(mine) > 1:
            notes.append(f"发现 {len(mine)} 个同模型实例，将全部清理")

        # Budget gate. Runs before anything is unloaded, so a refused load leaves
        # the machine exactly as it was.
        budget = float(vram_budget_gib or 0.0)
        if budget > 0:
            try:
                info = estimate_load(
                    model, int(context_length), parallel=int(parallel), timeout_sec=60
                )
            except Exception as exc:
                raise ValueError(
                    f"已开启显存预算，但预估失败：{exc}\n"
                    "把 vram_budget_gib 设回 0 可以跳过这个检查。"
                ) from exc

            estimated = float(info.get("gpu_gib") or 0.0)
            notes.append("📊 " + format_estimate(info))
            if estimated > budget:
                raise ValueError(
                    f"❌ 估算需要 {estimated:.2f} GiB，超过预算 {budget:.2f} GiB，已阻止加载。\n"
                    f"   当前 context_length={int(context_length)}，"
                    f"parallel={int(parallel)}。\n"
                    "   要么调小 context_length / eval_batch_size，"
                    "要么把 vram_budget_gib 提高。"
                )
            notes.append(f"✅ 在 {budget:.2f} GiB 预算内（余 {budget - estimated:.2f} GiB）")

        # Unload: everything (when asked), otherwise at least this model's own
        # stale instances — loading again while one is resident would just add
        # another suffixed instance.
        victims = before if unload_other_models else mine
        for inst in victims:
            try:
                _unload(origin, inst["id"])
                notes.append(f"已卸载 {inst['id']}")
            except Exception as exc:
                notes.append(f"卸载 {inst['id']} 失败: {exc}")

        try:
            r = requests.post(
                origin + "/api/v1/models/load",
                json={"model": model, **want},
                timeout=int(timeout_sec),
            )
        except Exception as exc:
            raise ValueError(f"加载请求失败: {exc}") from exc

        if r.status_code != 200:
            raise ValueError(f"加载失败 HTTP {r.status_code}: {r.text[:300]}")

        info = {}
        if r.text.strip():
            try:
                info = r.json()
            except ValueError:
                notes.append("返回不是 JSON")

        # Read the applied config back — knowing which parameters actually took
        # effect is the entire point of this node.
        try:
            after = mine_of(_instances(origin))
        except Exception as exc:
            after = []
            notes.append(f"回读配置失败: {exc}")

        secs = info.get("load_time_seconds")
        notes.append(f"加载完成{'' if secs is None else f'（{secs:.1f}s）'}")

        if not after:
            notes.append("⚠️ 回读不到已加载实例，无法确认参数")
        else:
            if len(after) > 1:
                notes.append(f"⚠️ 仍有 {len(after)} 个实例: " + ", ".join(x["id"] for x in after))
            got = after[0]["config"]
            mismatch = {
                k: {"实际": got.get(k), "期望": want[k]}
                for k in _CONFIG_KEYS
                if got.get(k) != want[k]
            }
            if mismatch:
                notes.append("⚠️ 生效值与期望不符: " + json.dumps(mismatch, ensure_ascii=False))
            else:
                notes.append("✅ 全部参数已生效")
            notes.append(json.dumps(_applied(got), ensure_ascii=False))

        return _report(model, notes)


NODE_CLASS_MAPPINGS = {"LMStudioLoadModel": LMStudioLoadModel}
NODE_DISPLAY_NAME_MAPPINGS = {"LMStudioLoadModel": "LM Studio Load Model"}


def _register_routes() -> None:
    """Expose the live model list so the dropdown can refresh without a restart."""
    from aiohttp import web

    from server import PromptServer

    @PromptServer.instance.routes.get("/lmstudio_loader/models")
    async def _models_route(request):
        base = request.rel_url.query.get("base") or DEFAULT_BASE
        try:
            return web.json_response({"models": list_model_keys(_origin(base))})
        except Exception as exc:
            return web.json_response({"models": [], "error": str(exc)})


try:
    _register_routes()
except Exception as exc:  # headless imports / registry scanners
    print(f"[QwenPE] 未注册刷新路由（{exc}）")
