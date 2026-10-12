"""Qwen-Image 2.1 prompt-enhancer pipeline for ComfyUI — one self-contained pack.

Everything the PE workflow needs lives here, so a fresh install is a single
folder copy and nothing has to be patched afterwards:

    Multi Image Loader        drag & drop N images -> N sockets + one pipe
    LM Studio Load Model      load parameters, plus a VRAM budget it enforces
    LLM Chat                  any OpenAI-compatible endpoint; multi-image at
                              native size; system prompts from prompts/
    PE Hint Resolve           JSON hint -> prompt + exact width/height
    Qwen Image 2.1 Multi-Ref Encode   all references over ONE wire

Typical wiring::

    [Multi Image Loader] --images_pipe--> [LM Studio Load Model]  (optional)
          |                                        |
          |  images_pipe                           | model
          v                                        v
    [LLM Chat] --text--> [PE Hint Resolve] --prompt--> [Qwen Multi-Ref Encode]
          ^                                                    ^
          +------------------- images_pipe ---------------------+

UI strings are localised through ``locales/`` (English and Chinese), which
ComfyUI picks up automatically and switches with its own language setting.

Third-party dependencies: none beyond what ComfyUI already ships (``requests``,
``numpy``, ``Pillow``, ``torch``).
"""

from .nodes_chat import (
    NODE_CLASS_MAPPINGS as _chat_nodes,
    NODE_DISPLAY_NAME_MAPPINGS as _chat_names,
)
from .nodes_loader import (
    NODE_CLASS_MAPPINGS as _loader_nodes,
    NODE_DISPLAY_NAME_MAPPINGS as _loader_names,
)
from .nodes_lmstudio import (
    NODE_CLASS_MAPPINGS as _lms_nodes,
    NODE_DISPLAY_NAME_MAPPINGS as _lms_names,
)
from .nodes_qwen import (
    NODE_CLASS_MAPPINGS as _qwen_nodes,
    NODE_DISPLAY_NAME_MAPPINGS as _qwen_names,
)
from .nodes_resolve import (
    NODE_CLASS_MAPPINGS as _resolve_nodes,
    NODE_DISPLAY_NAME_MAPPINGS as _resolve_names,
)

NODE_CLASS_MAPPINGS = {
    **_loader_nodes,
    **_lms_nodes,
    **_chat_nodes,
    **_resolve_nodes,
    **_qwen_nodes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_loader_names,
    **_lms_names,
    **_chat_names,
    **_resolve_names,
    **_qwen_names,
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
