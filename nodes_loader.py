"""Multi Image Loader — drag & drop N images, get N separate IMAGE outputs.

Output order is deliberately ``count``, ``image_batch``, ``images_list``,
``images_pipe``, ``image_1`` ... ``image_16``: the frontend can only append or
remove output sockets at the *tail* of a node, so keeping the per-image sockets
last lets the bundled web extension trim them to the number of uploaded images
without ever disturbing a link already made to an earlier socket.

``images_pipe`` carries the whole list as ONE opaque value, so a consumer such as
``QwenImage21MultiRefEncode`` or ``PERewriter`` receives every image — at its
native resolution — over a single connection. ``images_list`` is a real ComfyUI
list instead, which makes downstream nodes run once per element.

The ``files`` widget is a hidden STRING holding a JSON array of uploaded
filenames, written by ``web/auto_image_loader.js``. Missing or unreadable files
are skipped with a console warning rather than failing the whole node.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths

MAX_IMAGES = 16


def _load_image_tensor(path: str) -> torch.Tensor:
    """Load one image file as a ``(1, H, W, 3)`` float32 tensor in ``[0, 1]``."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode == "I":
        img = img.point(lambda i: i * (1 / 255))
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _fit_to(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a ``(1, H, W, C)`` tensor to the given spatial size."""
    if tensor.shape[1] == height and tensor.shape[2] == width:
        return tensor
    chw = tensor.permute(0, 3, 1, 2)
    chw = torch.nn.functional.interpolate(
        chw, size=(height, width), mode="bilinear", align_corners=False
    )
    return chw.permute(0, 2, 3, 1)


class MultiImageLoader:
    CATEGORY = "Qwen PE"
    FUNCTION = "load"
    DESCRIPTION = (
        "Drag & drop up to 16 images. The image_N sockets appear and disappear with "
        "the count, and images_pipe carries every image at native resolution over a "
        "single wire."
    )

    RETURN_TYPES = ("INT", "IMAGE", "IMAGE", "MIL_IMAGES") + ("IMAGE",) * MAX_IMAGES
    RETURN_NAMES = ("count", "image_batch", "images_list", "images_pipe") + tuple(
        f"image_{i}" for i in range(1, MAX_IMAGES + 1)
    )
    OUTPUT_IS_LIST = (False, False, True, False) + (False,) * MAX_IMAGES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Hidden in the UI; written by the bundled web extension.
                "files": ("STRING", {"default": "[]", "multiline": False}),
            },
        }

    def load(self, files):
        names: list[str] = []
        try:
            parsed = json.loads(files) if files else []
            if isinstance(parsed, list):
                names = [n for n in parsed if isinstance(n, str) and n]
        except Exception:
            pass
        names = names[:MAX_IMAGES]

        tensors: list[torch.Tensor] = []
        for name in names:
            try:
                path = folder_paths.get_annotated_filepath(name)
                tensors.append(_load_image_tensor(path))
            except Exception as exc:
                print(f"[MultiImageLoader] 跳过 '{name}': {exc}")

        count = len(tensors)

        if count == 0:
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (0, placeholder, [], []) + (placeholder,) * MAX_IMAGES

        placeholder = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        # Per-image outputs keep each file's native resolution (the PE contract
        # sees them as-is); only the convenience batch is harmonised, since
        # stacking requires one common spatial size.
        per_image = tuple(tensors) + (placeholder,) * (MAX_IMAGES - count)
        first = tensors[0]
        batch = torch.cat(
            [_fit_to(t, first.shape[1], first.shape[2]) for t in tensors], dim=0
        )
        return (count, batch, list(tensors), list(tensors)) + per_image


NODE_CLASS_MAPPINGS = {"MultiImageLoader": MultiImageLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MultiImageLoader": "Multi Image Loader"}
