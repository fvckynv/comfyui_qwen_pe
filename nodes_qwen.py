"""Qwen-Image 2.1 text encoding for all reference images over ONE wire.

ComfyUI's own ``TextEncodeQwenImage21`` exposes one autogrow socket per
reference, so every additional reference costs another wire and another output
socket. This node takes the whole ``images_pipe`` from ``MultiImageLoader``
instead and forwards it to ComfyUI's own encoder, so the wiring never changes no
matter how many references are uploaded.

The heavy lifting is delegated to ``TextEncodeQwenImage21.execute`` rather than
reimplemented, so the encoding stays in step with ComfyUI itself.
"""

from __future__ import annotations


class QwenImage21MultiRefEncode:
    CATEGORY = "Qwen PE"
    FUNCTION = "encode"
    DESCRIPTION = (
        "Encode every reference image from one images_pipe connection, so the wiring "
        "never changes no matter how many references there are. Delegates to "
        "ComfyUI's own TextEncodeQwenImage21."
    )

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    OUTPUT_TOOLTIPS = (
        "Positive conditioning, with every reference spliced in.",
        "Negative conditioning.",
        "Empty latent sized to the first reference image.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "images_pipe": ("MIL_IMAGES",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "vae": ("VAE",),
                "resolution": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "Reference images are resized to about "
                        "resolution x resolution pixels, preserving aspect ratio. "
                        "0 keeps each reference at its own size, rounded to 32.",
                    },
                ),
            },
        }

    def encode(
        self, clip, images_pipe, prompt, negative_prompt, vae=None, resolution=1024
    ):
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21

        tensors = [t for t in (images_pipe or []) if t is not None]
        if not tensors:
            raise ValueError(
                "images_pipe 是空的 —— 先在 Multi Image Loader 里上传至少一张图。"
            )

        out = TextEncodeQwenImage21.execute(
            clip=clip,
            prompt=prompt,
            negative_prompt=negative_prompt,
            vae=vae,
            resolution=resolution,
            images={f"image_{i + 1}": t for i, t in enumerate(tensors)},
        )
        return tuple(out.result)


NODE_CLASS_MAPPINGS = {"QwenImage21MultiRefEncode": QwenImage21MultiRefEncode}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImage21MultiRefEncode": "Qwen Image 2.1 Multi-Ref Encode",
}
