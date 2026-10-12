"""Qwen-Image 2.1 text encoding for all reference images over ONE wire.

ComfyUI's own ``TextEncodeQwenImage21`` exposes one autogrow socket per
reference, so every additional reference costs another wire and another output
socket. This node takes the whole ``images_pipe`` from ``MultiImageLoader``
instead and forwards it to ComfyUI's own encoder, so the wiring never changes no
matter how many references are uploaded.

**Zero references is a mode, not an error.** ComfyUI's encoder declares its
autogrow template with ``min=0`` and takes a text-only path when nothing is
attached, falling back to a ``resolution x resolution`` latent. That is exactly
text-to-image, and refusing it would make this node unusable for T2I — which is
what the first published version did.

The heavy lifting is delegated to ``TextEncodeQwenImage21.execute`` rather than
reimplemented, so the encoding stays in step with ComfyUI itself.
"""

from __future__ import annotations


class QwenImage21MultiRefEncode:
    CATEGORY = "Qwen PE"
    FUNCTION = "encode"
    DESCRIPTION = (
        "Encode any number of reference images — including none — from one "
        "images_pipe connection, so the wiring never changes. Delegates to "
        "ComfyUI's own TextEncodeQwenImage21."
    )

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    OUTPUT_TOOLTIPS = (
        "Positive conditioning, with every reference spliced in.",
        "Negative conditioning.",
        "Empty latent. With references it matches the first one's size; with none "
        "it is resolution x resolution.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "images_pipe": (
                    "MIL_IMAGES",
                    {
                        "tooltip": "Reference images from Multi Image Loader. Leave "
                        "unconnected for text-to-image.",
                    },
                ),
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
                        "0 keeps each reference at its own size, rounded to 32. "
                        "With no references, this is the square size of the "
                        "generated latent instead.",
                    },
                ),
            },
        }

    def encode(
        self, clip, prompt, negative_prompt, images_pipe=None, vae=None, resolution=1024
    ):
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21

        # No raise on an empty pipe: ComfyUI's encoder supports zero references
        # explicitly (min=0 on the autogrow template), takes the text-only path,
        # and returns a resolution x resolution latent. That IS text-to-image.
        tensors = [t for t in (images_pipe or []) if t is not None]

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
