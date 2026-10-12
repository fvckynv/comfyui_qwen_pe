# Licence notice for `prompts/`

`pe_i2i.txt` and `pe_t2i.txt` are the `system_prompt.txt` files published with
Qwen's Qwen-Image-2.1 prompt-enhancer models:

* <https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I>
* <https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I>

They are redistributed here verbatim, unmodified, so that the rewriter node can
load its contract without anyone having to paste 10–18 KB by hand.

**They are covered by the Qwen Research License, not by this pack's licence:**

* **Non-commercial use only.**
* Commercial use requires a separate licence from Qwen.

The rest of this node pack is independent of those files. If you need to
distribute this pack commercially, delete `prompts/` and use the
`(自定义 —— 用下面的 custom_system_prompt)` option to supply your own contract.

Note also that the model weights this pack is designed for
(`pottokao/Qwen-Image-2.1-PE-*-Heretic-GGUF`) are community derivatives
distributed under the same Qwen Research License, and are **not** affiliated
with or endorsed by Alibaba / Qwen.
