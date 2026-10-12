# Qwen-Image PE Pipeline for ComfyUI

One self-contained node pack for the **Qwen-Image 2.1 prompt-enhancer (PE)**
workflow: drop in reference images, have a local rewriter turn a short
instruction into a full JSON hint, and feed the result straight into
Qwen-Image 2.1 — with every load parameter under the workflow's control.

Nothing else has to be installed, and nothing has to be patched afterwards.
Wiring this up from generic building blocks instead needs two third-party packs
plus manual edits to both of them.

The chat node is generic, not PE-specific: it talks to any OpenAI-compatible
endpoint and takes its system prompt from a folder you can drop your own files
into. The PE contracts are simply the presets that ship with it.

## Nodes

| Node | Class | What it does |
|---|---|---|
| **Multi Image Loader** | `MultiImageLoader` | Drag & drop up to 16 images. Output sockets appear and disappear with the count, and `images_pipe` carries them all over one wire **at native resolution**. Reorder by dragging thumbnails. |
| **LM Studio Load Model** | `LMStudioLoadModel` | Loads a model with explicit context length / flash attention / KV offload / batch sizes / prediction slots through LM Studio's native API, and optionally **refuses to load a configuration that would not leave enough VRAM for the diffusion model**. The model dropdown refreshes without restarting ComfyUI. |
| **LLM Chat (Multi-Image / Presets)** | `LLMChat` | Any OpenAI-compatible endpoint. Sends every image from `images_pipe` at its native resolution, picks its system prompt from a dropdown of `prompts/*.txt`, and can evict the model afterwards to free VRAM. |
| **PE Hint Resolve** | `PEHintResolve` | Pulls the prompt out of the answer and resolves `wh_ratio` / `ratio_follow` into exact pixels. Tolerates fences, surrounding prose and trailing commas; degrades to 0x0 instead of failing when a size cannot be determined. |
| **Qwen Image 2.1 Multi-Ref Encode** | `QwenImage21MultiRefEncode` | Encodes every reference from a single `images_pipe` connection — **any number, including none** — so one node serves both image editing and text-to-image. Delegates to ComfyUI's own `TextEncodeQwenImage21`. |

All five appear under the **`Qwen PE`** category.

### Text-to-image needs no images

Unhook `images_pipe` from the encoder and the same graph does T2I: ComfyUI's
encoder declares its autogrow template with `min=0` and takes a text-only path,
returning a `resolution x resolution` latent. Nothing else changes — the chat
node already sends a plain string when the pipe is empty, and PE Hint Resolve
already reports `wh_ratio`-based sizes for the T2I contract.

The encoder used to refuse an empty pipe. That was a bug: it made the node
unusable for T2I, which is half of what Qwen-Image 2.1 does.

### Reading what happened

### Reading what happened

Three of these nodes produce a text report — what the load actually applied, why
a size came out the way it did, whether the answer was truncated. **None of them
needs to be wired anywhere.** ComfyUI never displays a node's text output unless
something consumes it, so each of them also returns the same text as `ui.text`
and prints it: it appears on the node itself and in the ComfyUI console. The
`status` / `report` sockets remain for when you *do* want to route it somewhere.

## Install

**ComfyUI Manager / registry**

```
comfy node install qwen-image-pe
```

**Git**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<you>/comfyui_qwen_pe.git
```

**Manual** — copy this folder into `ComfyUI/custom_nodes/`.

No `pip install` step: the only imports are `requests`, `numpy`, `Pillow` and
`torch`, all of which ComfyUI already ships. Restart ComfyUI afterwards.

## Example workflow

[`example_workflows/qwen_image_2_1_edit_with_pe.json`](example_workflows/qwen_image_2_1_edit_with_pe.json)
— Qwen-Image 2.1 image editing with the prompt enhancer inline. Drag it onto the
ComfyUI canvas, or use *Workflow → Open*.

The graph carries a note explaining itself, but the short version is that three
things are yours to set:

1. **Start LM Studio's server** (Developer → Start Server) before queueing.
2. **Swap the three loaders** for model files you actually have.
3. **Pick the PE model** in *LM Studio Load Model* — click refresh first if the
   list is stale.

Then drop reference images into *Multi Image Loader* and write the instruction in
plain language, e.g. `让图1的女孩换上图2的衣服`.

Two details worth noticing, because they are the parts that are easy to get wrong:

* **All three consumers share one `images_pipe`.** The chat node needs the images
  to see them, the resolver needs them to measure them, and the encoder needs
  them to condition on them — one source, one order, one resolution.
* **The output size comes from a switch**, not from the model: either the first
  reference's own size, or a ratio you pick in *ResolutionSelector*. A rewriter
  that reports `ratio_follow` is only useful if you route its `width`/`height`
  somewhere, and this example deliberately does not — it keeps the size
  predictable, and the resolver's `images_pipe` is wired anyway so its report
  stays accurate.

## Languages

The UI follows ComfyUI's own language setting. `locales/en/` and `locales/zh/`
ship with the pack; ComfyUI discovers the folder automatically, so adding
another language is a matter of dropping in `locales/<code>/nodeDefs.json`.

Node names, descriptions, input labels, tooltips and dropdown option labels are
all translated. **Output socket names are deliberately left alone** — they are
the wiring contract, and keeping them identical means a workflow shared between
two people speaking different languages still lines up.

## Wiring

```
[Multi Image Loader] ──images_pipe──┬──> [LLM Chat] ──text──> [PE Hint Resolve]
                                    │                              │
[LM Studio Load Model] ──model──────┘                              │ prompt
        ▲                                                          v
        │                                  [Qwen Multi-Ref Encode].prompt
        └── pick the model here            [Qwen Multi-Ref Encode].images_pipe
```

`PE Hint Resolve` also takes `images_pipe`, because a hint that says
`ratio_follow: "<image1>"` needs the reference's real pixel size. Feeding it the
same pipe means no second image connection.

`PE Hint Resolve`'s `width` / `height` drive an `EmptyLatentImage`. For pure
image editing they match the input, so the encoder's own `latent` output works
just as well; for text-to-image they come from the hint's `wh_ratio`.

### Two connections worth understanding

**`LM Studio Load Model.model` → `LLM Chat.model`**

Right-click the chat node's `model` box → *Convert widget to input*, then link.

The link is what makes the load step run **before** the request — it carries a
model *key*, not the load parameters. Those are applied by the native API call
the load node makes. Skip the link and LM Studio will just-in-time load the
model with whatever configuration it happens to have stored, silently ignoring
what you set. A node whose output nothing consumes is also never executed at
all, so an unlinked load node does nothing.

**`Multi Image Loader.images_pipe` → all three consumers**

One source, one order, one resolution for the chat node (which has to *see* the
references), the hint resolver (which measures them) and the encoder (which
conditions on them).

This is the part a plain batch cannot do. A batch is a `[B, H, W, C]` tensor
with **one height and width for the whole batch**, so stacking forces every
image to a single size and squashes the rest — and ComfyUI's own `Batch Images`
resizes to the first image for exactly that reason. A plain ComfyUI list is no
better: a node without `INPUT_IS_LIST` then runs **once per image**. `images_pipe`
carries the whole set as one opaque value instead, so every image arrives at its
own resolution in a single call.

## VRAM: budget, load, evict

A prompt enhancer and the diffusion model often will not both fit in VRAM — a Q4
PE model with its projector and a 16K cache is ~7.9 GB, and a bf16 Qwen-Image 2.1
DiT is ~13 GB. Two nodes bracket that problem.

### The budget gate

LM Studio will happily load the prompt enhancer, and *then* the diffusion model
has nowhere to go. Nothing errors — the run just gets slow, or the DiT falls back
to system memory. The load request always succeeds if the model fits at all; the
failure that bites is subtler than an out-of-memory.

LM Studio cannot know what your diffusion model needs next. You can:

```
LM Studio Load Model ▸ VRAM Budget (GiB) = 11
```

Above `0`, the node asks LM Studio's own resource estimator what the configuration
would cost (`lms load --estimate-only`) **before touching anything**, and refuses
to load when it overshoots — before unloading anything, so a refusal leaves the
machine exactly as it was.

```
📊 预计占显存 7.86 GiB
❌ 估算需要 7.86 GiB，超过预算 2.00 GiB，已阻止加载。
   当前 context_length=16384，parallel=1。
```

The estimator is the one thing the HTTP API cannot do, which is why it comes from
the `lms` CLI. Real numbers from a Q4_K_M PE model:

| context | estimated GPU memory |
|---|---|
| 16 384 | 7.86 GiB |
| 32 768 | 8.71 GiB |

Default is `0`, which skips the estimate entirely — no cost if you do not want it.

### Loading and evicting

* **`LM Studio Load Model`** decides what the model occupies while it runs, and
  its `unload_other_models` clears the decks before loading.
* **`LLM Chat`'s `unload_after`** frees it again once the answer is in — which is
  the moment that matters, because the sampling graph allocates afterwards.
  `this` evicts the model just used, `all` evicts everything. Default is `off`.

The eviction goes through LM Studio's native API, and the instances are read back
afterwards: LM Studio releases VRAM asynchronously, so "requested" is not the
same as "gone", and the node reports which of the two happened.

### What the load node can and cannot set

`/api/v1/models/load` validates strictly — an unknown key is a 400
`unrecognized_keys`, not a warning. For a llama.cpp GGUF it accepts exactly
fourteen keys, seven of which this pack exposes:

```
context_length · eval_batch_size · physical_batch_size · flash_attention
offload_kv_cache_to_gpu · parallel · context_checkpoints
```

The remaining seven are the speculative-decoding group
(`speculative_draft_mtp`, `_simple`, `_model`, `_max_tokens`, `_min_tokens`,
`_min_continue_probability`) plus `reasoning_budget_message`. They are left out
deliberately: enabling MTP on a model that does not support it fails the entire
load with a 500, and draft-simple needs a second draft model. Neither is worth a
toggle that can break a working graph.

**KV-cache quantization is not reachable at all** — it exists in LM Studio's GUI
settings (`llm.load.llama.kCacheQuantizationType`) but on neither the HTTP API nor
the `lms` CLI. To cut cache memory, lower `context_length`.

`parallel` divides nothing: LM Studio treats `context_length` as **per slot**, and
uses a unified KV cache, so extra slots cost ~150 MiB rather than a multiple of
the cache. A ComfyUI graph sends one request at a time, so the default of 1 is
both correct and cheapest.

### Reading the answer, not failing on it

A model that fences its JSON in ` ```json `, or writes "Here you go:" first, has
still answered correctly — strict `json.loads` would fail the whole run over
punctuation. `PE Hint Resolve` works through twelve real shapes:

| | |
|---|---|
| bare JSON, multi-line, nested objects | ✅ |
| ` ``` ` and ` ```json ` fences | ✅ |
| prose before, after, or both | ✅ |
| prose containing its own balanced `{braces}` | ✅ |
| trailing comma, including inside a fence | ✅ |
| single-quoted keys | ✅ (only when the answer has no `"` at all) |

Each repair is tried in order of how much it alters the text, and the one that
worked is named in the `report` output — so a habitually sloppy model stays
visible instead of being silently tolerated.

What it will **not** do is guess. A truncated object, a bare array, or plain prose
raises with the reason rather than inventing a prompt. And the repairs never touch
string contents: a prompt containing `,}`, an apostrophe, an escaped quote or a
backslash comes through byte-for-byte.

## System prompt presets

The chat node's `system_prompt` dropdown draws from **two sources**:

| Source | Appears as | Lives in |
|---|---|---|
| `prompts/*.txt` in this pack | `pe_i2i.txt` | this folder — works on a machine that has never run LM Studio |
| Presets saved in LM Studio | `[LM] Qwen-Image-2.1-PE-I2I` | `~/.lmstudio/config-presets/<name>.preset.json` |
| Presets from the LM Studio Hub | `[LM] neil/qwen3-thinking` | `~/.lmstudio/hub/presets/<owner>/<name>/preset.json` |

Only presets that actually carry a system prompt are listed — this dropdown picks
a system prompt, and a preset holding nothing but sampling settings has nothing to
offer it.

```bash
cd ComfyUI/custom_nodes/comfyui_qwen_pe
cp my-assistant.txt prompts/          # shows up after a restart
```

The LM Studio entries are **re-read on every run**, not cached at startup — so
editing a prompt in LM Studio's GUI takes effect on the very next queue, with no
restart. Only a *brand-new* preset needs the node's refresh button (or a restart),
because ComfyUI builds a dropdown's option list only once, at startup.

LM Studio has no HTTP endpoint for presets (`/api/v1/presets` returns 404), so the
files are read directly — which is also why this works on a machine where LM Studio
happens to be closed.

`(custom)` falls back to the `system_prompt_text` box. That box also acts as an
override whenever it is non-empty, so a system prompt can be linked in from
elsewhere instead of picked from the list.

### Where the PE contracts came from

`pe_i2i.txt` (17 KB) and `pe_t2i.txt` (10 KB) are the PE models' own
`system_prompt.txt` documents. They are what make those models emit the JSON
contract instead of free prose, and pasting one by hand is the most fragile step
in setting them up — which is why they ship here.

If a model answers with `rewritten_instruction` instead of `rewritten_prompt`,
the resolver still extracts the prompt and flags it: that field name is the
un-contracted fine-tune's own habit and almost always means the contract was not
applied.

## Model compatibility

Built and tested against
[`pottokao/Qwen-Image-2.1-PE-{I2I,T2I}-Heretic-GGUF`](https://huggingface.co/pottokao)
— the abliterated community builds — served by LM Studio.

`LLMChat` itself is generic: it talks to any OpenAI-compatible
`/v1/chat/completions` endpoint (Ollama, llama.cpp, OpenRouter, OpenAI, …).
`LMStudioLoadModel` is the one LM Studio-specific node; without it everything
else still works, you just lose control over load parameters.

## Licence

MIT for everything in this repository — **except `prompts/`**.

`prompts/pe_i2i.txt` and `prompts/pe_t2i.txt` come from Qwen and are distributed
under the **Qwen Research License, non-commercial use only**. See
[`prompts/LICENSE-NOTICE.md`](prompts/LICENSE-NOTICE.md).

Deleting `prompts/` leaves a fully functional MIT-only package; supply your own
contract with the `(custom)` option or drop your own `.txt` files in. The model
weights are community derivatives under the same licence and are not affiliated
with or endorsed by Alibaba / Qwen.
