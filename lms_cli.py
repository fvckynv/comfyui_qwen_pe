"""Thin wrapper around the ``lms`` CLI, for the one thing the HTTP API cannot do.

LM Studio's native API covers loading, unloading and listing, but it has no way
to answer "what would this configuration cost?" — that lives only in
``lms load --estimate-only``. The numbers come from LM Studio's own resource
guardrails, so they account for the context length, the offload ratio and the
runtime's overhead in a way that is not worth reimplementing.

Two quirks shape this module:

* The report is written to **stderr**; stdout holds nothing but a newline.
* It is UTF-8, but ``subprocess`` defaults to the system locale encoding, which
  is GBK on a Chinese Windows and raises on bytes the report legitimately
  contains.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

_GPU_RE = re.compile(r"Estimated\s+GPU\s+Memory:\s*([\d.]+)\s*GiB", re.I)
_TOTAL_RE = re.compile(r"Estimated\s+Total\s+Memory:\s*([\d.]+)\s*GiB", re.I)
_VERDICT_RE = re.compile(r"^\s*Estimate:\s*(.+)$", re.I | re.M)
_CTX_RE = re.compile(r"Context\s+Length:\s*([\d,]+)", re.I)
_OFFLOAD_RE = re.compile(r"GPU\s+Offload:\s*(\S+)", re.I)


def find_lms() -> str | None:
    """Locate the ``lms`` executable: PATH first, then the usual install spots."""
    for name in ("lms", "lms.exe", "lms.cmd"):
        found = shutil.which(name)
        if found:
            return found
    candidates = [
        os.path.join(os.path.expanduser("~"), ".lmstudio", "bin", "lms.exe"),
        os.path.join(os.path.expanduser("~"), ".lmstudio", "bin", "lms"),
        os.path.join(os.path.expanduser("~"), ".cache", "lm-studio", "bin", "lms"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def parse_estimate(text: str) -> dict:
    """Pull the numbers out of ``lms load --estimate-only`` output."""
    out: dict = {"raw": text}
    for key, pattern in (
        ("gpu_gib", _GPU_RE),
        ("total_gib", _TOTAL_RE),
        ("verdict", _VERDICT_RE),
        ("offload", _OFFLOAD_RE),
    ):
        match = pattern.search(text)
        if match:
            out[key] = match.group(1).strip() if key in ("verdict", "offload") else float(match.group(1))
    match = _CTX_RE.search(text)
    if match:
        out["context_length"] = int(match.group(1).replace(",", ""))
    return out


def estimate_load(
    model: str,
    context_length: int,
    parallel: int = 1,
    gpu_offload: str = "max",
    timeout_sec: int = 60,
) -> dict:
    """Ask ``lms`` what this configuration would cost. Raises on anything unusable.

    ``gpu_offload`` defaults to ``max`` because that is the scenario worth
    budgeting for: the whole model on the GPU, competing with whatever the
    diffusion graph wants next. Over the HTTP API the offload ratio is not even
    settable, so partial offload would not describe the load that follows.
    """
    model = (model or "").strip()
    if not model:
        raise ValueError("model 为空，无法估算。")

    lms = find_lms()
    if not lms:
        raise ValueError(
            "找不到 lms 命令行工具。它随 LM Studio 安装，通常在 ~/.lmstudio/bin；"
            "如果文件在那儿仍找不到，把该目录加进 PATH。"
        )

    cmd = [lms, "load", model, "--estimate-only", "-c", str(int(context_length)), "-y"]
    if int(parallel) > 1:
        cmd += ["--parallel", str(int(parallel))]
    if gpu_offload in ("max", "off"):
        cmd += ["--gpu", gpu_offload]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(timeout_sec),
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            f"lms 估算超时（{timeout_sec}s）—— 模型名可能不存在，"
            "lms 会退回交互式选择并一直等下去。"
        ) from exc
    except OSError as exc:
        raise ValueError(f"无法运行 lms: {exc}") from exc

    combined = "\n".join(part for part in (proc.stdout or "", proc.stderr or "") if part.strip())
    info = parse_estimate(combined)
    if "gpu_gib" not in info and "total_gib" not in info:
        raise ValueError("lms 没有返回可解析的估算结果。原始输出：" + (combined.strip()[:400] or "(空)"))
    return info


def format_estimate(info: dict) -> str:
    """One line summarising an estimate, for the load node's status report."""
    gpu = info.get("gpu_gib")
    total = info.get("total_gib")
    parts = []
    if gpu is not None:
        parts.append(f"预计占显存 {gpu:.2f} GiB")
    if total is not None and total != gpu:
        parts.append(f"总内存 {total:.2f} GiB")
    return "  ".join(parts) or "（估算无结果）"
