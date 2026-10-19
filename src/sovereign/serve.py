"""Serving with vLLM: the command line for every configuration the experiments compare.

vLLM is the serving engine because it implements the two things the spec asks to practise —
continuous batching (sequences join and leave the running batch at every decode step, rather than
waiting for a whole batch to finish) and automatic prefix caching (KV blocks of a shared prompt
prefix are reused across requests) — plus LoRA adapters served on top of one base model, selected
per request by name. This module does not reimplement any of that; it builds the command, and
`bench.py` measures what it does.

Written against vLLM 0.31 (pinned in the `serve` extra; flags move between releases). Facts that
shape the command:
  - prefix caching is *on by default* for decoder-only models, so the "off" arm of a cache A/B must
    say `--no-enable-prefix-caching`; `--enable-prompt-tokens-details` makes every response report
    how many prompt tokens came from the cache, a per-request hit signal for the benchmark;
  - since 0.28, bitsandbytes lives in the `vllm-bnb-plugin` package: `--quantization bitsandbytes`
    quantizes a bf16 checkpoint to NF4 while loading. There is no in-flight 8-bit mode;
  - an llm-compressor checkpoint (`sovereign compress`: fp8 or w4a16) carries its quantization
    config, and vLLM loads it as quantized with no `--quantization` flag. Its LoRA is already
    merged in, so it is served without adapters;
  - `--max-model-len` must be set: Qwen3-4B-Instruct-2507's native 262,144-token context would ask
    for ~36 GiB of KV cache for one sequence and fail at start-up on a 24 GB card;
  - `--max-lora-rank` accepts only 1, 8, 16, 32, 64, 128, 256, 320 or 512, and must be at least the
    adapter's rank; an adapter path containing a comma is parsed as JSON and fails.

vLLM needs Linux with a CUDA GPU (there are no macOS wheels); it is an optional, Linux-only extra.
"""

from __future__ import annotations

import shlex

from pydantic import BaseModel, Field

from sovereign.quant import Precision

LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


class ServeConfig(BaseModel):
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    served_name: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    precision: Precision = "bf16"
    max_model_len: int = Field(default=4096, ge=256)
    gpu_memory_utilization: float = Field(default=0.92, gt=0, le=1)  # vLLM's own default
    max_num_seqs: int = Field(default=256, ge=1)  # ceiling on sequences in the running batch
    prefix_caching: bool = True
    lora_adapters: dict[str, str] = Field(default_factory=dict)  # name -> adapter dir
    max_lora_rank: int = 16


def command(cfg: ServeConfig) -> list[str]:
    if cfg.max_lora_rank not in LORA_RANKS:
        raise ValueError(f"vLLM accepts --max-lora-rank in {LORA_RANKS}, not {cfg.max_lora_rank}")
    bad = [p for p in cfg.lora_adapters.values() if "," in p]
    if bad:
        raise ValueError(f"adapter paths may not contain a comma (vLLM parses them as JSON): {bad}")
    cmd = [
        "vllm",
        "serve",
        cfg.model,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--max-model-len",
        str(cfg.max_model_len),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--max-num-seqs",
        str(cfg.max_num_seqs),
        "--enable-prefix-caching" if cfg.prefix_caching else "--no-enable-prefix-caching",
        "--enable-prompt-tokens-details",
    ]
    if cfg.served_name:
        cmd += ["--served-model-name", cfg.served_name]
    if cfg.precision == "bf16":
        cmd += ["--dtype", "bfloat16"]
    elif cfg.precision == "nf4":
        # NF4 quantization of the bf16 checkpoint while loading (needs vllm-bnb-plugin)
        cmd += ["--dtype", "bfloat16", "--quantization", "bitsandbytes"]
    elif cfg.precision in ("fp8", "w4a16"):
        if cfg.lora_adapters:
            raise ValueError(
                "a compressed checkpoint has its adapter merged in; serve it without --adapter"
            )
        cmd += ["--dtype", "bfloat16"]  # weights' format comes from the checkpoint's own config
    else:
        raise ValueError(
            "vLLM has no in-flight int8 bitsandbytes mode; for an 8-bit model, compress the merge "
            "to fp8 (`sovereign compress --scheme fp8`) and serve that"
        )
    if cfg.lora_adapters:
        cmd += [
            "--enable-lora",
            "--max-lora-rank",
            str(cfg.max_lora_rank),
            "--max-loras",
            str(len(cfg.lora_adapters)),
            "--lora-modules",
        ]
        cmd += [f"{name}={path}" for name, path in cfg.lora_adapters.items()]
    return cmd


def shell(cfg: ServeConfig) -> str:
    return shlex.join(command(cfg))
