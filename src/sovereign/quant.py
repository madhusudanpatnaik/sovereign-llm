"""Quantization: what it costs in memory, what it buys in concurrency — computed, not guessed.

Weight memory is counted from the model's own architecture: the config is instantiated on PyTorch's
`meta` device (shapes, no storage, nothing downloaded but config.json), so the parameter count is
exact for any architecture — tied embeddings counted once, Qwen2's attention biases and Qwen3's
q/k norms included.

bitsandbytes quantizes `nn.Linear` weights inside the model; embeddings, norms, biases and the
output head stay in 16-bit (transformers skips the output embedding and anything tied to it). Per
quantized weight:

  - **int8** (LLM.int8): 1 byte, plus one fp32 scale per output row;
  - **nf4**: half a byte, plus a block absmax every 64 weights — fp32 (4/64 B per weight) without
    double quantization, an 8-bit absmax with an fp32 second-level constant per 256 blocks with it.

The serving formats (llm-compressor checkpoints, `compress.py`) quantize the same linear weights and
keep the output head in 16-bit (`ignore=["lm_head"]`):

  - **fp8** (`FP8_DYNAMIC`): 1 byte per weight plus one scale per output channel (counted at 4
    bytes); activations are quantized on the fly, so nothing else is stored;
  - **w4a16** (`W4A16`, GPTQ): half a byte per weight plus a 16-bit scale per group of 128.

The KV cache is what continuous batching spends: every token of every in-flight sequence holds a
key and a value per layer per KV head. Memory left after the weights, divided by the KV bytes per
token, is how many tokens the server can keep in flight — the ceiling on useful batch size. This
is where quantizing the weights pays twice: less memory for weights, more for concurrent sequences.
(Activations and the CUDA context are folded into `overhead_gb`; measure them on the real GPU.)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

Precision = Literal["bf16", "int8", "nf4", "fp8", "w4a16"]
PRECISIONS: tuple[Precision, ...] = ("bf16", "int8", "nf4", "fp8", "w4a16")
GIB = 1024**3


class ParamBreakdown(BaseModel):
    total: int
    linear_weights: int  # quantizable: nn.Linear weights in the decoder, output head excluded
    linear_out_features: int  # rows of those weights (one int8 scale each)
    other: int  # embeddings, norms, biases, output head: kept in 16-bit


class ModelMemory(BaseModel):
    params: ParamBreakdown
    weight_gib: dict[str, float]  # precision -> GiB
    kv_bytes_per_token: int


def breakdown(model: Any) -> ParamBreakdown:
    """Split a meta-device model's parameters into what bitsandbytes quantizes and what it keeps."""
    import torch

    out_emb = model.get_output_embeddings()
    skip = {id(p) for p in out_emb.parameters()} if out_emb is not None else set()
    linear = rows = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and id(module.weight) not in skip:
            linear += module.weight.numel()
            rows += module.weight.shape[0]
    total = sum(p.numel() for p in model.parameters())  # unique parameters: tied weights once
    return ParamBreakdown(
        total=total, linear_weights=linear, linear_out_features=rows, other=total - linear
    )


def weight_bytes(p: ParamBreakdown, precision: Precision, *, double_quant: bool = True) -> float:
    if precision == "bf16":
        return 2.0 * p.total
    if precision in ("int8", "fp8"):
        return 1.0 * p.linear_weights + 4.0 * p.linear_out_features + 2.0 * p.other
    if precision == "w4a16":
        return (0.5 + 2 / 128) * p.linear_weights + 2.0 * p.other
    per_weight = 0.5 + ((1 / 64 + 4 / (64 * 256)) if double_quant else 4 / 64)
    return per_weight * p.linear_weights + 2.0 * p.other


def kv_bytes_per_token(config: Any, dtype_bytes: int = 2) -> int:
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    return int(2 * config.num_hidden_layers * kv_heads * head_dim * dtype_bytes)


def from_config(config: Any) -> ModelMemory:
    import torch
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    p = breakdown(model)
    return ModelMemory(
        params=p,
        weight_gib={prec: round(weight_bytes(p, prec) / GIB, 3) for prec in PRECISIONS},
        kv_bytes_per_token=kv_bytes_per_token(config),
    )


def from_hub(model_id: str) -> ModelMemory:
    """Fetches config.json only — no weights."""
    from transformers import AutoConfig

    return from_config(AutoConfig.from_pretrained(model_id))


class Capacity(BaseModel):
    precision: str
    weights_gib: float
    kv_budget_gib: float
    tokens_in_flight: int
    sequences_at_context: int  # concurrent sequences that can each hold `context_tokens`


def capacity(
    mem: ModelMemory,
    precision: Precision,
    *,
    gpu_gib: float,
    utilization: float = 0.9,
    overhead_gib: float = 1.5,
    context_tokens: int = 4096,
) -> Capacity:
    """How much KV cache — and so how much batching — fits once the weights are loaded."""
    budget = gpu_gib * utilization - mem.weight_gib[precision] - overhead_gib
    tokens = max(0, int(budget * GIB // mem.kv_bytes_per_token))
    return Capacity(
        precision=precision,
        weights_gib=mem.weight_gib[precision],
        kv_budget_gib=round(max(budget, 0.0), 3),
        tokens_in_flight=tokens,
        sequences_at_context=tokens // context_tokens,
    )


def bnb_config(precision: Precision) -> Any:
    """The transformers `BitsAndBytesConfig` for a precision (None for bf16). QLoRA's recipe:
    NF4 weights, double quantization, bf16 compute."""
    import torch
    from transformers import BitsAndBytesConfig

    if precision == "bf16":
        return None
    if precision in ("fp8", "w4a16"):
        raise ValueError(f"{precision} is a serving format made by `sovereign compress`, not bnb")
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
