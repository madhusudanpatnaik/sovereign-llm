"""Post-training quantization of the merged fine-tune, for serving — the throughput path.

QLoRA trains against NF4 weights, and serving those same weights through vLLM's bitsandbytes
plugin keeps exactly what was trained; but bitsandbytes is built for fine-tuning, not for serving
throughput. vLLM's own guidance is FP8 where the GPU supports it and 4-bit weight-only formats
(GPTQ/AWQ) where memory matters more, each with dedicated kernels. So the serving arm here is:

    sovereign merge runs/qlora runs/merged                      # adapter folded into bf16
    sovereign compress runs/merged runs/merged-fp8 --scheme fp8 # or --scheme w4a16
    sovereign serve --model runs/merged-fp8 --precision fp8 --exec
    sovereign gate --candidate preds/fp8.jsonl --baseline preds/bf16.jsonl

  - **fp8** — llm-compressor's `FP8_DYNAMIC`: weights to FP8 with a scale per output channel,
    activations quantized on the fly. No calibration data. FP8 compute needs compute capability
    8.9+ (Ada, Hopper, Blackwell); older GPUs run it weight-only.
  - **w4a16** — GPTQ to 4-bit weights (groups of 128), 16-bit activations. GPTQ needs
    calibration data; it uses PubMedQA *training* prompts rendered through the chat template, so
    the quantizer sees in-domain text and never a test question.

The output head stays in 16-bit in both (`ignore=["lm_head"]`), and the checkpoint carries its
quantization config, so vLLM serves it with no `--quantization` flag. Whether either is good
enough is not assumed: the promotion gate decides, against the bf16 merge, on the same questions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from sovereign.data import Example, messages

Scheme = Literal["fp8", "w4a16"]


class CompressConfig(BaseModel):
    model_dir: str  # the merged bf16 checkpoint
    out_dir: str
    scheme: Scheme = "fp8"
    num_calibration_samples: int = Field(default=512, ge=1)  # w4a16 only
    max_seq_length: int = Field(default=2048, ge=128)


def recipe(scheme: Scheme) -> Any:
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier

    if scheme == "fp8":
        return QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    return GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])


def calibration_texts(train: Sequence[Example], tokenizer: Any, limit: int) -> list[str]:
    """Training prompts as the model will see them at inference — chat template, generation
    prompt, no answer. Only the training split: a test question must never shape the quantizer."""
    return [
        tokenizer.apply_chat_template(messages(ex), tokenize=False, add_generation_prompt=True)
        for ex in train[:limit]
    ]


def compress(
    cfg: CompressConfig,
    train: Sequence[Example],
    *,
    load: Callable[[str], tuple[Any, Any]] | None = None,
    oneshot: Callable[..., Any] | None = None,
    make_recipe: Callable[[Scheme], Any] = recipe,
) -> Path:
    """Quantize `cfg.model_dir` into `cfg.out_dir`; returns the output directory."""
    if load is None:

        def load(path: str) -> tuple[Any, Any]:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            return AutoModelForCausalLM.from_pretrained(path, dtype="auto"), (
                AutoTokenizer.from_pretrained(path)
            )

    if oneshot is None:
        from llmcompressor import oneshot as llmc_oneshot

        oneshot = llmc_oneshot
    model, tokenizer = load(cfg.model_dir)
    if cfg.scheme == "fp8":
        oneshot(model=model, recipe=make_recipe("fp8"))
    else:
        from datasets import Dataset

        texts = calibration_texts(train, tokenizer, cfg.num_calibration_samples)
        dataset = Dataset.from_dict({"text": texts}).map(
            lambda row: tokenizer(
                row["text"],
                padding=False,
                max_length=cfg.max_seq_length,
                truncation=True,
                add_special_tokens=False,  # the chat template already added them
            ),
            remove_columns=["text"],
        )
        oneshot(
            model=model,
            dataset=dataset,
            recipe=make_recipe("w4a16"),
            max_seq_length=cfg.max_seq_length,
            num_calibration_samples=len(texts),
        )
    out = Path(cfg.out_dir)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


def is_compressed(model_dir: str | Path) -> bool:
    """A checkpoint vLLM will load as quantized: its config.json carries a quantization config."""
    import json

    config = Path(model_dir) / "config.json"
    return config.is_file() and "quantization_config" in json.loads(config.read_text())
