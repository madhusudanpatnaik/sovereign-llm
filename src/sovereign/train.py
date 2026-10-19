"""QLoRA fine-tuning: a 4-bit frozen base, small trainable LoRA adapters, loss on the answer only.

The recipe (Dettmers et al., 2023) as transformers + peft express it:

  - the base model is loaded with NF4 weights, double quantization and bf16 compute
    (`quant.bnb_config("nf4")`), then `prepare_model_for_kbit_training` (fp32 norms, input grads
    for gradient checkpointing);
  - LoRA adapters (rank `r`, scale `alpha / r`) on every linear projection of every block — the
    attention q/k/v/o and the MLP gate/up/down, via peft's "all-linear" — which QLoRA found matters
    more than the rank;
  - **loss only on the assistant's tokens.** The prompt (system + abstract + question) is labelled
    -100, so the model is trained to produce the answer, not to reproduce the abstract. The answer
    tokens include the chat template's end-of-turn token, so it also learns to stop.

Label masking is done by tokenizing the conversation twice — up to the generation prompt, then in
full — and masking the shared prefix. That only works if the template renders the prompt as an
exact prefix of the full conversation; this is checked per example, and a template that breaks it
is an error, not a silently mislabelled dataset.

Too-long examples are dropped and counted rather than truncated: cutting the abstract changes the
question being asked, and cutting the answer trains on nothing.

`precision` is "nf4" (QLoRA, needs CUDA + bitsandbytes), "bf16" (plain LoRA on a 16-bit base), or
"fp32" (the CPU smoke run).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from sovereign.data import Example, messages

TrainPrecision = Literal["nf4", "bf16", "fp32"]
# peft's "all-linear": every linear layer of every block, output head excluded — the same set as
# q/k/v/o + gate/up/down on Qwen and Llama, and still right on models with fused projections
# (Phi-4-mini's qkv_proj / gate_up_proj), where hard-coded names would silently match nothing.
LORA_TARGETS = "all-linear"


class TrainConfig(BaseModel):
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    precision: TrainPrecision = "nf4"
    r: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: str | list[str] = LORA_TARGETS
    max_length: int = Field(default=1024, ge=16)
    epochs: float = Field(default=3.0, gt=0)
    max_steps: int = -1  # overrides epochs when > 0
    batch_size: int = Field(default=4, ge=1)
    grad_accum: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    warmup: float = Field(default=0.03, ge=0, lt=1)  # fraction of total steps
    seed: int = 0
    output_dir: str = "runs/qlora"
    gradient_checkpointing: bool = True


class Encoded(BaseModel):
    input_ids: list[int]
    labels: list[int]


class EncodeReport(BaseModel):
    kept: int
    dropped_too_long: int
    answer_tokens: int  # total supervised tokens


def encode(example: Example, tokenizer: Any) -> Encoded:
    """Token ids for the full conversation, with every prompt token labelled -100."""
    chat = messages(example, with_answer=True)
    prompt_text = tokenizer.apply_chat_template(
        chat[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(chat, tokenize=False)
    prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full[: len(prompt)] != prompt:
        raise ValueError(
            f"pubid {example.pubid}: the chat template does not render the prompt as a prefix of "
            "the full conversation, so the answer tokens cannot be located for label masking"
        )
    if len(full) == len(prompt):
        raise ValueError(f"pubid {example.pubid}: the answer rendered to no tokens")
    return Encoded(input_ids=full, labels=[-100] * len(prompt) + full[len(prompt) :])


def encode_all(
    examples: Sequence[Example], tokenizer: Any, max_length: int
) -> tuple[list[Encoded], EncodeReport]:
    kept: list[Encoded] = []
    dropped = 0
    for ex in examples:
        enc = encode(ex, tokenizer)
        if len(enc.input_ids) > max_length:
            dropped += 1
            continue
        kept.append(enc)
    supervised = sum(sum(1 for t in e.labels if t != -100) for e in kept)
    return kept, EncodeReport(kept=len(kept), dropped_too_long=dropped, answer_tokens=supervised)


@dataclass
class Collator:
    """Right-pads a batch; padded positions are masked out of attention and of the loss."""

    pad_token_id: int

    def __call__(self, batch: Sequence[Encoded | dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        rows = [b if isinstance(b, dict) else b.model_dump() for b in batch]
        width = max(len(r["input_ids"]) for r in rows)
        ids, mask, labels = [], [], []
        for r in rows:
            pad = width - len(r["input_ids"])
            ids.append(r["input_ids"] + [self.pad_token_id] * pad)
            mask.append([1] * len(r["input_ids"]) + [0] * pad)
            labels.append(r["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_base(cfg: TrainConfig) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    from sovereign.quant import bnb_config

    if cfg.precision == "nf4":
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            quantization_config=bnb_config("nf4"),
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        from peft import prepare_model_for_kbit_training

        return prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float32
    return AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype)


def add_lora(model: Any, cfg: TrainConfig) -> Any:
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def trainable_parameters(model: Any) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_trainer(
    model: Any,
    tokenizer: Any,
    train: list[Encoded],
    val: list[Encoded],
    cfg: TrainConfig,
    *,
    cpu: bool = False,
) -> Any:
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup,
        lr_scheduler_type="cosine",
        bf16=cfg.precision in ("nf4", "bf16"),
        gradient_checkpointing=cfg.gradient_checkpointing and not cpu,
        optim="paged_adamw_8bit" if cfg.precision == "nf4" else "adamw_torch",
        logging_steps=10,
        eval_strategy="epoch" if val else "no",
        save_strategy="no",
        report_to="none",
        seed=cfg.seed,
        use_cpu=cpu,
        remove_unused_columns=False,
    )
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    return Trainer(
        model=model,
        args=args,
        data_collator=Collator(pad_token_id=pad_id),
        train_dataset=[e.model_dump() for e in train],
        eval_dataset=[e.model_dump() for e in val] or None,
        processing_class=tokenizer,
    )


def train(
    cfg: TrainConfig, train_set: Sequence[Example], val_set: Sequence[Example]
) -> dict[str, Any]:
    """The full run on a GPU box: load, encode, attach LoRA, train, save the adapter."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    train_enc, train_report = encode_all(train_set, tokenizer, cfg.max_length)
    val_enc, _ = encode_all(val_set, tokenizer, cfg.max_length)
    model = add_lora(load_base(cfg), cfg)
    trainable, total = trainable_parameters(model)
    trainer = build_trainer(model, tokenizer, train_enc, val_enc, cfg)
    result = trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return {
        "train_loss": result.training_loss,
        "steps": result.global_step,
        "trainable_params": trainable,
        "total_params": total,
        "encode": train_report.model_dump(),
        "adapter_dir": cfg.output_dir,
    }


def merge(base_model_id: str, adapter_dir: str, out_dir: str) -> None:
    """Fold the adapter into a bf16 copy of the base — for servers without LoRA support, or to
    quantize the fine-tuned model as a whole. (Merge into 16-bit weights, never into 4-bit ones.)"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)
