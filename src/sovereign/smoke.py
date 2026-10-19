"""The whole fine-tune -> generate -> score loop on a CPU in seconds, with nothing downloaded.

A real Qwen2-architecture model, randomly initialised from a config a few hundred kilobytes in size;
a real `tokenizers` word-level tokenizer with a ChatML chat template (the format Qwen uses); real
peft LoRA training through `train.build_trainer`; real `generate` through `TransformersClient`;
real scoring through `evaluate`. The same code path a GPU run takes with a 4B model — only the
model is tiny and the task is a toy one a tiny model can learn: the answer is signalled by a
keyword in the question ("improves" -> yes, "worsens" -> no, "unclear" -> maybe).

What this proves: label masking, the collator, LoRA wiring (only adapter weights move), the
trainer, chat-template generation, answer parsing and scoring all work together. What it does not
prove: anything about PubMedQA accuracy. That needs a real model on a GPU.
"""

from __future__ import annotations

import asyncio
import random
import tempfile
from typing import Any

from pydantic import BaseModel

from sovereign.data import LABELS, Example, Label, messages
from sovereign.evaluate import Scores, predict, score

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
_CUE: dict[Label, str] = {"yes": "improves", "no": "worsens", "maybe": "unclear"}
_TOPICS = ["aspirin", "statins", "exercise", "sleep", "vitamin", "surgery", "therapy", "diet"]
_OUTCOMES = ["survival", "recovery", "pain", "mortality", "function", "symptoms"]
_FILLER = ["patients", "were", "studied", "in", "a", "cohort", "trial", "results", "show", "data"]


def toy_examples(n: int, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        label: Label = LABELS[i % 3]
        topic, outcome = rng.choice(_TOPICS), rng.choice(_OUTCOMES)
        context = " ".join(rng.choice(_FILLER) for _ in range(12))
        out.append(
            Example(
                pubid=f"toy{i}",
                question=f"Does {topic} {_CUE[label]} {outcome} ?",
                contexts=[context],
                label=label,
            )
        )
    return out


def tiny_tokenizer(examples: list[Example]) -> Any:
    """A word-level tokenizer over the toy vocabulary, with Qwen's ChatML template."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"]
    words: set[str] = set()
    for ex in examples:
        for m in messages(ex, with_answer=True):
            words.update(m["role"].split())
            words.update(m["content"].replace("\n", " ").split())
    vocab = {tok: i for i, tok in enumerate(specials + sorted(words))}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tok.chat_template = CHATML
    return tok


def tiny_model(vocab_size: int, eos_token_id: int, seed: int = 0) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, Qwen2Config

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        eos_token_id=eos_token_id,
        # A 64-wide model at the default init (std 0.02) behind a frozen final norm cannot produce
        # logits larger than about +-1.3, so no adapter could make it confident. A toy-model
        # setting only; real checkpoints arrive trained.
        initializer_range=0.2,
    )
    return AutoModelForCausalLM.from_config(config)


class SmokeResult(BaseModel):
    first_loss: float
    last_loss: float
    trainable_params: int
    total_params: int
    base_weights_unchanged: bool
    adapter_weights_changed: bool
    scores: Scores


def run(*, steps: int = 120, n_train: int = 90, n_test: int = 30, seed: int = 0) -> SmokeResult:
    import torch

    from sovereign.clients import TransformersClient
    from sovereign.train import (
        TrainConfig,
        add_lora,
        build_trainer,
        encode_all,
        trainable_parameters,
    )

    train_set = toy_examples(n_train, seed)
    test_set = toy_examples(n_test, seed + 1)
    scratch = tempfile.TemporaryDirectory(prefix="sovereign-smoke-")
    tokenizer = tiny_tokenizer(train_set + test_set)
    model = tiny_model(len(tokenizer), tokenizer.eos_token_id, seed)
    cfg = TrainConfig(
        precision="fp32",
        r=8,
        alpha=16,
        dropout=0.0,
        max_steps=steps,
        batch_size=8,
        grad_accum=1,
        learning_rate=3e-3,
        warmup=0.0,
        output_dir=scratch.name,  # the Trainer insists on a directory; nothing is saved
        gradient_checkpointing=False,
        seed=seed,
    )
    encoded, _ = encode_all(train_set, tokenizer, cfg.max_length)
    model = add_lora(model, cfg)
    frozen_before = {
        n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad
    }
    lora_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    trainer = build_trainer(model, tokenizer, encoded, [], cfg, cpu=True)
    try:
        trainer.train()
    finally:
        scratch.cleanup()
    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    params = dict(model.named_parameters())
    trainable, total = trainable_parameters(model)
    model.eval()
    predictions = asyncio.run(
        predict(
            TransformersClient(model, tokenizer, name="smoke"),
            test_set,
            concurrency=1,
            max_tokens=3,
        )
    )
    return SmokeResult(
        first_loss=losses[0],
        last_loss=losses[-1],
        trainable_params=trainable,
        total_params=total,
        base_weights_unchanged=all(torch.equal(params[n], w) for n, w in frozen_before.items()),
        adapter_weights_changed=any(not torch.equal(params[n], w) for n, w in lora_before.items()),
        scores=score("smoke", predictions),
    )
