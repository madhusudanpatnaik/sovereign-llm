from __future__ import annotations

import pytest
import torch
from transformers import Phi3Config, Qwen2Config, Qwen3Config

from sovereign import quant, smoke
from sovereign.data import Example
from sovereign.train import Collator, Encoded, encode, encode_all

# Published architectures (dims from the model cards); totals are the safetensors parameter counts.
QWEN3_4B = Qwen3Config(
    vocab_size=151936,
    hidden_size=2560,
    intermediate_size=9728,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    tie_word_embeddings=True,
)
QWEN25_3B = Qwen2Config(
    vocab_size=151936,
    hidden_size=2048,
    intermediate_size=11008,
    num_hidden_layers=36,
    num_attention_heads=16,
    num_key_value_heads=2,
    tie_word_embeddings=True,
)
PHI4_MINI = Phi3Config(
    vocab_size=200064,
    hidden_size=3072,
    intermediate_size=8192,
    num_hidden_layers=32,
    num_attention_heads=24,
    num_key_value_heads=8,
    tie_word_embeddings=True,
    partial_rotary_factor=0.75,  # type: ignore[call-arg]  # a real Phi-3 setting, not in the stubs
    pad_token_id=199999,
)


@pytest.mark.parametrize(
    ("config", "total"),
    [(QWEN3_4B, 4_022_468_096), (QWEN25_3B, 3_085_938_688), (PHI4_MINI, 3_836_021_760)],
)
def test_parameter_counts_match_the_published_checkpoints(config: object, total: int) -> None:
    # Qwen3's q/k norms and head_dim != hidden/heads, Qwen2's qkv biases, Phi's fused projections
    # and every tied embedding are handled by counting the real module tree, not a formula.
    assert quant.from_config(config).params.total == total


def test_tied_embeddings_are_counted_once_and_kept_in_16_bit() -> None:
    tied = quant.from_config(QWEN25_3B)
    untied = quant.from_config(
        QWEN25_3B.__class__(**{**QWEN25_3B.to_dict(), "tie_word_embeddings": False})
    )
    assert untied.params.total - tied.params.total == 151936 * 2048
    assert tied.params.linear_weights == untied.params.linear_weights  # the head is never quantized


def test_weight_memory_by_hand_and_by_precision() -> None:
    p = quant.ParamBreakdown(total=1000, linear_weights=640, linear_out_features=20, other=360)
    assert quant.weight_bytes(p, "bf16") == 2000
    assert quant.weight_bytes(p, "int8") == 640 + 4 * 20 + 720
    assert quant.weight_bytes(p, "nf4", double_quant=False) == pytest.approx(
        640 * (0.5 + 4 / 64) + 720
    )
    assert quant.weight_bytes(p, "fp8") == quant.weight_bytes(p, "int8")  # same layout
    assert quant.weight_bytes(p, "w4a16") == pytest.approx(640 * (0.5 + 2 / 128) + 720)
    mem = quant.from_config(QWEN3_4B)
    assert mem.weight_gib["nf4"] < mem.weight_gib["int8"] < mem.weight_gib["bf16"]
    assert mem.weight_gib["w4a16"] < mem.weight_gib["fp8"] == mem.weight_gib["int8"]
    assert mem.weight_gib["bf16"] == pytest.approx(4_022_468_096 * 2 / 1024**3, abs=1e-3)


def test_kv_cache_and_capacity() -> None:
    mem = quant.from_config(QWEN3_4B)
    assert mem.kv_bytes_per_token == 2 * 36 * 8 * 128 * 2  # 144 KiB per token
    caps = [
        quant.capacity(mem, p, gpu_gib=24, context_tokens=4096) for p in ("bf16", "int8", "nf4")
    ]
    assert (
        caps[0].sequences_at_context < caps[1].sequences_at_context < caps[2].sequences_at_context
    )
    assert quant.capacity(mem, "bf16", gpu_gib=8).tokens_in_flight == 0  # weights do not fit


def test_bnb_configs_follow_the_qlora_recipe() -> None:
    assert quant.bnb_config("bf16") is None
    nf4 = quant.bnb_config("nf4")
    assert nf4.load_in_4bit and nf4.bnb_4bit_quant_type == "nf4"
    assert nf4.bnb_4bit_use_double_quant and nf4.bnb_4bit_compute_dtype == torch.bfloat16
    assert quant.bnb_config("int8").load_in_8bit
    with pytest.raises(ValueError, match="sovereign compress"):
        quant.bnb_config("fp8")


# --------------------------------------------------------------------------- training data


@pytest.fixture(scope="module")
def tokenizer() -> object:
    return smoke.tiny_tokenizer(smoke.toy_examples(12))


def test_only_the_answer_is_supervised(tokenizer: object) -> None:
    ex = smoke.toy_examples(3)[1]
    enc = encode(ex, tokenizer)
    supervised = [t for t, label in zip(enc.input_ids, enc.labels, strict=True) if label != -100]
    assert tokenizer.convert_ids_to_tokens(supervised) == [ex.label, "<|im_end|>"]  # type: ignore[attr-defined]
    assert enc.labels[: -len(supervised)] == [-100] * (len(enc.labels) - len(supervised))


def test_a_template_that_rewrites_the_prompt_is_an_error(tokenizer: object) -> None:
    bad = smoke.tiny_tokenizer(smoke.toy_examples(12))
    # renders the system turn differently once an assistant turn exists: the prompt is no prefix
    bad.chat_template = (
        "{% for m in messages %}{% if m['role'] == 'system' and messages|length > 2 %}"
        "<|im_start|>user\n{% else %}<|im_start|>{{ m['role'] }}\n{% endif %}"
        "{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    with pytest.raises(ValueError, match="prefix"):
        encode(smoke.toy_examples(1)[0], bad)


def test_too_long_examples_are_dropped_and_counted(tokenizer: object) -> None:
    exs = smoke.toy_examples(6)
    long_ex = Example(
        pubid="long", question=exs[0].question, contexts=[" ".join(["data"] * 400)], label="yes"
    )
    kept, report = encode_all([*exs, long_ex], tokenizer, max_length=200)
    assert report.kept == 6 and report.dropped_too_long == 1 and len(kept) == 6
    assert report.answer_tokens == 12  # label + end-of-turn, per example


def test_collator_pads_ids_masks_attention_and_labels() -> None:
    batch = Collator(pad_token_id=0)(
        [Encoded(input_ids=[5, 6, 7], labels=[-100, 6, 7]), Encoded(input_ids=[8], labels=[8])]
    )
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]


# --------------------------------------------------------------------------- the CPU smoke run


def test_cpu_smoke_run_learns_with_only_the_adapter() -> None:
    # Real Qwen2 architecture, real peft LoRA through train.build_trainer, real generate and
    # scoring — the GPU path's code, on a tiny model and a toy task.
    r = smoke.run()
    assert r.base_weights_unchanged and r.adapter_weights_changed
    assert r.trainable_params < r.total_params / 5
    assert r.last_loss < r.first_loss / 10
    assert r.scores.accuracy >= 0.9 and r.scores.invalid_rate == 0.0
