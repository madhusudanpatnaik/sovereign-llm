"""The serving arm: llm-compressor quantization of the merged model (through stand-ins: the
library is not installed here), serving its checkpoint, and choice-constrained decoding."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from sovereign.cli import main
from sovereign.clients import OpenAICompatibleClient, choice_constraint
from sovereign.compress import CompressConfig, Scheme, compress, is_compressed
from sovereign.data import LABELS, Example
from sovereign.serve import ServeConfig, command


class _Model:
    def __init__(self) -> None:
        self.saved: Path | None = None

    def save_pretrained(self, out: Path) -> None:
        self.saved = out


class _Tokenizer:
    def __init__(self) -> None:
        self.rendered: list[list[dict[str, str]]] = []
        self.saved: Path | None = None

    def apply_chat_template(self, chat: list[dict[str, str]], **kw: Any) -> str:
        assert kw == {"tokenize": False, "add_generation_prompt": True}
        self.rendered.append(chat)
        return "<prompt>" + chat[-1]["content"][:20]

    def __call__(self, text: str, **kw: Any) -> dict[str, list[int]]:
        assert kw["add_special_tokens"] is False and kw["truncation"] is True
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    def save_pretrained(self, out: Path) -> None:
        self.saved = out


def _examples(n: int) -> list[Example]:
    return [
        Example(pubid=str(i), question=f"Question {i}?", contexts=[f"Context {i}."], label="yes")
        for i in range(n)
    ]


def _run(scheme: Scheme, train: list[Example], tmp_path: Path, **cfg: Any) -> dict[str, Any]:
    model, tok = _Model(), _Tokenizer()
    calls: dict[str, Any] = {}
    out = compress(
        CompressConfig(
            model_dir="runs/merged", out_dir=str(tmp_path / "out"), scheme=scheme, **cfg
        ),
        train,
        load=lambda path: (model, tok),
        oneshot=lambda **kw: calls.update(kw),
        make_recipe=lambda s: f"recipe:{s}",
    )
    assert out == tmp_path / "out" and model.saved == out and tok.saved == out
    return {"calls": calls, "tokenizer": tok}


def test_fp8_needs_no_calibration_data(tmp_path: Path) -> None:
    r = _run("fp8", _examples(5), tmp_path)
    assert set(r["calls"]) == {"model", "recipe"} and r["calls"]["recipe"] == "recipe:fp8"
    assert r["tokenizer"].rendered == []


def test_w4a16_calibrates_on_training_prompts_without_answers(tmp_path: Path) -> None:
    train = _examples(5)
    r = _run("w4a16", train, tmp_path, num_calibration_samples=3, max_seq_length=512)
    calls = r["calls"]
    assert calls["recipe"] == "recipe:w4a16"
    assert calls["num_calibration_samples"] == 3 and calls["max_seq_length"] == 512
    assert calls["dataset"].column_names == ["input_ids", "attention_mask"]
    rendered = r["tokenizer"].rendered
    assert len(rendered) == 3 and all(m[-1]["role"] == "user" for m in rendered)  # no answer


def test_is_compressed_reads_the_checkpoint_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert not is_compressed(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"format": "x"}}))
    assert is_compressed(tmp_path)
    assert not is_compressed(tmp_path / "missing")


def test_a_compressed_checkpoint_is_served_without_a_quantization_flag() -> None:
    cmd = command(ServeConfig(model="runs/merged-fp8", precision="fp8"))
    assert "--quantization" not in cmd and cmd[cmd.index("--dtype") + 1] == "bfloat16"
    with pytest.raises(ValueError, match="merged in"):
        command(ServeConfig(model="m", precision="w4a16", lora_adapters={"a": "runs/qlora"}))
    with pytest.raises(ValueError, match="sovereign compress"):
        command(ServeConfig(precision="int8"))


def test_serve_refuses_fp8_for_a_checkpoint_that_is_not_compressed(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(SystemExit, match="unquantized"):
        main(["serve", "--model", str(tmp_path), "--precision", "fp8"])


def test_the_choice_constraint_reaches_the_server() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        chunk = {"choices": [{"delta": {"content": "maybe"}, "finish_reason": "stop"}]}
        body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatibleClient(
        "http://server/v1", "pubmedqa", client=http, extra_body=choice_constraint(LABELS)
    )
    done = asyncio.run(client.complete([{"role": "user", "content": "q"}], max_tokens=4))
    assert done.text == "maybe"
    assert sent[0]["structured_outputs"] == {"choice": ["yes", "no", "maybe"]}
