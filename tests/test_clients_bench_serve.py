from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from fake_server import FakeServer
from sovereign import bench
from sovereign.cli import main
from sovereign.clients import AnthropicClient, OpenAICompatibleClient
from sovereign.data import Example, Label
from sovereign.evaluate import Prediction, predict, write_predictions
from sovereign.serve import ServeConfig, command

MSG = [{"role": "user", "content": "Does it work?"}]


def _ex(i: int, label: Label = "yes") -> Example:
    return Example(
        pubid=str(i), question=f"Question {i}?", contexts=[f"Abstract {i}."], label=label
    )


# ------------------------------------------------------------------- OpenAI-compatible client


def test_streaming_client_records_tokens_timing_usage_and_cache_hits() -> None:
    async def go() -> tuple[Any, list[dict[str, object]]]:
        srv = FakeServer(tokens=4, prefill_s=0.05, itl_s=0.01, cached_tokens=30)
        async with srv.running() as url:
            client = OpenAICompatibleClient(url, "pubmedqa", extra_body={"ignore_eos": True})
            c = await client.complete(MSG, max_tokens=4)
            await client.aclose()
        return c, srv.requests

    c, requests = asyncio.run(go())
    assert c.text == "yes..." and c.stop_reason == "stop"
    assert (c.input_tokens, c.output_tokens, c.cached_tokens) == (42, 4, 30)
    assert c.ttft_s is not None and c.ttft_s >= 0.05  # the empty role chunk is not a first token
    assert len(c.token_times_s) == 4 and all(gap >= 0.009 for gap in c.inter_token_s)
    body = requests[0]
    assert body["model"] == "pubmedqa" and body["stream"] is True and body["ignore_eos"] is True
    assert body["stream_options"] == {"include_usage": True} and body["temperature"] == 0.0


@pytest.mark.parametrize("server", [FakeServer(status=500), FakeServer(fail_mid_stream=True)])
def test_streaming_client_raises_on_http_and_mid_stream_errors(server: FakeServer) -> None:
    async def go() -> None:
        async with server.running() as url:
            client = OpenAICompatibleClient(url, "m")
            try:
                await client.complete(MSG, max_tokens=4)
            finally:
                await client.aclose()

    with pytest.raises(RuntimeError, match="500|stream error"):
        asyncio.run(go())


# --------------------------------------------------------------------------- frontier client


def _anthropic(responses: list[httpx2.Response]) -> tuple[AnthropicClient, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    queue = iter(responses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(
            {
                "url": str(request.url),
                "beta": request.headers.get("anthropic-beta"),
                "body": json.loads(request.content),
            }
        )
        return next(queue)

    http = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    sdk = anthropic.AsyncAnthropic(api_key="test-key", http_client=http, max_retries=0)
    return AnthropicClient(client=sdk), sent


def _message(text: str | None, stop_reason: str = "end_turn", **extra: Any) -> httpx2.Response:
    content: list[dict[str, Any]] = [{"type": "thinking", "thinking": "...", "signature": "s"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    return httpx2.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 300, "output_tokens": 40},
            **extra,
        },
    )


def test_frontier_client_through_the_real_sdk() -> None:
    refusal = {"type": "refusal", "category": "bio", "explanation": None}
    client, sent = _anthropic([_message("Yes."), _message("No", "refusal", stop_details=refusal)])
    preds = asyncio.run(predict(client, [_ex(1), _ex(2, "no")], concurrency=1))
    assert [p.pred for p in preds] == ["yes", None]  # a refusal is scored invalid, not raised
    body = sent[0]["body"]
    assert sent[0]["url"].endswith("/v1/messages?beta=true")
    assert sent[0]["beta"] == "server-side-fallback-2026-07-01"
    assert body["model"] == "claude-opus-5-5" and body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "high"} and body["max_tokens"] == 16000
    assert (
        body["system"].startswith("You answer biomedical") and body["messages"][0]["role"] == "user"
    )
    assert "temperature" not in body and "thinking" not in body


# --------------------------------------------------------------------------- benchmark


def test_closed_loop_scales_until_the_server_saturates_then_queues() -> None:
    async def go() -> list[bench.RunStats]:
        srv = FakeServer(tokens=6, prefill_s=0.03, itl_s=0.005, slots=4)
        async with srv.running() as url:
            client = OpenAICompatibleClient(url, "m")
            out = [
                await bench.closed_loop(client, [MSG] * 24, concurrency=c, max_tokens=6)
                for c in (1, 4, 12)
            ]
            await client.aclose()
        return out

    one, four, twelve = asyncio.run(go())
    assert four.requests_per_s > 2.5 * one.requests_per_s  # batching: 4 slots work in parallel
    assert twelve.requests_per_s < 1.5 * four.requests_per_s  # saturated at 4 slots
    assert twelve.ttft_s.p50 is not None and four.ttft_s.p50 is not None
    assert twelve.ttft_s.p50 > 2 * four.ttft_s.p50  # beyond capacity, requests queue
    assert one.tpot_s.p50 == pytest.approx(0.005, abs=0.004)  # decode rate measured, not queueing
    assert one.errors == 0 and one.requests == 24


def test_open_loop_and_error_accounting() -> None:
    async def go() -> tuple[bench.RunStats, bench.RunStats]:
        ok, broken = FakeServer(tokens=3, cached_tokens=21), FakeServer(status=500)
        async with ok.running() as url, broken.running() as bad_url:
            good = OpenAICompatibleClient(url, "m")
            bad = OpenAICompatibleClient(bad_url, "m")
            a = await bench.open_loop(good, [MSG] * 10, rate_per_s=50, max_tokens=3, seed=3)
            b = await bench.closed_loop(bad, [MSG] * 5, concurrency=2, max_tokens=3)
            await good.aclose()
            await bad.aclose()
        return a, b

    a, b = asyncio.run(go())
    assert a.requests == 10 and a.errors == 0 and a.prefix_cache_hit_rate == 0.5  # 21 / 42
    assert b.errors == 5 and b.ttft_s.n == 0 and b.requests_per_s == 0.0


def test_poisson_arrivals_are_seeded_and_have_the_right_rate() -> None:
    a = bench.poisson_arrivals(2000, 10.0, seed=7)
    assert a == bench.poisson_arrivals(2000, 10.0, seed=7) and a == sorted(a)
    assert 2000 / a[-1] == pytest.approx(10.0, rel=0.1)


def test_percentiles_carry_their_sample_size() -> None:
    d = bench.dist([float(x) for x in range(50)])
    assert d.n == 50 and d.unreliable == ["p99"]
    assert bench.dist([]).p50 is None


def test_few_shot_workload_shares_one_prefix_and_never_reuses_a_question() -> None:
    shots = [_ex(100), _ex(101, "no")]
    prompts = bench.few_shot([_ex(1), _ex(2)], shots)
    assert prompts[0][:-1] == prompts[1][:-1] and len(prompts[0]) == 1 + 2 * len(shots) + 1
    assert (
        prompts[0][2]["content"] == "yes" and prompts[0][-1]["content"] != prompts[1][-1]["content"]
    )
    with pytest.raises(ValueError, match="also a question"):
        bench.few_shot([_ex(100)], shots)


# --------------------------------------------------------------------------- serving command


def test_vllm_command_for_each_configuration() -> None:
    base = command(ServeConfig())
    assert base[:3] == ["vllm", "serve", "Qwen/Qwen3-4B-Instruct-2507"]
    assert "--enable-prefix-caching" in base and "--enable-prompt-tokens-details" in base
    assert base[base.index("--max-model-len") + 1] == "4096"
    off = command(ServeConfig(prefix_caching=False))
    assert "--no-enable-prefix-caching" in off and "--enable-prefix-caching" not in off
    nf4 = command(ServeConfig(precision="nf4", lora_adapters={"pubmedqa": "runs/qlora"}))
    assert nf4[nf4.index("--quantization") + 1] == "bitsandbytes"
    assert "--enable-lora" in nf4
    assert (
        nf4[nf4.index("--max-lora-rank") + 1] == "16" and nf4[nf4.index("--max-loras") + 1] == "1"
    )
    assert nf4[nf4.index("--lora-modules") + 1 :] == ["pubmedqa=runs/qlora"]


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (ServeConfig(precision="int8"), "int8"),
        (ServeConfig(max_lora_rank=24), "max-lora-rank"),
        (ServeConfig(lora_adapters={"a": "runs/x,y"}), "comma"),
    ],
)
def test_vllm_command_refuses_what_vllm_would_reject(cfg: ServeConfig, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        command(cfg)


# --------------------------------------------------------------------------- CLI


def _preds(path: Path, right: int, n: int = 100) -> str:
    write_predictions(
        path,
        [
            Prediction(pubid=str(i), gold="yes", raw="x", pred="yes" if i < right else "no")
            for i in range(n)
        ],
    )
    return str(path)


def test_cli_compare_and_gate_on_prediction_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ref, cand = _preds(tmp_path / "claude.jsonl", 80), _preds(tmp_path / "qlora.jsonl", 70)
    assert main(["compare", ref, cand, "--reference", "claude"]) == 0
    out = capsys.readouterr().out
    assert "qlora" in out and "-10.0%" in out and "only claude right: 10" in out
    assert main(["gate", "--candidate", cand, "--baseline", ref]) == 1
    assert "REFUSE" in capsys.readouterr().out
    assert main(["gate", "--candidate", ref, "--baseline", ref]) == 0


def test_cli_memory_from_a_local_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from transformers import Qwen3Config

    Qwen3Config(
        vocab_size=151936,
        hidden_size=2560,
        intermediate_size=9728,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        tie_word_embeddings=True,
    ).save_pretrained(tmp_path)
    assert main(["memory", "--config", str(tmp_path), "--gpu-gib", "24"]) == 0
    out = capsys.readouterr().out
    assert "4.022B params" in out and "144 KiB per token" in out and "nf4" in out


def test_cli_serve_prints_the_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["serve", "--adapter", "pubmedqa=runs/qlora", "--no-prefix-caching"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("vllm serve") and "--no-enable-prefix-caching" in out


# --------------------------------------------------------------------------- assets


def test_download_lists_without_fetching(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["download", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-4B-Instruct-2507" in out and "qiaojin/PubMedQA" in out
    assert "nothing downloaded" in out


def test_download_fetches_models_and_runs_the_dataset_loader() -> None:
    from sovereign import assets

    fetched: list[dict[str, object]] = []
    loaded: list[str] = []

    def fake_snapshot(**kw: object) -> str:
        fetched.append(kw)
        return "/cache/qwen"

    lines = assets.download(
        assets.find([]),
        cache_dir="/cache",
        fetch_model=fake_snapshot,
        load_dataset={"pubmedqa": lambda: loaded.append("pubmedqa")},
    )
    assert fetched == [
        {"repo_id": "Qwen/Qwen3-4B-Instruct-2507", "repo_type": "model", "cache_dir": "/cache"}
    ]
    assert loaded == ["pubmedqa"] and lines[0] == "qwen3-4b: /cache/qwen"
    with pytest.raises(ValueError, match="unknown asset"):
        assets.find(["llama"])


def test_serve_says_how_to_install_vllm_and_checks_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="requirements-serve.txt"):
        main(["serve", "--exec"])
    with pytest.raises(SystemExit, match="name=path"):
        main(["serve", "--adapter", "runs/qlora"])


def test_a_missing_training_library_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    from sovereign import cli

    monkeypatch.setitem(sys.modules, "llmcompressor", None)
    monkeypatch.setattr(cli, "_splits", lambda: None)
    assert main(["compress", "runs/merged", "runs/out", "--scheme", "fp8"]) == 2
    assert "pip install -r requirements-compress.txt" in capsys.readouterr().err
