"""One async interface for every model: a local server, a local `generate`, or a frontier API.

  - `OpenAICompatibleClient` talks to vLLM (or llama.cpp's server, Ollama, SGLang — anything that
    speaks the OpenAI chat-completions API). It always streams, so every request records the time
    to the first token and the gap between chunks — the numbers continuous batching moves. Token
    counts come from the final usage chunk (a streamed chunk can carry several tokens), and so does
    the number of prompt tokens the prefix cache served, when the server reports it. With vLLM, a
    LoRA adapter is chosen per request by passing its served name as `model`.
  - `TransformersClient` runs `model.generate` in-process: evaluation without a server, and the CPU
    smoke run. Calls are serialized (one forward at a time); it is not a throughput path.
  - `AnthropicClient` is the frontier reference. Same messages, same parser, so the small-vs-
    frontier gap is measured on identical inputs. Server-side refusal fallback on; a refusal or a
    max_tokens cut is returned as an empty answer with its stop reason, which the scorer counts as
    invalid, rather than raised — one refused question must not abort a 500-question run.

All three return a `Completion` with wall-clock latency; streaming clients add token timing.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field

Messages = Sequence[dict[str, str]]


class Completion(BaseModel):
    text: str
    latency_s: float
    ttft_s: float | None = None
    token_times_s: list[float] = Field(default_factory=list)  # arrival of each streamed chunk
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int | None = None  # prompt tokens served from the prefix cache, if reported
    stop_reason: str | None = None

    @property
    def inter_token_s(self) -> list[float]:
        t = self.token_times_s
        return [b - a for a, b in zip(t, t[1:], strict=False)]


class ChatClient(Protocol):
    name: str

    async def complete(self, messages: Messages, *, max_tokens: int) -> Completion: ...


# --------------------------------------------------------------------------- OpenAI-compatible


async def sse_events(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
    """Parse a server-sent-events stream of `data: {json}` lines, ending at `data: [DONE]`."""
    async for line in lines:
        if not line.startswith("data:"):
            continue  # comments, event names, keep-alives, blank separators
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        if payload:
            yield json.loads(payload)


def choice_constraint(choices: Sequence[str]) -> dict[str, Any]:
    """vLLM structured outputs (0.12+): the reply is decoded as exactly one of `choices`. Pass it
    as `extra_body`; it removes the format failures strict parsing would otherwise count."""
    return {"structured_outputs": {"choice": list(choices)}}


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        timeout_s: float = 300.0,
        client: Any = None,
        clock: Callable[[], float] = time.perf_counter,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.name = model
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout_s
        self._client = client
        self._clock = clock
        self._extra = dict(extra_body or {})

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
            self._client = httpx.AsyncClient(timeout=self._timeout, limits=limits)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def complete(self, messages: Messages, *, max_tokens: int) -> Completion:
        body = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": 0.0,  # greedy: an eval should be repeatable
            "stream": True,
            "stream_options": {"include_usage": True},
            **self._extra,
        }
        started = self._clock()
        parts: list[str] = []
        times: list[float] = []
        usage: dict[str, Any] = {}
        finish: str | None = None
        async with self._http().stream("POST", self.url, json=body, headers=self._headers) as r:
            if r.status_code != 200:
                detail = (await r.aread()).decode(errors="replace")[:300]
                raise RuntimeError(f"{self.url} returned {r.status_code}: {detail}")
            async for event in sse_events(r.aiter_lines()):
                if "error" in event:  # vLLM reports a mid-stream failure inside the 200 stream
                    raise RuntimeError(f"{self.url} stream error: {event['error']}")
                for choice in event.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        parts.append(delta)
                        times.append(self._clock() - started)
                    finish = choice.get("finish_reason") or finish
                if event.get("usage"):
                    usage = event["usage"]
        return Completion(
            text="".join(parts),
            latency_s=self._clock() - started,
            ttft_s=times[0] if times else None,
            token_times_s=times,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", len(times))),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            stop_reason=finish,
        )


# ------------------------------------------------------------------- in-process transformers


class TransformersClient:
    def __init__(self, model: Any, tokenizer: Any, *, name: str = "transformers") -> None:
        self.name = name
        self._model = model
        self._tokenizer = tokenizer
        self._lock = asyncio.Lock()

    def _generate(self, messages: Messages, max_tokens: int) -> Completion:
        import torch

        started = time.perf_counter()
        text = self._tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        enc = self._tokenizer(text, add_special_tokens=False, return_tensors="pt")
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        pad = self._tokenizer.pad_token_id
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=pad if pad is not None else self._tokenizer.eos_token_id,
            )
        new = out[0, enc["input_ids"].shape[1] :]
        return Completion(
            text=self._tokenizer.decode(new, skip_special_tokens=True),
            latency_s=time.perf_counter() - started,
            input_tokens=int(enc["input_ids"].shape[1]),
            output_tokens=int(new.shape[0]),
        )

    async def complete(self, messages: Messages, *, max_tokens: int) -> Completion:
        async with self._lock:
            return await asyncio.to_thread(self._generate, messages, max_tokens)


# --------------------------------------------------------------------------- frontier reference


class AnthropicClient:
    """Claude as the frontier reference. Thinking is always on for Claude Opus 5.5, so
    `max_tokens` must leave room for it; the answer itself is one word."""

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        *,
        effort: str = "high",
        max_tokens: int = 16000,
        client: Any = None,
    ) -> None:
        self.name = model
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = client

    def _ensure(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def complete(self, messages: Messages, *, max_tokens: int) -> Completion:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        started = time.perf_counter()
        response = await self._ensure().beta.messages.create(
            model=self.model,
            max_tokens=max(self.max_tokens, max_tokens),
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort},
            system=system,
            messages=turns,
        )
        latency = time.perf_counter() - started
        stop = response.stop_reason
        text = (
            ""
            if stop in ("refusal", "max_tokens")
            else "".join(b.text for b in response.content if b.type == "text")
        )
        return Completion(
            text=text,
            latency_s=latency,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=stop,
        )
