"""Load generation against an OpenAI-compatible server: what continuous batching and prefix caching
actually buy, measured from the client's side.

Two ways to load a server, because they answer different questions:

  - **Closed loop** (`concurrency` users, each sending its next request when the last one ends):
    throughput as a function of how many sequences are in flight. With continuous batching, output
    tokens/s should climb with concurrency while time-per-output-token degrades slowly — until the
    KV cache (see `quant.capacity`) or compute saturates. A static-batching server would show TTFT
    climbing in steps instead.
  - **Open loop** (Poisson arrivals at `rate` requests/s, sent whether or not earlier ones
    finished): latency under a given offered load, including queueing — the view an SLA needs.
    Closed loops hide queueing because a slow server slows its own clients down.

Per request: time to first token (TTFT, dominated by prefill and queueing), time per output token
after the first (TPOT, the decode rate a reader sees), and end-to-end latency.

Workloads: `zero_shot` sends each question alone; `few_shot` prepends the same K solved examples to
every request — a long shared prefix, which is what prefix caching (vLLM's automatic prefix
caching) reuses. Run the few-shot workload against the same server with prefix caching on and off;
the TTFT difference is the cache's worth.

Percentiles are reported with their sample size; a p99 from fewer than 100 requests is flagged —
below that, it is an interpolation toward the maximum, not a tail measurement.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel

from sovereign.clients import ChatClient, Messages
from sovereign.data import SYSTEM_PROMPT, Example, messages, user_prompt

RELIABLE_N = {50: 2, 95: 20, 99: 100}  # roughly 1 / (1 - p) samples before a percentile means much


class RequestRecord(BaseModel):
    ok: bool
    error: str | None = None
    sent_s: float  # relative to the start of the run
    ttft_s: float | None = None
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int | None = None

    @property
    def tpot_s(self) -> float | None:
        if self.ttft_s is None or self.output_tokens < 2:
            return None
        return (self.latency_s - self.ttft_s) / (self.output_tokens - 1)


class Dist(BaseModel):
    n: int
    p50: float | None
    p95: float | None
    p99: float | None
    unreliable: list[str]  # percentiles computed from too few samples


def dist(values: Sequence[float]) -> Dist:
    n = len(values)

    def pct(p: int) -> float | None:
        return round(float(np.percentile(values, p)), 4) if n else None

    return Dist(
        n=n,
        p50=pct(50),
        p95=pct(95),
        p99=pct(99),
        unreliable=[f"p{p}" for p, need in RELIABLE_N.items() if n < need],
    )


class RunStats(BaseModel):
    label: str
    mode: str  # "closed" | "open"
    load: float  # concurrency (closed) or offered requests/s (open)
    requests: int
    errors: int
    duration_s: float
    requests_per_s: float
    output_tokens_per_s: float
    ttft_s: Dist
    tpot_s: Dist
    latency_s: Dist
    prefix_cache_hit_rate: float | None  # cached / prompt tokens, when the server reports it


def summarize(
    label: str, mode: str, load: float, records: Sequence[RequestRecord], duration_s: float
) -> RunStats:
    ok = [r for r in records if r.ok]
    tokens = sum(r.output_tokens for r in ok)
    return RunStats(
        label=label,
        mode=mode,
        load=load,
        requests=len(records),
        errors=len(records) - len(ok),
        duration_s=round(duration_s, 4),
        requests_per_s=round(len(ok) / duration_s, 3) if duration_s else 0.0,
        output_tokens_per_s=round(tokens / duration_s, 2) if duration_s else 0.0,
        ttft_s=dist([r.ttft_s for r in ok if r.ttft_s is not None]),
        tpot_s=dist([t for r in ok if (t := r.tpot_s) is not None]),
        latency_s=dist([r.latency_s for r in ok]),
        prefix_cache_hit_rate=_hit_rate(ok),
    )


def _hit_rate(records: Sequence[RequestRecord]) -> float | None:
    reported = [r for r in records if r.cached_tokens is not None]
    prompt = sum(r.input_tokens for r in reported)
    return round(sum(r.cached_tokens or 0 for r in reported) / prompt, 4) if prompt else None


async def _send(client: ChatClient, prompt: Messages, max_tokens: int, t0: float) -> RequestRecord:
    sent = time.perf_counter() - t0
    try:
        c = await client.complete(prompt, max_tokens=max_tokens)
    except Exception as e:  # noqa: BLE001 - a failed request is a measurement, not a crash
        return RequestRecord(ok=False, error=f"{type(e).__name__}: {e}"[:300], sent_s=sent)
    return RequestRecord(
        ok=True,
        sent_s=sent,
        ttft_s=c.ttft_s,
        latency_s=c.latency_s,
        input_tokens=c.input_tokens,
        output_tokens=c.output_tokens,
        cached_tokens=c.cached_tokens,
    )


async def closed_loop(
    client: ChatClient,
    prompts: Sequence[Messages],
    *,
    concurrency: int,
    max_tokens: int,
    label: str = "",
) -> RunStats:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    queue: asyncio.Queue[Messages] = asyncio.Queue()
    for p in prompts:
        queue.put_nowait(p)
    records: list[RequestRecord] = []
    t0 = time.perf_counter()

    async def user() -> None:
        while not queue.empty():
            records.append(await _send(client, queue.get_nowait(), max_tokens, t0))

    await asyncio.gather(*(user() for _ in range(concurrency)))
    return summarize(label, "closed", concurrency, records, time.perf_counter() - t0)


def poisson_arrivals(n: int, rate_per_s: float, seed: int = 0) -> list[float]:
    if rate_per_s <= 0:
        raise ValueError("rate must be > 0")
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate_per_s)
        out.append(t)
    return out


async def open_loop(
    client: ChatClient,
    prompts: Sequence[Messages],
    *,
    rate_per_s: float,
    max_tokens: int,
    seed: int = 0,
    label: str = "",
) -> RunStats:
    arrivals = poisson_arrivals(len(prompts), rate_per_s, seed)
    t0 = time.perf_counter()

    async def at(when: float, prompt: Messages) -> RequestRecord:
        await asyncio.sleep(max(0.0, when - (time.perf_counter() - t0)))
        return await _send(client, prompt, max_tokens, t0)

    records = await asyncio.gather(*(at(w, p) for w, p in zip(arrivals, prompts, strict=True)))
    return summarize(label, "open", rate_per_s, list(records), time.perf_counter() - t0)


# --------------------------------------------------------------------------- workloads


def zero_shot(examples: Sequence[Example]) -> list[Messages]:
    return [messages(ex) for ex in examples]


def few_shot(examples: Sequence[Example], shots: Sequence[Example]) -> list[Messages]:
    """The same solved examples in front of every question: a long prefix shared by every
    request — the case prefix caching is for. `shots` must not overlap the questions."""
    shot_ids = {s.pubid for s in shots}
    if shot_ids & {e.pubid for e in examples}:
        raise ValueError("a few-shot example is also a question")
    prefix: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s in shots:
        prefix += [
            {"role": "user", "content": user_prompt(s)},
            {"role": "assistant", "content": s.label},
        ]
    return [[*prefix, {"role": "user", "content": user_prompt(ex)}] for ex in examples]
