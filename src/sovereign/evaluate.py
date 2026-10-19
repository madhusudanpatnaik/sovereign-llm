"""Scoring: the same predictions file for every system, scored the same way.

- **Parsing is strict.** A reply counts only if it *starts* with yes / no / maybe (after an
  optional "answer:"), case and punctuation aside. Anything else is an invalid output — counted,
  reported, and scored as wrong, never coerced to the nearest label. A small model that rambles
  should lose points for it.
- **Accuracy and macro-F1.** The labels are imbalanced (55% yes), so accuracy alone rewards a
  model that says "yes" to everything; macro-F1 does not. The majority-class baseline is reported
  beside every result.
- **Paired bootstrap.** Two systems answer the *same* questions, so their difference is measured
  pairwise: resample questions, recompute the accuracy difference, take the 2.5th / 97.5th
  percentiles. With 500 questions an unpaired comparison would need a much larger gap to mean
  anything; pairing removes the between-question variance both systems share.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from sovereign.data import LABELS, Example, Label, messages

if TYPE_CHECKING:
    from sovereign.clients import ChatClient

_PREFIX = re.compile(r"^\s*(?:final\s+)?answer\s*[:\-]\s*", re.I)
_FIRST_WORD = re.compile(r"[A-Za-z]+")


def parse_label(reply: str) -> Label | None:
    """'Yes.' -> 'yes'; 'Answer: maybe' -> 'maybe'; 'It depends' -> None."""
    text = _PREFIX.sub("", reply.strip().strip("*`\"' "))
    m = _FIRST_WORD.match(text)
    word = m.group(0).lower() if m else ""
    for label in LABELS:
        if word == label:
            return label
    return None


class Prediction(BaseModel):
    pubid: str
    gold: Label
    raw: str
    pred: Label | None
    latency_s: float = 0.0
    ttft_s: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def correct(self) -> bool:
        return self.pred == self.gold


class ClassScore(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class Scores(BaseModel):
    system: str
    n: int
    accuracy: float
    macro_f1: float
    invalid_rate: float
    majority_baseline: float
    per_class: dict[str, ClassScore]
    confusion: dict[str, dict[str, int]]  # gold -> predicted (incl. "invalid") -> count
    p50_latency_s: float
    p95_latency_s: float
    output_tokens: int


def score(system: str, predictions: Sequence[Prediction]) -> Scores:
    if not predictions:
        raise ValueError(f"{system}: no predictions to score")
    n = len(predictions)
    columns = [*LABELS, "invalid"]
    confusion: dict[str, dict[str, int]] = {g: dict.fromkeys(columns, 0) for g in LABELS}
    for p in predictions:
        confusion[p.gold][p.pred or "invalid"] += 1
    per_class: dict[str, ClassScore] = {}
    for label in LABELS:
        tp = confusion[label][label]
        predicted = sum(confusion[g][label] for g in LABELS)
        support = sum(confusion[label].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = ClassScore(
            precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4), support=support
        )
    present = [c for c in per_class.values() if c.support]
    latencies = [p.latency_s for p in predictions]
    gold_counts = [sum(1 for p in predictions if p.gold == label) for label in LABELS]
    return Scores(
        system=system,
        n=n,
        accuracy=round(sum(p.correct for p in predictions) / n, 4),
        macro_f1=round(sum(c.f1 for c in present) / len(present), 4),
        invalid_rate=round(sum(p.pred is None for p in predictions) / n, 4),
        majority_baseline=round(max(gold_counts) / n, 4),
        per_class=per_class,
        confusion=confusion,
        p50_latency_s=round(float(np.percentile(latencies, 50)), 4),
        p95_latency_s=round(float(np.percentile(latencies, 95)), 4),
        output_tokens=sum(p.output_tokens for p in predictions),
    )


class PairedDelta(BaseModel):
    """Accuracy of `candidate` minus `reference` on the same questions, with a 95% CI."""

    candidate: str
    reference: str
    n: int
    delta: float
    ci_low: float
    ci_high: float
    candidate_only_right: int  # discordant pairs: the questions that decide the difference
    reference_only_right: int


def align(a: Sequence[Prediction], b: Sequence[Prediction]) -> list[tuple[Prediction, Prediction]]:
    """Pair two systems' predictions by question. Refuses unless both answered the same set."""
    by_id = {p.pubid: p for p in b}
    ids_a = [p.pubid for p in a]
    if len(set(ids_a)) != len(ids_a) or len(by_id) != len(b):
        raise ValueError("duplicate question ids in a predictions file")
    if set(ids_a) != set(by_id):
        missing = len(set(ids_a) ^ set(by_id))
        raise ValueError(f"the two systems answered different questions ({missing} not shared)")
    return [(p, by_id[p.pubid]) for p in a]


def paired_bootstrap(
    candidate: str,
    cand_preds: Sequence[Prediction],
    reference: str,
    ref_preds: Sequence[Prediction],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> PairedDelta:
    pairs = align(cand_preds, ref_preds)
    c = np.array([p.correct for p, _ in pairs], dtype=np.float64)
    r = np.array([q.correct for _, q in pairs], dtype=np.float64)
    diff = c - r
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    deltas = diff[idx].mean(axis=1)
    return PairedDelta(
        candidate=candidate,
        reference=reference,
        n=len(diff),
        delta=round(float(diff.mean()), 4),
        ci_low=round(float(np.percentile(deltas, 2.5)), 4),
        ci_high=round(float(np.percentile(deltas, 97.5)), 4),
        candidate_only_right=int(((c == 1) & (r == 0)).sum()),
        reference_only_right=int(((c == 0) & (r == 1)).sum()),
    )


def write_predictions(path: str | Path, predictions: Iterable[Prediction]) -> None:
    with Path(path).open("w") as f:
        for p in predictions:
            f.write(p.model_dump_json() + "\n")


def read_predictions(path: str | Path) -> list[Prediction]:
    with Path(path).open() as f:
        return [Prediction.model_validate(json.loads(line)) for line in f if line.strip()]


async def predict(
    client: ChatClient,
    examples: Sequence[Example],
    *,
    concurrency: int = 8,
    max_tokens: int = 8,
) -> list[Prediction]:
    """Ask every question, `concurrency` at a time; results in input order. The answer is one
    word, so `max_tokens` stays small — a model that needs more has not followed the format."""
    gate = asyncio.Semaphore(concurrency)

    async def one(ex: Example) -> Prediction:
        async with gate:
            c = await client.complete(messages(ex), max_tokens=max_tokens)
        return Prediction(
            pubid=ex.pubid,
            gold=ex.label,
            raw=c.text,
            pred=parse_label(c.text),
            latency_s=c.latency_s,
            ttft_s=c.ttft_s,
            input_tokens=c.input_tokens,
            output_tokens=c.output_tokens,
        )

    return list(await asyncio.gather(*(one(ex) for ex in examples)))
