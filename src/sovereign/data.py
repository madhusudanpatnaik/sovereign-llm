"""PubMedQA: the task the small-vs-frontier gap is measured on.

`qiaojin/PubMedQA`, config `pqa_labeled`: 1,000 questions written from PubMed abstract titles, each
labelled yes / no / maybe by experts (552 / 338 / 110). Every model sees the same thing: the
abstract's *contexts* (the abstract without its conclusion) and the question — the paper's
"reasoning-required" setting, the one published numbers use.

Two things in the hub rows must never reach a prompt:
  - `long_answer`, the abstract's conclusion — with it the task becomes reading the answer off
    the page (the "reasoning-free" setting; one human scores 90% there vs 78% without);
  - the `context` dict's `reasoning_required_pred` / `reasoning_free_pred` fields, which are the
    two human annotators' own answers. Only `context["contexts"]` is ever read.

Splits: the test set is the official one — the 500 PMIDs of `data/test_ground_truth.json` in the
PubMedQA repository, pinned by commit and checksum in `pubmedqa_test_pmids.txt` (276 yes / 169 no
/ 55 maybe; majority-class accuracy 55.2%, macro-F1 23.7). Numbers here are therefore comparable
with published ones. The other 500 questions are split deterministically, by a salted hash of the
PubMed id, into 450 for fine-tuning and 50 for validation.

Why not `pqa_artificial` (211k auto-labelled questions) for fine-tuning: its labels come from a
heuristic on the title's verb, it is 92.8% "yes" and contains no "maybe" at all, and its questions
are converted statement titles. Fine-tuning on it naively teaches a model never to say "maybe".

Why PubMedQA rather than BillSum: a three-way label makes the gap an exact, judge-free number with
a confidence interval; a ROUGE score on bill summaries measures word overlap, not correctness.

Caveat no split can fix: PubMedQA has been public since 2019 and may be in any model's
pre-training data. The *fine-tuning delta* is measured cleanly on held-out questions; a base
model's absolute score should be read with that in mind.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from importlib.resources import files
from typing import Any, Literal

from pydantic import BaseModel

Label = Literal["yes", "no", "maybe"]
LABELS: tuple[Label, ...] = ("yes", "no", "maybe")
DATASET = "qiaojin/PubMedQA"
CONFIG = "pqa_labeled"
SPLIT_SALT = "sovereign/pubmedqa/v1"

SYSTEM_PROMPT = (
    "You answer biomedical research questions using only the abstract you are given. "
    "Reply with exactly one word: yes, no, or maybe."
)


class Example(BaseModel):
    pubid: str
    question: str
    contexts: list[str]
    label: Label


class Splits(BaseModel):
    train: list[Example]
    val: list[Example]
    test: list[Example]


def from_rows(rows: Iterable[Mapping[str, Any]]) -> list[Example]:
    """Rows shaped like the hub's `pqa_labeled` records -> examples."""
    out = []
    for row in rows:
        label = str(row["final_decision"]).strip().lower()
        if label not in LABELS:
            raise ValueError(f"pubid {row['pubid']}: unknown label {label!r}")
        out.append(
            Example(
                pubid=str(row["pubid"]),
                question=str(row["question"]).strip(),
                contexts=[str(c).strip() for c in row["context"]["contexts"]],
                label=label,  # checked against LABELS above; pydantic validates the Literal
            )
        )
    return out


def load_labeled() -> list[Example]:
    """Download (or read the cached) `pqa_labeled` from the Hugging Face hub — a few MB."""
    from datasets import load_dataset

    return from_rows(load_dataset(DATASET, CONFIG, split="train"))


def official_test_pmids() -> frozenset[str]:
    """The 500 PMIDs of PubMedQA's official test set (see pubmedqa_test_pmids.txt)."""
    text = files("sovereign").joinpath("pubmedqa_test_pmids.txt").read_text()
    return frozenset(
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    )


def _order_key(pubid: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}:{pubid}".encode()).hexdigest()


def split(
    examples: list[Example], *, test_ids: frozenset[str] | None = None, val_size: int = 50
) -> Splits:
    """Official test set; the rest split into train / val by a salted hash of the PubMed id."""
    ids = [e.pubid for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate pubids: a question would land in two splits")
    test_ids = official_test_pmids() if test_ids is None else test_ids
    missing = test_ids - set(ids)
    if missing:
        raise ValueError(f"{len(missing)} test PMIDs are not in the data (e.g. {min(missing)})")
    test = [e for e in examples if e.pubid in test_ids]
    rest = sorted(
        (e for e in examples if e.pubid not in test_ids), key=lambda e: _order_key(e.pubid)
    )
    if val_size >= len(rest):
        raise ValueError("the validation split would leave nothing to train on")
    return Splits(test=test, val=rest[:val_size], train=rest[val_size:])


def user_prompt(example: Example) -> str:
    abstract = "\n\n".join(example.contexts)
    return f"Abstract:\n{abstract}\n\nQuestion: {example.question}\nAnswer yes, no, or maybe."


def messages(example: Example, *, with_answer: bool = False) -> list[dict[str, str]]:
    """Chat messages for one example: the same prompt for every model, local or frontier. With
    `with_answer`, the gold label is appended as the assistant turn — a training example."""
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(example)},
    ]
    if with_answer:
        chat.append({"role": "assistant", "content": example.label})
    return chat


def label_counts(examples: Iterable[Example]) -> dict[str, int]:
    counts: dict[str, int] = dict.fromkeys(LABELS, 0)
    for e in examples:
        counts[e.label] += 1
    return counts
