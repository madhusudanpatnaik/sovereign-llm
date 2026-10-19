from __future__ import annotations

import asyncio

import pytest

from sovereign import data
from sovereign.clients import Completion, Messages
from sovereign.data import Example
from sovereign.evaluate import (
    Prediction,
    align,
    paired_bootstrap,
    parse_label,
    predict,
    read_predictions,
    score,
    write_predictions,
)
from sovereign.gate import GatePolicy, decide

# --------------------------------------------------------------------------- data


def _row(pubid: int, label: str = "yes") -> dict[str, object]:
    """Shaped like a hub `pqa_labeled` row, including the two fields that must never leak."""
    return {
        "pubid": pubid,
        "question": f"Does treatment {pubid} improve outcomes?",
        "context": {
            "contexts": [f"Background for {pubid}.", "Methods and results."],
            "labels": ["BACKGROUND", "METHODS"],
            "meshes": ["Humans"],
            "reasoning_required_pred": list("LEAKED_ANNOTATOR_ANSWER"),
            "reasoning_free_pred": list("ALSO_LEAKED"),
        },
        "long_answer": "THE_CONCLUSION_SAYS_YES",
        "final_decision": label,
    }


def test_prompts_never_contain_the_conclusion_or_the_annotators_answers() -> None:
    [ex] = data.from_rows([_row(1)])
    rendered = " ".join(m["content"] for m in data.messages(ex))
    for leak in ("THE_CONCLUSION_SAYS_YES", "LEAKED", "ALSO_LEAKED", "BACKGROUND"):
        assert leak not in rendered
    assert "Background for 1." in rendered and ex.pubid == "1"


def test_unknown_labels_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        data.from_rows([_row(1, "probably")])


def test_the_official_test_set_is_pinned() -> None:
    ids = data.official_test_pmids()
    assert len(ids) == 500 and "12377809" in ids


def _examples(n: int) -> list[Example]:
    return data.from_rows([_row(i, ("yes", "no", "maybe")[i % 3]) for i in range(n)])


def test_split_uses_the_given_test_ids_and_is_deterministic_and_disjoint() -> None:
    exs = _examples(100)
    test_ids = frozenset(str(i) for i in range(0, 100, 2))
    a = data.split(exs, test_ids=test_ids, val_size=10)
    b = data.split(list(reversed(exs)), test_ids=test_ids, val_size=10)
    assert {e.pubid for e in a.test} == test_ids
    assert [e.pubid for e in a.val] == [e.pubid for e in b.val]  # independent of row order
    parts = [{e.pubid for e in s} for s in (a.train, a.val, a.test)]
    assert sum(map(len, parts)) == 100 and len(set().union(*parts)) == 100
    assert (len(a.train), len(a.val)) == (40, 10)


def test_split_refuses_missing_test_ids_and_duplicates() -> None:
    exs = _examples(10)
    with pytest.raises(ValueError, match="not in the data"):
        data.split(exs, test_ids=frozenset({"999"}), val_size=2)
    with pytest.raises(ValueError, match="duplicate"):
        data.split(exs + exs[:1], test_ids=frozenset({"1"}), val_size=2)


def test_training_messages_end_with_the_label() -> None:
    [ex] = data.from_rows([_row(7, "maybe")])
    chat = data.messages(ex, with_answer=True)
    assert [m["role"] for m in chat] == ["system", "user", "assistant"]
    assert chat[-1]["content"] == "maybe"


# --------------------------------------------------------------------------- parsing + scoring


@pytest.mark.parametrize(
    ("reply", "label"),
    [
        ("yes", "yes"),
        ("Yes.", "yes"),
        ("  No", "no"),
        ("**Maybe**", "maybe"),
        ("Answer: no", "no"),
        ("Final answer - yes", "yes"),
        ("It depends.", None),
        ("Yesterday's data", None),
        ("", None),
        ("The answer is yes", None),
    ],
)
def test_parse_label_is_strict(reply: str, label: str | None) -> None:
    assert parse_label(reply) == label


def _p(pubid: str, gold: str, pred: str | None, latency: float = 0.1) -> Prediction:
    # validated from plain strings, as a predictions file is read
    return Prediction.model_validate(
        {"pubid": pubid, "gold": gold, "raw": pred or "?", "pred": pred, "latency_s": latency}
    )


def test_scores_by_hand() -> None:
    preds = [
        _p("1", "yes", "yes"),
        _p("2", "yes", "yes"),
        _p("3", "yes", "no"),
        _p("4", "no", "no"),
        _p("5", "no", None),
        _p("6", "maybe", "yes"),
    ]
    s = score("sys", preds)
    assert s.accuracy == pytest.approx(3 / 6, abs=1e-4)
    assert s.invalid_rate == pytest.approx(1 / 6, abs=1e-4)
    assert s.majority_baseline == pytest.approx(3 / 6, abs=1e-4)
    # yes: P=2/3 R=2/3 F1=0.667; no: P=1/2 R=1/2 F1=0.5; maybe: 0
    assert s.per_class["yes"].f1 == pytest.approx(0.6667, abs=1e-3)
    assert s.per_class["no"].f1 == pytest.approx(0.5, abs=1e-3)
    assert s.macro_f1 == pytest.approx((0.6667 + 0.5 + 0) / 3, abs=1e-3)
    assert s.confusion["no"]["invalid"] == 1 and s.confusion["maybe"]["yes"] == 1


def test_paired_bootstrap() -> None:
    golds = ["yes"] * 100
    ref = [_p(str(i), g, "yes") for i, g in enumerate(golds)]
    same = paired_bootstrap("a", ref, "b", ref)
    assert (same.delta, same.ci_low, same.ci_high) == (0.0, 0.0, 0.0)
    worse = [_p(str(i), "yes", "no" if i < 10 else "yes") for i in range(100)]
    d = paired_bootstrap("worse", worse, "ref", ref, seed=1)
    assert d.delta == pytest.approx(-0.10)
    assert d.ci_low < -0.10 < d.ci_high < 0  # the drop is real and its interval excludes 0
    assert (d.candidate_only_right, d.reference_only_right) == (0, 10)


def test_align_refuses_different_question_sets() -> None:
    with pytest.raises(ValueError, match="different questions"):
        align([_p("1", "yes", "yes")], [_p("2", "yes", "yes")])


def test_predictions_round_trip(tmp_path: object) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "p.jsonl"
    preds = [_p("1", "yes", "yes"), _p("2", "no", None)]
    write_predictions(path, preds)
    assert read_predictions(path) == preds


class _Scripted:
    """A ChatClient answering from a table, slowest first, to check ordering under concurrency."""

    name = "scripted"

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    async def complete(self, messages: Messages, *, max_tokens: int) -> Completion:
        question = messages[-1]["content"]
        pubid = next(k for k in self.answers if f"treatment {k} " in question)
        await asyncio.sleep(0.01 * (10 - int(pubid)))
        return Completion(text=self.answers[pubid], latency_s=0.01, output_tokens=1)


def test_predict_keeps_input_order_and_counts_invalid_outputs() -> None:
    exs = _examples(6)
    answers = {e.pubid: ("Yes." if i % 2 else "I am not sure") for i, e in enumerate(exs)}
    preds = asyncio.run(predict(_Scripted(answers), exs, concurrency=3))
    assert [p.pubid for p in preds] == [e.pubid for e in exs]
    assert [p.pred for p in preds] == [None, "yes", None, "yes", None, "yes"]


# --------------------------------------------------------------------------- the gate


def _run(right: int, n: int = 200, invalid: int = 0, latency: float = 0.1) -> list[Prediction]:
    out = []
    for i in range(n):
        pred = None if i < invalid else ("yes" if i < right else "no")
        out.append(_p(str(i), "yes", pred, latency))
    return out


def test_gate_promotes_an_equivalent_candidate() -> None:
    d = decide("nf4", _run(150), "bf16", _run(150))
    assert d.promote and d.reasons == []


def test_gate_refuses_a_drop_it_cannot_rule_out() -> None:
    d = decide("nf4", _run(140), "bf16", _run(150), GatePolicy(max_accuracy_drop=0.02))
    assert not d.promote and "cannot rule out a drop" in d.reasons[0]


def test_gate_refuses_format_regressions_and_slow_candidates() -> None:
    d = decide(
        "nf4", _run(150, invalid=5), "bf16", _run(150, invalid=5), GatePolicy(max_invalid_rate=0.01)
    )
    assert not d.promote and "invalid-output rate" in d.reasons[0]
    slow = decide(
        "nf4", _run(150, latency=2.0), "bf16", _run(150), GatePolicy(max_p95_latency_s=1.0)
    )
    assert not slow.promote and "p95 latency" in slow.reasons[0]


def test_gate_refuses_runs_on_different_questions() -> None:
    d = decide("nf4", _run(150, n=200), "bf16", _run(150, n=199))
    assert not d.promote and d.reasons[0].startswith("not comparable")
