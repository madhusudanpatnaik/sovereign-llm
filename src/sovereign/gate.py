"""THE GATE: an optimized model ships only if the optimization did not cost what it claims not to.

Every optimization here trades something: 4-bit weights trade accuracy for memory, a LoRA adapter
can regress the questions it was not trained on, a bigger batch trades latency for throughput. The
promotion gate compares a *candidate* (say, the fine-tuned model quantized to 4-bit) against a
named *baseline* (the same model in bf16) on the same held-out questions, and refuses unless:

  1. both answered exactly the same questions (otherwise the comparison means nothing);
  2. the accuracy change is *confidently* no worse than `max_accuracy_drop`: the lower end of the
     paired-bootstrap 95% interval must clear it — "the point estimate looks fine" is not enough;
  3. the candidate's invalid-output rate is at most `max_invalid_rate` — a model that stops
     following the answer format has regressed even if its valid answers are good;
  4. optionally, its p95 latency is within `max_p95_latency_s`.

Every refusal says which rule failed and by how much. The thresholds are product decisions; the
defaults are a starting point, not a recommendation.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from sovereign.evaluate import PairedDelta, Prediction, paired_bootstrap, score


class GatePolicy(BaseModel):
    max_accuracy_drop: float = Field(default=0.02, ge=0, le=1)
    max_invalid_rate: float = Field(default=0.01, ge=0, le=1)
    max_p95_latency_s: float | None = Field(default=None, gt=0)


class GateDecision(BaseModel):
    promote: bool
    candidate: str
    baseline: str
    reasons: list[str]
    delta: PairedDelta | None = None


def decide(
    candidate: str,
    cand_preds: Sequence[Prediction],
    baseline: str,
    base_preds: Sequence[Prediction],
    policy: GatePolicy | None = None,
) -> GateDecision:
    policy = policy or GatePolicy()
    try:
        delta = paired_bootstrap(candidate, cand_preds, baseline, base_preds)
    except ValueError as e:
        return GateDecision(
            promote=False, candidate=candidate, baseline=baseline, reasons=[f"not comparable: {e}"]
        )
    reasons: list[str] = []
    if delta.ci_low < -policy.max_accuracy_drop:
        reasons.append(
            f"accuracy change {delta.delta:+.1%} (95% CI {delta.ci_low:+.1%} to "
            f"{delta.ci_high:+.1%}) cannot rule out a drop larger than "
            f"{policy.max_accuracy_drop:.1%}"
        )
    cand = score(candidate, cand_preds)
    if cand.invalid_rate > policy.max_invalid_rate:
        reasons.append(
            f"invalid-output rate {cand.invalid_rate:.1%} exceeds {policy.max_invalid_rate:.1%}"
        )
    if policy.max_p95_latency_s is not None and cand.p95_latency_s > policy.max_p95_latency_s:
        reasons.append(
            f"p95 latency {cand.p95_latency_s:.3f}s exceeds {policy.max_p95_latency_s:.3f}s"
        )
    return GateDecision(
        promote=not reasons, candidate=candidate, baseline=baseline, reasons=reasons, delta=delta
    )
