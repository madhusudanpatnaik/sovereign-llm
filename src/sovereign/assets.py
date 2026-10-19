"""Everything this project downloads, in one list — fetched on the machine that runs it, never on
import, in tests, or while building.

`sovereign download --list` shows each asset, its size and what needs it; `sovereign download
[NAME ...]` fetches them. Models go into the Hugging Face cache (`HF_HOME`, or `--cache-dir`),
which is where transformers and vLLM look first, so later runs start without a download. The
dataset is fetched by calling the same loader the project uses, so it lands exactly where that
loader will look. Set `HF_TOKEN` for gated repositories and higher rate limits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Asset:
    name: str
    kind: Literal["model", "dataset"]
    repo_id: str
    size: str  # approximate, for planning disk
    needed_by: str
    gated: bool = False


ASSETS: tuple[Asset, ...] = (
    Asset(
        "qwen3-4b",
        "model",
        "Qwen/Qwen3-4B-Instruct-2507",
        "~8 GB",
        "train, merge, serve (vLLM), memory",
    ),
    Asset(
        "pubmedqa", "dataset", "qiaojin/PubMedQA", "~3 MB (pqa_labeled)", "data, train, eval, bench"
    ),
)


def find(names: Sequence[str]) -> list[Asset]:
    """The named assets (all of them when no name is given)."""
    if not names:
        return list(ASSETS)
    by_name = {a.name: a for a in ASSETS}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"unknown asset(s) {unknown}; choose from {sorted(by_name)}")
    return [by_name[n] for n in names]


def describe(assets: Sequence[Asset]) -> list[str]:
    rows = [f"{'name':<10} {'kind':<8} {'size':<20} {'repository':<32} needed by"]
    for a in assets:
        repo = a.repo_id + (" (gated)" if a.gated else "")
        rows.append(f"{a.name:<10} {a.kind:<8} {a.size:<20} {repo:<32} {a.needed_by}")
    return rows


def _load_pubmedqa() -> object:
    from sovereign import data

    return data.load_labeled()


_DATASET_LOADERS: dict[str, Callable[[], object]] = {"pubmedqa": _load_pubmedqa}


def download(
    assets: Sequence[Asset],
    *,
    cache_dir: str | None = None,
    fetch_model: Callable[..., Any] | None = None,
    load_dataset: dict[str, Callable[[], object]] | None = None,
) -> list[str]:
    """Fetch each asset; returns one line per asset saying where it went."""
    if fetch_model is None:
        from huggingface_hub import snapshot_download

        fetch_model = snapshot_download
    loaders = _DATASET_LOADERS if load_dataset is None else load_dataset
    done = []
    for a in assets:
        if a.kind == "model":
            path = fetch_model(repo_id=a.repo_id, repo_type="model", cache_dir=cache_dir)
            done.append(f"{a.name}: {path}")
        else:
            loaders[a.name]()
            done.append(f"{a.name}: cached by its loader")
    return done
