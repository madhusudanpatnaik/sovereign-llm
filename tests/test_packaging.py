"""The requirements files are generated from pyproject.toml; this keeps them from drifting."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _requirements(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith(("#", "-r ", "-e "))]


def test_requirements_files_match_pyproject() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert _requirements(ROOT / "requirements.txt") == project["dependencies"]
    extras = project.get("optional-dependencies", {})
    for extra, deps in extras.items():
        assert _requirements(ROOT / f"requirements-{extra}.txt") == deps, extra
    on_disk = {p.name for p in ROOT.glob("requirements-*.txt")}
    assert on_disk == {f"requirements-{e}.txt" for e in extras}
