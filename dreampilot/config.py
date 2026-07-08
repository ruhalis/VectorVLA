"""One load point for everything read from disk/env: .env, measured.json, worlds.json.

Re-measuring the deployment (tests/record.py + tests/analyze.py) just
regenerates measured.json; adding a world is a worlds.json entry plus a seed
image. Nothing else in the codebase re-reads these files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Union[str, Path, None] = None) -> None:
    """Set os.environ from a .env file for variables not already set."""
    p = Path(path) if path is not None else ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Measured:
    """Deployment constants from the characterization run (run_001)."""

    latency_ms: int          # action command -> visible effect
    chunk_hz: Optional[float]
    fps: Optional[float]     # content fps (docs say 16; run_001 measured ~40)
    raw: dict                # full measured.json for anything else

    @property
    def action_to_effect_s(self) -> float:
        return self.latency_ms / 1000


def load_measured(path: Union[str, Path, None] = None) -> Measured:
    raw = json.loads(Path(path or ROOT / "measured.json").read_text())
    return Measured(
        latency_ms=raw["latency_ms"],
        chunk_hz=raw.get("chunk_hz"),
        fps=raw.get("fps"),
        raw=raw,
    )


@dataclass(frozen=True)
class World:
    name: str
    image: Path              # resolved against the repo root
    prompt: str              # static scene description — NO motion verbs
    commands: tuple[str, ...]  # judge-ready example commands


def load_worlds(path: Union[str, Path, None] = None) -> dict[str, World]:
    raw = json.loads(Path(path or ROOT / "worlds.json").read_text())
    return {
        name: World(
            name=name,
            image=ROOT / w["image"],
            prompt=w["prompt"],
            commands=tuple(w.get("commands", ())),
        )
        for name, w in raw.items()
    }
