"""ReplaySession: a zero-credit stand-in for ReactorSession fed by recorded frames.

Presents the exact surface the runner and the web control room use
(latest_frame / set_action / zero_actions / seconds_remaining / meter /
stage_world / connect / disconnect) but pumps JPEGs from a recorded run
(data/measure/run_001/frames by default) into the same (index, t, frame)
ring-buffer shape. The VLM is still called for real, so this is the offline
gate with a UI on it: develop and rehearse the whole demo surface without
opening a billed Reactor session.

Deliberately imports no reactor_sdk — replay must work on any machine with
the base deps.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from dreampilot.actions import ENUMS, IDLE_STATE

logger = logging.getLogger("vectorvla.replay")

REPLAY_FPS = 15  # display-smooth, cheap to decode; content rate is irrelevant offline
REPLAY_SESSION_CAP_S = 1200  # mirror the live 20-min cap so the countdown UI is honest


class ReplayMeter:
    """Same read surface as CreditMeter; replay burns nothing."""

    billed_seconds = 0.0
    credits = 0.0
    dollars = 0.0

    def summary(self) -> str:
        return "replay: 0 credits"


class ReplaySession:
    """One replay 'session'. Same lifecycle contract as ReactorSession."""

    is_replay = True

    def __init__(self, frames_dir: str | Path, run_dir: Optional[Path] = None,
                 fps: float = REPLAY_FPS, frame_buffer_size: int = 64):
        self.frames_dir = Path(frames_dir)
        self._files = sorted(self.frames_dir.glob("*.jpg"))
        if not self._files:
            raise FileNotFoundError(f"no .jpg frames in {self.frames_dir}")
        self.fps = fps
        self.frames: deque = deque(maxlen=frame_buffer_size)  # (index, t, frame)
        self.frame_count = 0
        self.action_state = dict(IDLE_STATE)
        self.meter = ReplayMeter()
        self._started_at: Optional[float] = None
        self._pump: Optional[asyncio.Task] = None

    # ------------------------------------------------------------- lifecycle

    async def connect(self, ready_timeout: float = 0.0) -> None:
        self._started_at = time.monotonic()
        logger.info("replay session over %d frames from %s", len(self._files), self.frames_dir)

    async def stage_world(self, image: str | Path, prompt: str,
                          seed: Optional[int] = None,
                          rotation_speed_deg: Optional[float] = None,
                          start: bool = True) -> None:
        await asyncio.sleep(1.0)  # a beat of "staging" so the UI flow reads the same
        if self._pump is None:
            self._pump = asyncio.get_running_loop().create_task(self._pump_frames())
        logger.info("replay staged (image=%s ignored; pumping recorded frames)", image)

    async def disconnect(self) -> None:
        if self._pump:
            self._pump.cancel()
            self._pump = None
        logger.info("replay session done: %s", self.meter.summary())

    def seconds_remaining(self) -> Optional[float]:
        if self._started_at is None:
            return None
        return REPLAY_SESSION_CAP_S - (time.monotonic() - self._started_at)

    # ------------------------------------------------------------- frames

    def _load(self, path: Path) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))

    async def _pump_frames(self) -> None:
        period = 1.0 / self.fps
        while True:
            path = self._files[self.frame_count % len(self._files)]
            frame = await asyncio.to_thread(self._load, path)
            self.frames.append((self.frame_count, time.monotonic(), frame))
            self.frame_count += 1
            await asyncio.sleep(period)

    def latest_frame(self) -> Optional[tuple[int, float, np.ndarray]]:
        return self.frames[-1] if self.frames else None

    # ------------------------------------------------------------- actions

    async def set_action(self, movement: Optional[str] = None,
                         look_horizontal: Optional[str] = None,
                         look_vertical: Optional[str] = None) -> None:
        """Same validate/only-on-change semantics as the live session."""
        for axis, value in (("movement", movement),
                            ("look_horizontal", look_horizontal),
                            ("look_vertical", look_vertical)):
            if value is None or value == self.action_state[axis]:
                continue
            if value not in ENUMS[axis]:
                raise ValueError(f"invalid {axis}={value!r}, must be one of {ENUMS[axis]}")
            self.action_state[axis] = value
            logger.info("replay set_%s=%s (no world to move — recorded frames)", axis, value)

    async def zero_actions(self) -> None:
        await self.set_action(movement="idle", look_horizontal="idle", look_vertical="idle")
