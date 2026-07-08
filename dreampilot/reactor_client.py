"""Session wrapper around reactor-sdk for VectorVLA.

Every live Reactor session goes through this module so that:
  - the credit meter logs the burn (client-side: READY wall-clock x 33 credits/s),
  - every send_command is gated on READY (the SDK silently no-ops otherwise),
  - the frame callback only copies into a drop-oldest ring buffer,
  - every server message (chunk_complete above all) is logged to events.jsonl,
  - manual reconnect is wired (on_error recoverable -> reconnect()),
  - the 20-min session expiry is watched,
  - staging follows the verified sequence: upload -> set_image -> wait
    image_accepted -> set_prompt -> start.

Usage:
    session = ReactorSession(api_key=os.environ["REACTOR_API_KEY"],
                             run_dir=Path("data/measure/run_001"))
    await session.connect()
    await session.stage_world("seed.jpg", "A cobblestone village square...")
    ...
    await session.disconnect()
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
from reactor_sdk import Reactor, ReactorStatus

from dreampilot.actions import LOOKS_H, LOOKS_V, MOVEMENTS

logger = logging.getLogger("vectorvla.client")

# Billing facts (SDK_NOTES.md section 6) - verified 2026-07.
CREDITS_PER_SECOND = 33
CREDITS_PER_DOLLAR = 10_000
SESSION_HARD_CAP_S = 1200  # 20 min, then the server terminates.

# Documented as 16 fps but run_001 measured ~40 fps content rate (24-frame
# chunks at 1.65 Hz). Downstream code reads the real value from measured.json.
STREAM_FPS_DOCUMENTED = 16


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    # DEBUG on the SDK logger prints every frame message - keep it at INFO.
    logging.getLogger("reactor_sdk").setLevel(logging.INFO)


class EventLog:
    """Append-only JSONL log; every entry gets monotonic + wall timestamps."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", buffering=1)
        self._lock = threading.Lock()

    def write(self, kind: str, **fields: Any) -> dict:
        entry = {"t": time.monotonic(), "wall": time.time(), "kind": kind, **fields}
        with self._lock:
            self._f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def close(self) -> None:
        self._f.close()


class FrameWriter:
    """Writes frames to JPEG on a worker thread; drops oldest under backpressure.

    The on_frame callback must never block, so frames are handed off through a
    bounded queue and encoded off the event loop.
    """

    def __init__(self, frames_dir: Path, event_log: EventLog, quality: int = 88):
        from PIL import Image  # imported here so headless analysis never needs it

        self._Image = Image
        self.frames_dir = frames_dir
        frames_dir.mkdir(parents=True, exist_ok=True)
        self.event_log = event_log
        self.quality = quality
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self.dropped = 0
        self.written = 0
        self._index_f = open(frames_dir.parent / "frames.jsonl", "a", buffering=1)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, index: int, t: float, frame: np.ndarray) -> None:
        try:
            self._q.put_nowait((index, t, frame))
        except queue.Full:
            try:
                self._q.get_nowait()  # drop oldest
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait((index, t, frame))
            except queue.Full:
                self.dropped += 1

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            index, t, frame = item
            name = f"{index:06d}.jpg"
            self._Image.fromarray(frame).save(
                self.frames_dir / name, quality=self.quality
            )
            self._index_f.write(json.dumps({"i": index, "t": t, "file": name}) + "\n")
            self.written += 1

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=30)
        self._index_f.close()
        if self.dropped:
            self.event_log.write("frame_writer", dropped=self.dropped, written=self.written)
            logger.warning("FrameWriter dropped %d frames (wrote %d)", self.dropped, self.written)


class CreditMeter:
    """Client-side billing: wall-clock seconds in READY x 33 credits/s.

    No billing events exist on the message channel; this is the only meter.
    pause does NOT stop it - only disconnecting does.
    """

    def __init__(self) -> None:
        self._ready_since: Optional[float] = None
        self._accumulated_s = 0.0

    def on_status(self, status: ReactorStatus) -> None:
        now = time.monotonic()
        if status == ReactorStatus.READY:
            if self._ready_since is None:
                self._ready_since = now
        else:
            if self._ready_since is not None:
                self._accumulated_s += now - self._ready_since
                self._ready_since = None

    @property
    def billed_seconds(self) -> float:
        extra = time.monotonic() - self._ready_since if self._ready_since else 0.0
        return self._accumulated_s + extra

    @property
    def credits(self) -> float:
        return self.billed_seconds * CREDITS_PER_SECOND

    @property
    def dollars(self) -> float:
        return self.credits / CREDITS_PER_DOLLAR

    def summary(self) -> str:
        return (
            f"billed {self.billed_seconds:.0f}s = {self.credits:.0f} credits"
            f" = ${self.dollars:.2f}"
        )


class ReactorSession:
    """One live LingBot session. Create, use, disconnect - never leave idle."""

    def __init__(
        self,
        api_key: str,
        run_dir: Path,
        model_name: str = "lingbot",
        frame_buffer_size: int = 64,
        save_frames: bool = False,
        auto_reconnect: bool = True,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventLog(self.run_dir / "events.jsonl")
        self.meter = CreditMeter()

        self.frames: deque = deque(maxlen=frame_buffer_size)  # (index, t, frame)
        self.frame_count = 0
        self.chunks: list[dict] = []  # every chunk_complete data payload (+ timestamps)
        self.expires_at: Optional[float] = None  # unix ts from session_expiration_changed
        self.action_state = {"movement": "idle", "look_horizontal": "idle", "look_vertical": "idle"}

        self._writer = FrameWriter(self.run_dir / "frames", self.events) if save_frames else None
        self._auto_reconnect = auto_reconnect
        self._waiters: dict[str, list[asyncio.Future]] = {}
        self._status_waiters: dict[ReactorStatus, list[asyncio.Future]] = {}
        self._billing_task: Optional[asyncio.Task] = None

        self.reactor = Reactor(model_name=model_name, api_key=api_key)
        self._wire_handlers()

    # ------------------------------------------------------------- handlers

    def _wire_handlers(self) -> None:
        @self.reactor.on_frame
        def on_frame(frame: np.ndarray) -> None:
            # Runs synchronously on the SDK event loop: copy and return, nothing else.
            t = time.monotonic()
            idx = self.frame_count
            self.frame_count += 1
            copied = frame.copy()
            self.frames.append((idx, t, copied))
            if self._writer is not None:
                self._writer.submit(idx, t, copied)

        @self.reactor.on_message
        def on_message(msg: dict) -> None:
            mtype = msg.get("type", "unknown")
            data = msg.get("data", {})
            self.events.write("message", type=mtype, data=data)
            if mtype == "chunk_complete":
                self.chunks.append({"t": time.monotonic(), "wall": time.time(), **data})
            elif mtype == "command_error":
                logger.warning("command_error: %s", data)
            self._resolve_waiters(self._waiters, mtype, msg)

        @self.reactor.on_status
        def on_status(status: ReactorStatus) -> None:
            self.meter.on_status(status)
            self.events.write("status", status=status.value)
            logger.info("status -> %s", status.value)
            self._resolve_waiters(self._status_waiters, status, status)

        @self.reactor.on_error
        def on_error(err) -> None:
            self.events.write(
                "error", code=err.code, message=err.message,
                recoverable=err.recoverable, component=err.component,
            )
            logger.error("reactor error %s (recoverable=%s): %s", err.code, err.recoverable, err.message)
            if err.recoverable and self._auto_reconnect:
                asyncio.get_running_loop().create_task(self._reconnect_later(err.retry_after or 3))

        def on_expiration(expires_at) -> None:
            self.expires_at = expires_at
            self.events.write("expiration", expires_at=expires_at)

        self.reactor.on("session_expiration_changed", on_expiration)

        def on_capabilities(caps) -> None:
            self.events.write("capabilities", capabilities=self._caps_to_dict(caps))

        self.reactor.on("capabilities_received", on_capabilities)

    @staticmethod
    def _caps_to_dict(caps: Any) -> Any:
        try:
            import dataclasses
            if dataclasses.is_dataclass(caps):
                return dataclasses.asdict(caps)
        except Exception:
            pass
        return caps if isinstance(caps, (dict, list)) else str(caps)

    @staticmethod
    def _resolve_waiters(table: dict, key: Any, value: Any) -> None:
        for fut in table.pop(key, []):
            if not fut.done():
                fut.set_result(value)

    async def _reconnect_later(self, delay: float) -> None:
        logger.warning("transport dropped; reconnecting in %.1fs (30s billed window)", delay)
        await asyncio.sleep(delay)
        try:
            await self.reactor.reconnect()
            self.events.write("reconnected")
        except Exception as e:
            self.events.write("reconnect_failed", error=str(e))
            logger.error("reconnect failed: %s", e)

    # ------------------------------------------------------------- waiting

    async def wait_for_message(self, mtype: str, timeout: float = 30.0) -> dict:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters.setdefault(mtype, []).append(fut)
        return await asyncio.wait_for(fut, timeout)

    async def wait_for_status(self, status: ReactorStatus, timeout: float = 300.0) -> None:
        if self.reactor.get_status() == status:
            return
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._status_waiters.setdefault(status, []).append(fut)
        await asyncio.wait_for(fut, timeout)

    async def wait_for_chunks(self, n: int, timeout: float = 60.0) -> None:
        """Block until n more chunk_complete messages arrive."""
        target = len(self.chunks) + n
        deadline = time.monotonic() + timeout
        while len(self.chunks) < target:
            if time.monotonic() > deadline:
                raise TimeoutError(f"waited {timeout}s for {n} chunks")
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------- lifecycle

    async def connect(self, ready_timeout: float = 300.0) -> None:
        self.events.write("connect_start")
        await self.reactor.connect()
        # WAITING (GPU queue) is free; billing starts at READY.
        await self.wait_for_status(ReactorStatus.READY, timeout=ready_timeout)
        self.events.write("ready", session_id=self.reactor.get_session_id())
        self._billing_task = asyncio.get_running_loop().create_task(self._billing_loop())

    async def _billing_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            remaining = ""
            if self.expires_at:
                remaining = f", session expires in {self.expires_at - time.time():.0f}s"
            logger.info("credit meter: %s%s", self.meter.summary(), remaining)

    async def disconnect(self) -> None:
        """Non-recoverable disconnect: DELETEs the session and stops the meter."""
        try:
            await self.zero_actions()
        except Exception:
            pass
        if self._billing_task:
            self._billing_task.cancel()
        await self.reactor.disconnect(recoverable=False)
        self.meter.on_status(ReactorStatus.DISCONNECTED)
        self.events.write(
            "disconnect", billed_seconds=round(self.meter.billed_seconds, 1),
            credits=round(self.meter.credits), dollars=round(self.meter.dollars, 3),
            frames_received=self.frame_count,
        )
        if self._writer:
            self._writer.close()
        logger.info("disconnected; %s; %d frames received", self.meter.summary(), self.frame_count)
        self.events.close()

    def seconds_remaining(self) -> Optional[float]:
        return None if self.expires_at is None else self.expires_at - time.time()

    # ------------------------------------------------------------- commands

    async def send(self, command: str, data: dict) -> None:
        if self.reactor.get_status() != ReactorStatus.READY:
            logger.warning("dropping %s: status=%s (send_command would no-op)",
                           command, self.reactor.get_status().value)
            return
        self.events.write("command", command=command, data={k: str(v) for k, v in data.items()})
        await self.reactor.send_command(command, data)

    async def stage_world(
        self,
        image: str | Path,
        prompt: str,
        seed: Optional[int] = None,
        rotation_speed_deg: Optional[float] = None,
        start: bool = True,
    ) -> None:
        """Verified staging sequence; 'start' before image+prompt errors out."""
        ref = await self.reactor.upload_file(str(image))
        await self.send("set_image", {"image": ref})
        await self.wait_for_message("image_accepted", timeout=30)
        await self.send("set_prompt", {"prompt": prompt})
        await self.wait_for_message("prompt_accepted", timeout=15)
        if seed is not None:
            await self.send("set_seed", {"seed": seed})
        if rotation_speed_deg is not None:
            await self.send("set_rotation_speed_deg", {"rotation_speed_deg": rotation_speed_deg})
        if start:
            await self.send("start", {})
            await self.wait_for_message("generation_started", timeout=60)
        self.events.write("staged", prompt=prompt, image=str(image), seed=seed)

    async def set_action(
        self,
        movement: Optional[str] = None,
        look_horizontal: Optional[str] = None,
        look_vertical: Optional[str] = None,
    ) -> None:
        """Send only changed axes; action state persists server-side until changed."""
        for axis, value, valid in (
            ("movement", movement, MOVEMENTS),
            ("look_horizontal", look_horizontal, LOOKS_H),
            ("look_vertical", look_vertical, LOOKS_V),
        ):
            if value is None or value == self.action_state[axis]:
                continue
            if value not in valid:
                raise ValueError(f"invalid {axis}={value!r}, must be one of {valid}")
            await self.send(f"set_{axis}", {axis: value})
            self.action_state[axis] = value

    async def zero_actions(self) -> None:
        """State persists forever otherwise - call on every episode end/reset."""
        await self.set_action(movement="idle", look_horizontal="idle", look_vertical="idle")

    def latest_frame(self) -> Optional[tuple[int, float, np.ndarray]]:
        return self.frames[-1] if self.frames else None
