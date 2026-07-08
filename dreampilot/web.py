"""DreamPilot web control room: the judge-facing demo surface (PIVOT.md section 6).

    .venv/bin/python -m dreampilot.web              # live (bills credits per second!)
    .venv/bin/python -m dreampilot.web --replay     # recorded run_001 frames, 0 credits

One aiohttp app in the same process as the control loop:
  GET  /            the single-page UI (dreampilot/static/index.html)
  GET  /stream      MJPEG of the ring buffer (multipart/x-mixed-replace)
  GET  /worlds      worlds.json entries + thumbnail URLs
  GET  /thumb/{n}   a world's seed image
  POST /upload      judge-supplied seed image (re-encoded to JPEG, max 1664 w)
  GET  /ws          control channel (JSON ops in, JSON events out)

Hard rules honored here:
  - live sessions go through ReactorSession only (credit meter, READY gating);
  - human drive and agent episodes are mutually exclusive, enforced server-side
    (both write the same persistent per-axis action state and would fight);
  - a READY session idle for --idle-limit seconds is auto-disconnected — the
    meter runs whenever the session is up, driving or not;
  - the process disconnects the session on shutdown (Ctrl-C never leaks a
    billed session);
  - episode logic stays in runner.run_episode — this module only feeds it
    commands and streams its decisions out.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from aiohttp import WSMsgType, web

from dreampilot.actions import AXES, IDLE_STATE
from dreampilot.config import ROOT, load_env, load_worlds
from dreampilot.frames import frame_to_jpeg
from dreampilot.policy import Decision, MODES
from dreampilot.replay import ReplaySession
from dreampilot.runner import (ACTION_TO_EFFECT_S, DEFAULT_PROMPT,
                               download_recording, run_episode)

logger = logging.getLogger("vectorvla.web")

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = ROOT / "data" / "uploads"
STREAM_FPS = 15
UPLOAD_MAX_BYTES = 32 * 1024 * 1024
SEED_IMAGE_MAX_W = 1664  # native stream width; larger uploads buy nothing


class RoomError(Exception):
    """Operator-facing rejection (wrong state, bad input) — not a server bug."""


def _is_replay(session: Any) -> bool:
    return bool(getattr(session, "is_replay", False))


class ControlRoom:
    """All mutable demo state + the ops the websocket exposes.

    States: idle -> connecting -> ready <-> episode -> disconnecting -> idle.
    Single event loop, no awaits between a state check and its transition, so
    the guards are race-free without extra locking; the lifecycle lock only
    serializes connect/disconnect against each other.
    """

    def __init__(self, replay_dir: Optional[Path], period_s: float,
                 default_timeout_s: float, idle_limit_s: float):
        self.replay_dir = replay_dir
        self.period_s = period_s
        self.default_timeout_s = default_timeout_s
        self.idle_limit_s = idle_limit_s

        self.session: Any = None
        self.state = "idle"
        self.world_name: Optional[str] = None
        self.run_dir: Optional[Path] = None
        self.episode_task: Optional[asyncio.Task] = None
        self.episode_command: Optional[str] = None
        self.sockets: set[web.WebSocketResponse] = set()
        self.last_activity = time.monotonic()
        self._lifecycle_lock = asyncio.Lock()

    # ------------------------------------------------------------ broadcast

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    async def _send(self, ws: web.WebSocketResponse, payload: dict) -> None:
        with contextlib.suppress(Exception):
            await ws.send_json(payload)

    async def broadcast(self, payload: dict) -> None:
        for ws in list(self.sockets):
            if ws.closed:
                self.sockets.discard(ws)
                continue
            await self._send(ws, payload)

    async def broadcast_log(self, text: str) -> None:
        await self.broadcast({"ev": "log", "text": text})

    def state_payload(self) -> dict:
        s = self.session
        expires = s.seconds_remaining() if s else None
        return {
            "ev": "state",
            "state": self.state,
            "world": self.world_name,
            "replay": self.replay_dir is not None,
            "credits": round(s.meter.credits) if s else 0,
            "dollars": round(s.meter.dollars, 2) if s else 0.0,
            "expires_in": round(expires) if expires is not None else None,
            "action": dict(s.action_state) if s else dict(IDLE_STATE),
            "command": self.episode_command,
        }

    async def broadcast_state(self) -> None:
        await self.broadcast(self.state_payload())

    # ------------------------------------------------------------ ops

    async def op_connect(self, p: dict) -> None:
        async with self._lifecycle_lock:
            if self.state != "idle":
                raise RoomError(f"already {self.state} — leave the current world first")
            if p.get("world"):
                world = load_worlds().get(p["world"])
                if world is None:
                    raise RoomError(f"unknown world {p['world']!r}")
                image, self.world_name = world.image, world.name
                prompt = (p.get("prompt") or "").strip() or world.prompt
            else:
                image = (ROOT / str(p.get("image", ""))).resolve()
                if not image.is_relative_to(ROOT) or not image.is_file():
                    raise RoomError("upload an image first, or pick a world")
                prompt = (p.get("prompt") or "").strip() or DEFAULT_PROMPT
                self.world_name = "custom"

            self.state = "connecting"
            await self.broadcast_state()
            await self.broadcast_log(f"staging {self.world_name!r}...")
            run_dir = ROOT / "data" / "live" / f"web_{datetime.now():%Y%m%d_%H%M%S}"
            session = None
            try:
                if self.replay_dir is not None:
                    session = ReplaySession(self.replay_dir)
                else:
                    from dreampilot.reactor_client import ReactorSession

                    session = ReactorSession(api_key=os.environ["REACTOR_API_KEY"],
                                             run_dir=run_dir)
                await session.connect()
                await session.stage_world(image, prompt,
                                          seed=p.get("seed"),
                                          rotation_speed_deg=p.get("rotation_speed"))
                self.session = session  # /stream picks frames up from here on
                while session.latest_frame() is None:
                    await asyncio.sleep(0.2)
                if not _is_replay(session):
                    # Let the model settle into the seed before anyone drives.
                    await asyncio.sleep(2 * ACTION_TO_EFFECT_S)
                self.run_dir = run_dir
                self.state = "ready"
                self.touch()
                await self.broadcast_state()
                await self.broadcast_log(f"world ready — {self.world_name}"
                                         + (" (replay)" if _is_replay(session) else ""))
            except Exception:
                self.session, self.world_name, self.state = None, None, "idle"
                if session is not None:
                    with contextlib.suppress(Exception):
                        await session.disconnect()
                await self.broadcast_state()
                raise

    async def op_disconnect(self) -> None:
        async with self._lifecycle_lock:
            if self.session is None:
                return
            if self.episode_task is not None:
                self.episode_task.cancel()
                await asyncio.gather(self.episode_task, return_exceptions=True)
            self.state = "disconnecting"
            await self.broadcast_state()
            session, self.session = self.session, None
            try:
                if not _is_replay(session) and self.run_dir is not None:
                    await download_recording(session, self.run_dir)
            finally:
                with contextlib.suppress(Exception):
                    await session.disconnect()
            summary = session.meter.summary()
            self.state, self.world_name, self.run_dir = "idle", None, None
            await self.broadcast_state()
            await self.broadcast_log(f"session closed — {summary}")

    def op_command(self, p: dict) -> None:
        text = str(p.get("text", "")).strip()
        mode = p.get("mode", "full")
        if not text:
            raise RoomError("type a command first")
        if mode not in MODES:
            raise RoomError(f"unknown mode {mode!r}, expected one of {sorted(MODES)}")
        if self.state == "episode":
            raise RoomError("an episode is already running — stop it first")
        if self.state != "ready":
            raise RoomError("enter a world first")
        timeout = float(p.get("timeout") or self.default_timeout_s)
        self.state = "episode"
        self.episode_command = text
        self.episode_task = asyncio.get_running_loop().create_task(
            self._episode(text, mode, timeout))

    async def _episode(self, text: str, mode: str, timeout: float) -> None:
        t0 = time.monotonic()
        success = False
        await self.broadcast({"ev": "episode_start", "command": text, "mode": mode})
        await self.broadcast_state()
        try:
            success = await run_episode(self.session, text, mode, timeout,
                                        self.period_s, expiry_margin_s=60,
                                        on_decision=self._on_decision)
            if success and self.session is not None and not _is_replay(self.session):
                await download_recording(self.session, self.run_dir)
        except asyncio.CancelledError:
            await self.broadcast_log(f"stopped: {text!r}")
            raise
        except Exception as e:  # noqa: BLE001 — surface to the operator, keep serving
            logger.exception("episode failed")
            await self.broadcast({"ev": "error", "text": f"episode failed: {e}"})
        finally:
            if self.session is not None:
                with contextlib.suppress(Exception):
                    await self.session.zero_actions()
            self.episode_task = None
            self.episode_command = None
            if self.state == "episode":  # not if a disconnect raced us
                self.state = "ready"
            self.touch()
            await self.broadcast({"ev": "episode_end", "success": success,
                                  "command": text,
                                  "elapsed": round(time.monotonic() - t0, 1)})
            await self.broadcast_state()

    def _on_decision(self, elapsed: float, d: Decision) -> None:
        self.touch()
        asyncio.get_running_loop().create_task(self.broadcast({
            "ev": "decision", "t": round(elapsed, 1),
            "movement": d.movement, "look_horizontal": d.look_horizontal,
            "look_vertical": d.look_vertical, "arrived": d.arrived,
            "ok": d.ok, "reasoning": d.reasoning,
            "latency_s": round(d.latency_s, 1),
        }))

    async def op_drive(self, p: dict) -> None:
        if self.state == "episode":
            raise RoomError("the agent is driving — stop the episode to take the wheel")
        if self.state != "ready":
            raise RoomError("enter a world first")
        axes = {axis: p[axis] for axis in AXES if p.get(axis)}
        await self.session.set_action(**axes)  # validates enums, sends changes only
        self.touch()
        await self.broadcast_state()  # snappy HUD update; 1 Hz tick otherwise

    async def op_stop(self) -> None:
        if self.episode_task is not None:
            self.episode_task.cancel()  # _episode's finally zeroes actions + broadcasts
        elif self.state == "ready":
            await self.session.zero_actions()
            await self.broadcast_state()

    async def dispatch(self, data: dict, ws: web.WebSocketResponse) -> None:
        op = data.get("op")
        try:
            if op == "connect":
                await self.op_connect(data)
            elif op == "command":
                self.op_command(data)
            elif op == "drive":
                await self.op_drive(data)
            elif op == "stop":
                await self.op_stop()
            elif op == "disconnect":
                await self.op_disconnect()
            else:
                raise RoomError(f"unknown op {op!r}")
        except (RoomError, ValueError) as e:
            await self._send(ws, {"ev": "error", "text": str(e)})
        except Exception as e:  # noqa: BLE001 — one bad op must not kill the socket
            logger.exception("op %r failed", op)
            await self._send(ws, {"ev": "error", "text": f"{op} failed: {e}"})

    # ------------------------------------------------------------ background

    async def tick(self) -> None:
        """1 Hz state broadcast + the idle watchdog (live sessions bill idle)."""
        while True:
            await asyncio.sleep(1.0)
            if self.sockets:
                await self.broadcast_state()
            idle_for = time.monotonic() - self.last_activity
            if (self.replay_dir is None and self.state == "ready"
                    and idle_for > self.idle_limit_s):
                await self.broadcast_log(
                    f"no activity for {idle_for:.0f}s — leaving the world "
                    f"(live sessions bill every second)")
                with contextlib.suppress(Exception):
                    await self.op_disconnect()


# ------------------------------------------------------------------ handlers


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


async def stream(request: web.Request) -> web.StreamResponse:
    """MJPEG: new ring-buffer frames, deduped by index, encoded off-loop."""
    room: ControlRoom = request.app["room"]
    resp = web.StreamResponse(headers={
        "Content-Type": "multipart/x-mixed-replace; boundary=frame",
        "Cache-Control": "no-store",
    })
    await resp.prepare(request)
    last_index = None
    with contextlib.suppress(ConnectionResetError, asyncio.CancelledError):
        while True:
            session = room.session
            latest = session.latest_frame() if session is not None else None
            if latest is None or latest[0] == last_index:
                await asyncio.sleep(1 / (2 * STREAM_FPS))
                continue
            last_index = latest[0]
            jpeg = await asyncio.to_thread(frame_to_jpeg, latest[2])
            await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                             + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                             + jpeg + b"\r\n")
            await asyncio.sleep(1 / STREAM_FPS)
    return resp


async def worlds(request: web.Request) -> web.Response:
    return web.json_response([
        {"name": w.name, "prompt": w.prompt, "commands": list(w.commands),
         "thumb": f"/thumb/{w.name}"}
        for w in load_worlds().values()
    ])


async def thumb(request: web.Request) -> web.FileResponse:
    world = load_worlds().get(request.match_info["name"])
    if world is None or not world.image.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(world.image)


async def upload(request: web.Request) -> web.Response:
    """Accept a judge-supplied seed image; normalize to JPEG <= 1664 wide."""
    reader = await request.multipart()
    field = await reader.next()
    if field is None or field.name != "image":
        raise web.HTTPBadRequest(text="expected multipart field 'image'")
    raw = await field.read(decode=False)

    def normalize(data: bytes) -> bytes:
        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.width > SEED_IMAGE_MAX_W:
            img = img.resize((SEED_IMAGE_MAX_W,
                              round(img.height * SEED_IMAGE_MAX_W / img.width)),
                             Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=90)
        return out.getvalue()

    try:
        jpeg = await asyncio.to_thread(normalize, raw)
    except Exception:
        raise web.HTTPBadRequest(text="that file isn't an image this server can read "
                                      "(use JPEG, PNG, or WebP)")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"upload_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    path.write_bytes(jpeg)
    return web.json_response({"path": str(path.relative_to(ROOT))})


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    room: ControlRoom = request.app["room"]
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    room.sockets.add(ws)
    await ws.send_json({"ev": "hello", "modes": sorted(MODES),
                        "replay": room.replay_dir is not None,
                        "default_prompt": DEFAULT_PROMPT})
    await ws.send_json(room.state_payload())
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            # Ops run as tasks: staging takes ~10-30 s and must not block
            # the receive loop (a Stop must be able to overtake a Go).
            asyncio.get_running_loop().create_task(room.dispatch(data, ws))
    finally:
        room.sockets.discard(ws)
    return ws


# ------------------------------------------------------------------ app


def build_app(room: ControlRoom) -> web.Application:
    app = web.Application(client_max_size=UPLOAD_MAX_BYTES)
    app["room"] = room
    app.router.add_get("/", index)
    app.router.add_get("/stream", stream)
    app.router.add_get("/worlds", worlds)
    app.router.add_get("/thumb/{name}", thumb)
    app.router.add_post("/upload", upload)
    app.router.add_get("/ws", ws_handler)

    async def background(app: web.Application):
        task = asyncio.get_running_loop().create_task(room.tick())
        yield
        task.cancel()

    async def on_shutdown(app: web.Application) -> None:
        # Credit safety: Ctrl-C must never leak a billed session.
        with contextlib.suppress(Exception):
            await room.op_disconnect()

    app.cleanup_ctx.append(background)
    app.on_shutdown.append(on_shutdown)
    return app


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", nargs="?", const="data/measure/run_001/frames",
                    default=None, metavar="FRAMES_DIR",
                    help="pump recorded frames instead of a live session (0 credits)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--period", type=float, default=2.0,
                    help="episode decision period (sleep after each send)")
    ap.add_argument("--timeout", type=float, default=90.0, help="episode timeout")
    ap.add_argument("--idle-limit", type=float, default=180.0,
                    help="auto-disconnect a READY live session idle this long")
    args = ap.parse_args()

    from dreampilot.reactor_client import setup_logging

    setup_logging()
    room = ControlRoom(replay_dir=Path(args.replay) if args.replay else None,
                       period_s=args.period, default_timeout_s=args.timeout,
                       idle_limit_s=args.idle_limit)
    mode = f"REPLAY from {args.replay}" if args.replay else "LIVE (bills credits!)"
    logger.info("DreamPilot control room: http://%s:%d  [%s]", args.host, args.port, mode)
    web.run_app(build_app(room), host=args.host, port=args.port,
                print=lambda *_: None)


if __name__ == "__main__":
    main()
