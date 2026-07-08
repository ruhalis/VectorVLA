"""DreamPilot live runner (PIVOT.md section 4): stage a world, then drive it.

    .venv/bin/python run_agent.py --image assets/village_seed.jpg \
        --prompt "A quiet cobblestone village square..." \
        [--command "go to the maypole"] [--mode full|fallback]

Control loop (hard rules from PIVOT.md / CLAUDE.md):
  - Sequential at ~0.5 Hz: grab LATEST frame -> VLM -> send changed axes ->
    sleep ~2 s measured FROM THE SEND -> next frame. Never a fixed timer from
    frame-grab: action-to-effect is ~1.5 s (measured.json) and the next
    observation must postdate the previous action's effect.
  - Arrival: zero movement on the FIRST arrived, success banner only on the
    SECOND consecutive one. 90 s timeout -> zero actions.
  - After each episode the runner zeroes actions and awaits the next command
    on stdin — multiple commands per session (the judge demo needs this).
  - Server-side recording is downloaded before disconnect (demo insurance).
  - Everything goes through reactor_client.ReactorSession (credit meter,
    READY-gated sends, ring buffer, reconnect, expiry watch). Disconnect
    (non-recoverable) at the end — never leave a session idle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from reactor_client import ReactorSession, setup_logging
from vlm_policy import VLMPolicy, load_env

logger = logging.getLogger("vectorvla.agent")

MEASURED = json.loads(Path(__file__).with_name("measured.json").read_text())
ACTION_TO_EFFECT_S = MEASURED["latency_ms"] / 1000  # ~1.5 s — why the sleep is ~2 s

DEFAULT_PROMPT = (
    "A quiet European village square on a sunny summer day. A tall maypole "
    "decorated with a green wreath stands in the center of the cobblestone "
    "square, surrounded by red brick houses with tiled roofs and small green "
    "trees. Bright blue sky with scattered white clouds."
)


async def run_episode(session: ReactorSession, command: str, mode: str,
                      timeout_s: float, period_s: float, expiry_margin_s: float) -> bool:
    policy = VLMPolicy(command, mode=mode)
    print(f"\n=== EPISODE [{mode}] command: {command!r} (timeout {timeout_s:.0f}s) ===")
    t_start = time.monotonic()
    arrived_streak = 0

    while (elapsed := time.monotonic() - t_start) < timeout_s:
        remaining = session.seconds_remaining()
        if remaining is not None and remaining < expiry_margin_s:
            logger.warning("session expires in %.0fs — ending episode", remaining)
            break

        latest = session.latest_frame()
        if latest is None:
            await asyncio.sleep(0.5)
            continue
        _, _, frame = latest

        # Blocking VLM call off the event loop; frame callback never waits on it.
        decision = await asyncio.to_thread(policy.decide, frame)
        print(f"[{elapsed:5.1f}s] {decision.line()}")

        if decision.arrived:
            arrived_streak += 1
            if arrived_streak >= 2:
                await session.zero_actions()
                print(f"\n*** ARRIVED: {command!r} in {elapsed:.0f}s "
                      f"({policy.calls} decisions, {policy.failures} held) ***\n")
                return True
            # First arrived: stop translating, keep looking; confirm next tick.
            await session.set_action(movement="idle",
                                     look_horizontal=decision.look_horizontal,
                                     look_vertical=decision.look_vertical)
        else:
            arrived_streak = 0
            if decision.ok:  # on failure hold previous state — send nothing
                await session.set_action(**decision.action)

        # Cadence measured from the send (hard rule) — set_action just returned.
        await asyncio.sleep(period_s)

    await session.zero_actions()
    print(f"\n--- episode ended without arrival after {time.monotonic() - t_start:.0f}s "
          f"({policy.calls} decisions, {policy.failures} held) ---\n")
    return False


async def download_recording(session: ReactorSession, run_dir: Path) -> None:
    """Demo insurance: fetch the server-side recording (free, expires in 24 h)."""
    try:
        from reactor_sdk.utils.tokens import fetch_jwt_token

        clip = await session.reactor.request_recording()
        # The Coordinator manifest endpoint needs a Bearer JWT (else HTTP 401).
        jwt = await fetch_jwt_token(api_key=os.environ["REACTOR_API_KEY"])
        path = run_dir / f"recording_{datetime.now():%H%M%S}.mp4"
        await session.reactor.download_clip_as_file(clip, str(path), jwt=jwt)
        print(f"[recording saved: {path}]")
    except Exception as e:  # noqa: BLE001 — insurance must never kill the run
        logger.warning("recording download failed: %s", e)


async def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default=None,
                    help="named world from worlds.json (overrides --image/--prompt)")
    ap.add_argument("--image", default="assets/village_seed.jpg")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="static scene description — NO motion verbs")
    ap.add_argument("--command", default=None, help="first command; else read from stdin")
    ap.add_argument("--mode", choices=("full", "fallback"), default="full")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rotation-speed", type=float, default=None,
                    help="set_rotation_speed_deg 0-30 (default server 5)")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--period", type=float, default=2.0,
                    help="sleep after each send; ~0.5 Hz decisions")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    if args.world:
        world = json.loads(Path(__file__).with_name("worlds.json").read_text())[args.world]
        args.image, args.prompt = world["image"], world["prompt"]

    setup_logging()
    run_dir = Path(args.run_dir or f"data/live/run_{datetime.now():%Y%m%d_%H%M%S}")
    session = ReactorSession(api_key=os.environ["REACTOR_API_KEY"], run_dir=run_dir)

    await session.connect()
    try:
        await session.stage_world(args.image, args.prompt,
                                  seed=args.seed, rotation_speed_deg=args.rotation_speed)
        print("[world staged; waiting for frames...]")
        while session.latest_frame() is None:
            await asyncio.sleep(0.2)
        # Let the model settle into the seed image before the first decision.
        await asyncio.sleep(2 * ACTION_TO_EFFECT_S)

        command = args.command
        while True:
            if not command:
                try:
                    command = (await asyncio.to_thread(
                        input, "command> (empty or 'q' to disconnect) ")).strip()
                except EOFError:  # non-interactive run: one --command, then out
                    break
                if command in ("", "q", "quit"):
                    break
            success = await run_episode(session, command, args.mode,
                                        args.timeout, args.period, expiry_margin_s=60)
            if success:
                await download_recording(session, run_dir)  # clip covers the win
            command = None  # next command from stdin
            remaining = session.seconds_remaining()
            if remaining is not None and remaining < 90:
                print(f"[session expires in {remaining:.0f}s — disconnecting]")
                break
    finally:
        await download_recording(session, run_dir)
        await session.disconnect()
        print(f"[session done: {session.meter.summary()}]")


if __name__ == "__main__":
    asyncio.run(main())
