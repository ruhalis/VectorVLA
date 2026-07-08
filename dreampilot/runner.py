"""DreamPilot live runner (PIVOT.md section 4): stage a world, then drive it.

    .venv/bin/python -m dreampilot --world village   # or run_agent.py (shim)
    .venv/bin/python -m dreampilot --image assets/village_seed.jpg \
        --prompt "A quiet cobblestone village square..." \
        [--command "go to the maypole"] [--mode full|fallback]

Control loop (hard rules from PIVOT.md / CLAUDE.md) — PULSE mode:
  - Sequential pulses: grab LATEST settled frame -> VLM picks ONE action ->
    send it -> let it run --period s -> send idle (actions are persistent;
    the explicit stop ends the pulse) -> wait --settle s (default 3 s: measured
    action-to-effect is ~1.5 s plus server round-trip) -> next frame. Every
    observation postdates the previous pulse completely, so the policy sees
    what its last action did instead of acting on stale motion.
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
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from dreampilot.config import load_env, load_measured, load_worlds
from dreampilot.policy import Decision, MODES, make_policy
from dreampilot.reactor_client import ReactorSession, setup_logging

logger = logging.getLogger("vectorvla.agent")

ACTION_TO_EFFECT_S = load_measured().action_to_effect_s  # ~1.5 s measured
# Wait after each pulse's stop before feeding the next frame. Deliberately
# generous vs. the measured 1.5 s: server round-trip pushes the real stop
# later, and deciding on a frame that is still moving re-creates the
# act-while-already-moving oscillation.
DEFAULT_SETTLE_S = 3.0

DEFAULT_PROMPT = (
    "A quiet European village square on a sunny summer day. A tall maypole "
    "decorated with a green wreath stands in the center of the cobblestone "
    "square, surrounded by red brick houses with tiled roofs and small green "
    "trees. Bright blue sky with scattered white clouds."
)


async def run_episode(session: ReactorSession, command: str, mode: str,
                      timeout_s: float, period_s: float, expiry_margin_s: float,
                      on_decision: Optional[Callable[[float, Decision], None]] = None,
                      settle_s: Optional[float] = None) -> bool:
    if settle_s is None:
        settle_s = DEFAULT_SETTLE_S  # stop must be visible before observing
    policy = make_policy(command, mode=mode)
    print(f"\n=== EPISODE [{mode}] command: {command!r} (timeout {timeout_s:.0f}s, "
          f"pulse {period_s:.1f}s + settle {settle_s:.1f}s) ===")
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
        if on_decision is not None:
            on_decision(elapsed, decision)  # web UI hook; sync and non-blocking

        if decision.arrived:
            arrived_streak += 1
            if arrived_streak >= 2:
                await session.zero_actions()
                print(f"\n*** ARRIVED: {command!r} in {elapsed:.0f}s "
                      f"({policy.calls} decisions, {policy.failures} held) ***\n")
                return True
            # First arrived: stay stopped, confirm from the next settled frame.
            await session.zero_actions()
            await asyncio.sleep(settle_s)
            continue
        arrived_streak = 0

        if decision.ok:
            # Pulse: apply the single action, let it play out, then stop it —
            # actions are persistent, so the explicit idle is what ends the pulse.
            await session.set_action(**decision.action)
            await asyncio.sleep(period_s)
            await session.zero_actions()
        # On failure nothing was sent — the world is already stopped; either
        # way, wait for a settled view before the next decision.
        await asyncio.sleep(settle_s)

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
    ap.add_argument("--mode", choices=sorted(MODES), default="full")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rotation-speed", type=float, default=None,
                    help="set_rotation_speed_deg 0-30 (default server 5)")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--period", type=float, default=2.0,
                    help="pulse length: how long each single action runs before the stop")
    ap.add_argument("--settle", type=float, default=None,
                    help="wait after the stop before the next frame "
                         f"(default {DEFAULT_SETTLE_S:.1f}s — covers server round-trip)")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    if args.world:
        world = load_worlds()[args.world]
        args.image, args.prompt = str(world.image), world.prompt

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
                                        args.timeout, args.period, expiry_margin_s=60,
                                        settle_s=args.settle)
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
