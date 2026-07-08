"""Phase 0 live characterization run (PLAN.md Phase 0).

One short session (~3-4 min READY time, ~$0.7) that records everything needed
to verify the control-loop priors offline:

  1. Stage a world, record ~30 s of passive generation (frames + chunk_completes).
  2. Action-to-effect latency trials: timestamped look/movement pulses.
  3. Rotation-speed unit measurement: constant rotation at a known deg setting;
     the full-turn period is recovered offline (unit-free).
  4. pause / resume / reset semantics; optional manual-reconnect test.

Everything lands in a run directory (events.jsonl, frames.jsonl, frames/) that
phase0_analyze.py turns into measured.json. No analysis happens live.

Usage:
    .venv/bin/python tools/phase0_record.py --image seed.jpg \
        --prompt "A cobblestone village square with a stone fountain..." \
        [--run-dir data/phase0/run_001] [--test-reconnect] [--skip-rotation]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root — tools run as plain scripts

from dreampilot.reactor_client import ReactorSession, setup_logging

logger = logging.getLogger("vectorvla.phase0")

READY_BUDGET_S = 8 * 60  # hard abort: never let this run bill more than 8 min

ROTATION_TEST_DEG = 15.0  # if unit is per-second: 24 s/turn; per-chunk: 18 s/turn
ROTATION_TEST_DURATION_S = 50.0  # covers ~2 full turns under either hypothesis
DEFAULT_ROTATION_DEG = 5.0  # SDK default, restored after the test


async def guard_budget(session: ReactorSession) -> None:
    while True:
        await asyncio.sleep(5)
        if session.meter.billed_seconds > READY_BUDGET_S:
            logger.error("READY budget exceeded (%.0fs) - forcing disconnect", READY_BUDGET_S)
            await session.disconnect()
            os._exit(2)


async def passive_recording(session: ReactorSession, duration: float = 30.0) -> None:
    session.events.write("phase", name="passive", duration=duration)
    logger.info("passive recording for %.0fs...", duration)
    t0, f0 = time.monotonic(), session.frame_count
    await asyncio.sleep(duration)
    got = session.frame_count - f0
    logger.info("passive done: %d frames in %.1fs (%.1f fps), %d chunks total",
                got, time.monotonic() - t0, got / duration, len(session.chunks))


async def latency_trials(session: ReactorSession) -> None:
    """Timestamped action pulses; the analyzer finds visual onset via frame diff.

    Directions alternate so the camera roughly returns to where it started.
    """
    trials = [
        ("look_horizontal", "right"),
        ("look_horizontal", "left"),
        ("movement", "forward"),
        ("movement", "back"),
        ("look_horizontal", "right"),
        ("look_horizontal", "left"),
    ]
    session.events.write("phase", name="latency", trials=len(trials))
    for i, (axis, value) in enumerate(trials):
        session.events.write("latency_trial", index=i, axis=axis, value=value)
        await session.set_action(**{axis: value})
        await asyncio.sleep(3.0)  # hold long enough for the effect to be visible
        await session.set_action(**{axis: "idle"})
        await asyncio.sleep(3.0)  # settle so trials don't bleed into each other
        logger.info("latency trial %d/%d (%s=%s) done", i + 1, len(trials), axis, value)


async def rotation_measurement(session: ReactorSession) -> None:
    """Constant rotation at a known setting; period -> deg/s recovered offline.

    At 15 deg: per-second unit means 360/15 = 24 s per full turn, per-chunk
    (0.75 s) means 360/20 = 18 s. 50 s of rotation covers both cleanly.
    """
    session.events.write("phase", name="rotation", deg=ROTATION_TEST_DEG,
                         duration=ROTATION_TEST_DURATION_S)
    await session.send("set_rotation_speed_deg", {"rotation_speed_deg": ROTATION_TEST_DEG})
    await asyncio.sleep(1.5)  # let the setting land at a chunk boundary
    session.events.write("rotation_start")
    await session.set_action(look_horizontal="right")
    await asyncio.sleep(ROTATION_TEST_DURATION_S)
    await session.set_action(look_horizontal="idle")
    session.events.write("rotation_end")
    await session.send("set_rotation_speed_deg", {"rotation_speed_deg": DEFAULT_ROTATION_DEG})
    await asyncio.sleep(2.0)
    logger.info("rotation measurement done")


async def pause_resume_reset(session: ReactorSession) -> None:
    session.events.write("phase", name="pause_resume_reset")

    await session.send("pause", {})
    try:
        await session.wait_for_message("generation_paused", timeout=10)
    except asyncio.TimeoutError:
        logger.warning("no generation_paused within 10s")
    f0 = session.frame_count
    await asyncio.sleep(4.0)
    session.events.write("pause_frame_check", frames_during_pause=session.frame_count - f0)
    logger.info("frames during 4s pause: %d (expect ~0)", session.frame_count - f0)

    await session.send("resume", {})
    try:
        await session.wait_for_message("generation_resumed", timeout=10)
    except asyncio.TimeoutError:
        logger.warning("no generation_resumed within 10s")
    await asyncio.sleep(3.0)

    await session.send("reset", {})
    try:
        msg = await session.wait_for_message("generation_reset", timeout=10)
        session.events.write("reset_observed", data=msg.get("data", {}))
    except asyncio.TimeoutError:
        logger.warning("no generation_reset within 10s")
    # Server may clear persistent action state on reset; drop the local cache
    # so the next set_action re-sends everything.
    session.action_state = {k: "?" for k in session.action_state}
    await asyncio.sleep(2.0)

    # Does start work again without restaging (has_prompt/has_image retained)?
    await session.send("start", {})
    try:
        await session.wait_for_message("generation_started", timeout=30)
        session.events.write("restart_after_reset", ok=True)
        logger.info("start after reset works without restaging")
    except asyncio.TimeoutError:
        session.events.write("restart_after_reset", ok=False)
        logger.warning("start after reset did not produce generation_started in 30s")
    await asyncio.sleep(3.0)


async def reconnect_test(session: ReactorSession) -> None:
    """Force a transport drop and verify the manual reconnect() path.

    Bills through the 30 s recovery window - only run with --test-reconnect.
    """
    session.events.write("phase", name="reconnect_test")
    logger.info("forcing recoverable disconnect...")
    await session.reactor.disconnect(recoverable=True)
    await asyncio.sleep(2.0)
    t0 = time.monotonic()
    await session.reactor.reconnect()
    session.events.write("manual_reconnect", took_s=round(time.monotonic() - t0, 2))
    from reactor_sdk import ReactorStatus
    await session.wait_for_status(ReactorStatus.READY, timeout=60)
    f0 = session.frame_count
    await asyncio.sleep(4.0)
    ok = session.frame_count > f0
    session.events.write("reconnect_frames_flowing", ok=ok)
    logger.info("reconnect done; frames flowing again: %s", ok)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="seed image (photoreal scene)")
    ap.add_argument("--prompt", default=(
        "A quiet German village square on a sunny day. Red brick houses with steep "
        "red tiled roofs surround a paved plaza. A tall maypole with a green wreath "
        "stands in the center. Small trees line the square under a blue sky with "
        "scattered white clouds."
    ), help="static scene description - no motion verbs")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-rotation", action="store_true")
    ap.add_argument("--test-reconnect", action="store_true",
                    help="also test the manual reconnect path (bills the 30s window)")
    args = ap.parse_args()

    api_key = os.environ.get("REACTOR_API_KEY")
    if not api_key:
        sys.exit("REACTOR_API_KEY is not set")
    if not Path(args.image).exists():
        sys.exit(f"seed image not found: {args.image}")

    run_dir = Path(args.run_dir or f"data/phase0/{datetime.now():%Y%m%d_%H%M%S}")
    setup_logging()
    logger.info("run dir: %s", run_dir)

    session = ReactorSession(api_key=api_key, run_dir=run_dir, save_frames=True)
    guard = None
    try:
        await session.connect()
        guard = asyncio.get_running_loop().create_task(guard_budget(session))

        await session.stage_world(args.image, args.prompt, seed=args.seed,
                                  rotation_speed_deg=DEFAULT_ROTATION_DEG)
        await passive_recording(session, 30.0)
        await latency_trials(session)
        if not args.skip_rotation:
            await rotation_measurement(session)
        await pause_resume_reset(session)
        if args.test_reconnect:
            await reconnect_test(session)
    finally:
        if guard:
            guard.cancel()
        await session.disconnect()

    print(f"\nrun complete: {run_dir}")
    print(f"next: .venv/bin/python tools/phase0_analyze.py {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
