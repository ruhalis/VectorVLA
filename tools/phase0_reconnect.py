"""Minimal live test of the manual reconnect path (PLAN.md Phase 0).

Connects, stages a world, forces a recoverable transport drop, calls
reconnect(), and verifies frames flow again. ~60 s READY time (~$0.25,
including the billed recovery gap). Kept separate from phase0_record.py so
the main characterization run doesn't pay for the 30 s recovery window.

Usage: .venv/bin/python tools/phase0_reconnect.py --image assets/village_seed.jpg
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

from reactor_sdk import ReactorStatus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root — tools run as plain scripts

from dreampilot.reactor_client import ReactorSession, setup_logging

logger = logging.getLogger("vectorvla.phase0.reconnect")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=(
        "A quiet German village square on a sunny day. Red brick houses with steep "
        "red tiled roofs surround a paved plaza. A tall maypole with a green wreath "
        "stands in the center. Small trees line the square under a blue sky with "
        "scattered white clouds."
    ))
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    api_key = os.environ.get("REACTOR_API_KEY")
    if not api_key:
        sys.exit("REACTOR_API_KEY is not set")

    run_dir = Path(args.run_dir or f"data/phase0/reconnect_{datetime.now():%Y%m%d_%H%M%S}")
    setup_logging()

    # auto_reconnect off: this test drives reconnect() by hand to time it.
    session = ReactorSession(api_key=api_key, run_dir=run_dir, auto_reconnect=False)
    result = {}
    try:
        await session.connect()
        await session.stage_world(args.image, args.prompt)
        await asyncio.sleep(4.0)
        frames_before = session.frame_count
        result["frames_before_drop"] = frames_before
        logger.info("forcing recoverable disconnect (session preserved server-side 30s)...")
        session.events.write("force_drop")
        await session.reactor.disconnect(recoverable=True)
        session.meter.on_status(ReactorStatus.DISCONNECTED)
        await asyncio.sleep(3.0)

        t0 = time.monotonic()
        await session.reactor.reconnect()
        await session.wait_for_status(ReactorStatus.READY, timeout=60)
        took = time.monotonic() - t0
        result["reconnect_s"] = round(took, 2)
        session.events.write("manual_reconnect", took_s=result["reconnect_s"])

        f0 = session.frame_count
        await asyncio.sleep(5.0)
        result["frames_after_reconnect_5s"] = session.frame_count - f0
        result["frames_flowing"] = session.frame_count - f0 > 10
        # Was generation still running, or does it need a restart?
        state = session.reactor.get_state()
        result["state_after_reconnect"] = str(state)
        session.events.write("reconnect_result", **{k: str(v) for k, v in result.items()})
    finally:
        await session.disconnect()

    print("\nreconnect test result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"\nrun dir: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
