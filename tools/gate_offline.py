"""Offline go/no-go gate for the VLM policy (PIVOT.md T+0 block). Zero credits.

Runs the exact live policy path (same Policy class, same frame_to_data_url) over
~20 recorded run_001 frames with a landmark command and checks:
  1. valid JSON every time (any held/failed decision fails the gate),
  2. decision latency: median <= 3 s (the PIVOT risk threshold),
  3. directional sanity: printed per-frame actions + reasoning for eyeballing,
     plus a crude auto-check that the policy is not frozen on one action.

    .venv/bin/python tools/gate_offline.py [--command "..."] [--n 20] [--mode full|fallback]

Frames are fed sequentially through ONE policy instance so the text-memory
(history) path is exercised exactly as it will be live.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root — tools run as plain scripts

from dreampilot.config import ROOT, load_env
from dreampilot.policy import MODES, make_policy

FRAMES_DIR = ROOT / "data" / "phase0" / "run_001" / "frames"
DEFAULT_COMMAND = "go to the maypole in the middle of the square"


def pick_frames(n: int) -> list[Path]:
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {FRAMES_DIR} — run_001 missing?")
    # Skip the warmup (duplicated deliveries) at the start, spread over the rest.
    usable = frames[200:] if len(frames) > 400 else frames
    step = max(1, len(usable) // n)
    return usable[::step][:n]


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--command", default=DEFAULT_COMMAND)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mode", choices=sorted(MODES), default="full")
    args = ap.parse_args()

    frames = pick_frames(args.n)
    policy = make_policy(args.command, mode=args.mode)
    print(f"gate: {len(frames)} frames | mode={args.mode} | model={policy.model}")
    print(f"command: {args.command!r}\n")

    decisions = []
    for path in frames:
        d = policy.decide(path)
        decisions.append(d)
        print(f"{path.name}  {d.line()}")

    ok = [d for d in decisions if d.ok]
    latencies = sorted(d.latency_s for d in ok)
    actions = {(d.movement, d.look_horizontal) for d in ok}
    n_arrived = sum(d.arrived for d in ok)

    print(f"\n--- gate summary ---")
    print(f"valid JSON: {len(ok)}/{len(decisions)}")
    if latencies:
        print(f"latency: median {statistics.median(latencies):.2f}s, "
              f"max {latencies[-1]:.2f}s")
    print(f"distinct (movement, look_h) pairs: {len(actions)}; arrived flags: {n_arrived}")

    checks = {
        "valid JSON every time": len(ok) == len(decisions) and decisions,
        "median latency <= 3s": bool(latencies) and statistics.median(latencies) <= 3.0,
        "policy not frozen (>=2 distinct actions)": len(actions) >= 2,
    }
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print("\nEyeball the reasoning lines above for directional sanity "
          "(landmark left of center -> left-ish actions).")
    if all(checks.values()):
        print("GATE: PASS — go live (T+1:00 block).")
    else:
        print("GATE: FAIL — tune the prompt or switch --mode fallback before burning credits.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
