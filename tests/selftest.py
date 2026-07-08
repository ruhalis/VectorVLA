"""Offline self-test for analyze.py - costs zero credits.

Fabricates a synthetic run directory that mimics what record.py writes
(events.jsonl, frames.jsonl, frames/) with known ground truth:

  - chunk_complete every 0.75 s, frames_emitted = 12  -> chunk_hz 1.333
  - action-to-effect latency = 1.2 s on every trial
  - rotation pan rate = 12 screen-widths per 18 s = 0.667 sw/s
  - a 4 s frame gap during pause

Then runs the analyzer and asserts the truth is recovered. Run this before
spending credits on the live characterization.

Usage: .venv/bin/python tests/selftest.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "data" / "measure" / "_selftest"
FPS = 16.0
DT = 1.0 / FPS
W, H = 160, 96

TRUE_LATENCY_S = 1.2
TRUE_TURN_PERIOD_S = 18.0  # 15 deg setting, deg-per-chunk => 20 deg/s => 18 s/turn
ROTATION_DEG_SETTING = 15.0


def make_panorama(rng: np.random.Generator) -> np.ndarray:
    """Smooth random 360-degree panorama, tiled horizontally for wraparound."""
    pano_w = W * 12
    noise = rng.random((H // 8, pano_w // 8))
    img = np.kron(noise, np.ones((8, 8)))[:H, :pano_w]
    # Heavy horizontal smoothing so panning produces coherent structure.
    kernel = np.ones(15) / 15
    img = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, img)
    return (img * 255).astype(np.float32)


def main() -> None:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    (RUN_DIR / "frames").mkdir(parents=True)
    rng = np.random.default_rng(0)
    pano = make_panorama(rng)
    pano_w = pano.shape[1]
    px_per_s_turn = pano_w / TRUE_TURN_PERIOD_S  # full pano width = one turn

    events, frame_index = [], []
    t = 100.0  # arbitrary monotonic origin
    frame_i = 0
    cam_x = 0.0  # panorama offset in px
    moving_until = -1.0
    move_px_per_s = 0.0

    def ev(kind: str, **kw):
        events.append({"t": round(t, 4), "wall": time.time(), "kind": kind, **kw})

    def emit_frame():
        nonlocal frame_i
        x = int(cam_x) % pano_w
        view = np.take(pano, range(x, x + W), axis=1, mode="wrap")
        # mild sensor noise so "static" frames aren't identical
        noisy = np.clip(view + rng.normal(0, 1.0, view.shape), 0, 255).astype(np.uint8)
        name = f"{frame_i:06d}.jpg"
        Image.fromarray(noisy).save(RUN_DIR / "frames" / name, quality=90)
        frame_index.append({"i": frame_i, "t": round(t, 4), "file": name})
        frame_i += 1

    def advance(dur: float, paused: bool = False):
        """Advance sim time, emitting frames and chunk messages."""
        nonlocal t, cam_x, next_chunk_t
        end = t + dur
        while t < end - 1e-9:
            t += DT
            if not paused:
                if t <= moving_until:
                    cam_x += move_px_per_s * DT
                emit_frame()
                while t >= next_chunk_t:
                    ev("message", type="chunk_complete",
                       data={"chunk_index": len([e for e in events if e["kind"] == "message"
                                                 and e.get("type") == "chunk_complete"]),
                             "frames_emitted": 12, "active_prompt": "p", "active_action": "still"})
                    next_chunk_t += 0.75

    next_chunk_t = t + 0.75
    ev("connect_start"); ev("ready", session_id="fake")
    ev("staged", prompt="p", image="x.jpg", seed=42)

    # passive 30 s
    ev("phase", name="passive", duration=30.0)
    advance(30.0)

    # latency trials: motion starts TRUE_LATENCY_S after the command
    trials = [("look_horizontal", "right", px_per_s_turn), ("look_horizontal", "left", -px_per_s_turn),
              ("movement", "forward", px_per_s_turn * 0.6), ("movement", "back", -px_per_s_turn * 0.6),
              ("look_horizontal", "right", px_per_s_turn), ("look_horizontal", "left", -px_per_s_turn)]
    ev("phase", name="latency", trials=len(trials))
    for i, (axis, value, vel) in enumerate(trials):
        ev("latency_trial", index=i, axis=axis, value=value)
        ev("command", command=f"set_{axis}", data={axis: value})
        # motion runs from cmd+latency until idle lands (3 s hold + latency)
        start_moving = t + TRUE_LATENCY_S
        # emulate by scheduling: advance latency with no motion, then move
        advance(TRUE_LATENCY_S)
        move_px_per_s = vel
        moving_until = t + 3.0
        advance(3.0 - TRUE_LATENCY_S + 3.0)  # rest of hold + settle
        move_px_per_s = 0.0

    # rotation: constant pan, full turn every TRUE_TURN_PERIOD_S
    ev("phase", name="rotation", deg=ROTATION_DEG_SETTING, duration=50.0)
    advance(1.5)
    ev("rotation_start")
    move_px_per_s = px_per_s_turn
    moving_until = t + 50.0
    advance(50.0)
    move_px_per_s = 0.0
    ev("rotation_end")
    advance(2.0)

    # pause: 4 s with no frames
    ev("phase", name="pause_resume_reset")
    ev("message", type="generation_paused", data={"chunk_index": 1})
    advance(4.0, paused=True)
    ev("pause_frame_check", frames_during_pause=0)
    ev("message", type="generation_resumed", data={"chunk_index": 1})
    advance(3.0)
    ev("message", type="generation_reset", data={"reason": "client"})
    ev("restart_after_reset", ok=True)
    advance(3.0)
    ev("disconnect", billed_seconds=180.0, credits=5940, dollars=0.594, frames_received=frame_i)

    with open(RUN_DIR / "events.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in events)
    with open(RUN_DIR / "frames.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in frame_index)
    print(f"synthetic run: {frame_i} frames, {len(events)} events")

    out = RUN_DIR / "measured_selftest.json"
    analyzer = Path(__file__).with_name("analyze.py")
    subprocess.run([sys.executable, str(analyzer), str(RUN_DIR), "-o", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    m = json.loads(out.read_text())

    failures = []

    def check(name: str, cond: bool, got):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}: {got}")
        if not cond:
            failures.append(name)

    print("\nassertions:")
    check("chunk_hz ~= 1.333", abs(m["chunking"]["chunk_hz"] - 4 / 3) < 0.05, m["chunking"]["chunk_hz"])
    check("frames_per_chunk == 12", m["chunking"]["frames_per_chunk"] == 12, m["chunking"]["frames_per_chunk"])
    check("content fps ~= 16", abs(m["stream"]["content_fps"] - 16) < 0.5, m["stream"]["content_fps"])
    lat = m["action_latency"].get("latency_s_median")
    check("latency ~= 1.2 s", lat is not None and abs(lat - TRUE_LATENCY_S) < 0.35, lat)
    check("all 6 trials measured", m["action_latency"].get("n_measured") == 6,
          m["action_latency"].get("n_measured"))
    # pano is 12 screen widths wide, one turn per TRUE_TURN_PERIOD_S
    true_sw_per_s = 12.0 / TRUE_TURN_PERIOD_S
    sw = m["rotation"].get("pan_rate_screen_widths_per_s")
    check(f"pan rate ~= {true_sw_per_s:.3f} sw/s", sw is not None and abs(sw - true_sw_per_s) < 0.07, sw)
    check("pause frames == 0", m["pause_resume_reset"]["frames_during_4s_pause"] == 0,
          m["pause_resume_reset"]["frames_during_4s_pause"])

    if failures:
        sys.exit(f"\nSELFTEST FAILED: {failures}")
    print("\nselftest passed - the analyzer recovers known ground truth")


if __name__ == "__main__":
    main()
