"""Phase 0 offline analysis: run directory -> measured.json (PLAN.md Phase 0 gate).

Reads events.jsonl / frames.jsonl / frames/ produced by phase0_record.py and
computes, fully offline:

  - chunk_hz + frames per chunk (from chunk_complete timing and frames_emitted)
  - stream fps (from frame arrival timestamps)
  - action-to-effect latency (command timestamp -> visual-motion onset)
  - rotation unit (full-turn period at a known deg setting -> deg/s)
  - pause/resume/reset observations
  - action enums (from server capabilities when present)

measured.json is committed; all downstream code reads constants from it.

Usage:
    .venv/bin/python tools/phase0_analyze.py data/phase0/<run> [-o measured.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

STREAM_FPS_NOMINAL = 16.0

# Priors being verified (SDK_NOTES.md section 5).
PRIOR_CHUNK_HZ = 1.33
PRIOR_LATENCY_MAX_S = 1.75


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------- frame signal

def load_gray_frames(run_dir: Path, size: tuple[int, int] = (160, 96)) -> tuple[np.ndarray, np.ndarray]:
    """All saved frames as a downscaled grayscale stack + their timestamps."""
    from PIL import Image

    index = load_jsonl(run_dir / "frames.jsonl")
    if not index:
        return np.empty((0,)), np.empty((0,))
    ts, imgs = [], []
    for entry in index:
        p = run_dir / "frames" / entry["file"]
        if not p.exists():
            continue
        img = Image.open(p).convert("L").resize(size, Image.BILINEAR)
        imgs.append(np.asarray(img, dtype=np.float32))
        ts.append(entry["t"])
    return np.stack(imgs), np.array(ts)


def motion_signal(stack: np.ndarray) -> np.ndarray:
    """Mean absolute inter-frame difference; index i = motion at frame i (i>=1)."""
    if len(stack) < 2:
        return np.zeros(len(stack))
    d = np.abs(np.diff(stack, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[0.0], d])


# --------------------------------------------------------------- measurements

def measure_chunks(events: list[dict]) -> dict:
    chunks = [e for e in events if e["kind"] == "message" and e.get("type") == "chunk_complete"]
    if len(chunks) < 3:
        return {"error": f"only {len(chunks)} chunk_complete messages"}
    ts = [c["t"] for c in chunks]
    intervals = np.diff(ts)
    frames_emitted = [c["data"].get("frames_emitted") for c in chunks if c.get("data")]
    frames_per_chunk = None
    fe = [f for f in frames_emitted if isinstance(f, (int, float))]
    if fe:
        # frames_emitted may be cumulative or per-chunk. Cumulative iff strictly
        # growing; a constant series (e.g. always 12) is per-chunk.
        diffs = np.diff(fe)
        cumulative = len(fe) >= 3 and np.all(diffs >= 0) and fe[-1] > fe[0] * 2
        frames_per_chunk = float(np.median(diffs)) if cumulative else float(np.median(fe))
    return {
        "n_chunks": len(chunks),
        "chunk_interval_s_median": round(float(np.median(intervals)), 4),
        "chunk_interval_s_p10_p90": [round(float(np.percentile(intervals, p)), 4) for p in (10, 90)],
        "chunk_hz": round(1.0 / float(np.median(intervals)), 4),
        "frames_per_chunk": frames_per_chunk,
        "frames_emitted_cumulative": cumulative if fe else None,
        "prior_chunk_hz": PRIOR_CHUNK_HZ,
    }


def measure_fps(frame_ts: np.ndarray, chunking: dict) -> dict:
    """Content fps = frames_per_chunk x chunk_hz (robust against duplicated
    deliveries: run_001 showed raw callback rates far above the content rate
    during warmup)."""
    if len(frame_ts) < 10:
        return {"error": "too few frames"}
    intervals = np.diff(frame_ts)
    intervals = intervals[intervals < 1.0]
    out = {
        "n_frames": int(len(frame_ts)),
        "raw_callback_fps_median": round(1.0 / float(np.median(intervals)), 2),
        "nominal_fps_documented": STREAM_FPS_NOMINAL,
    }
    if chunking.get("frames_per_chunk") and chunking.get("chunk_hz"):
        out["content_fps"] = round(chunking["frames_per_chunk"] * chunking["chunk_hz"], 2)
    return out


def measure_latency(events: list[dict], motion: np.ndarray, frame_ts: np.ndarray) -> dict:
    """Per trial: time from the action command to sustained motion onset."""
    trials = [e for e in events if e["kind"] == "latency_trial"]
    if not trials or len(frame_ts) == 0:
        return {"error": "no latency trials or no frames"}
    latencies = []
    details = []
    for tr in trials:
        t_cmd = tr["t"]
        pre = motion[(frame_ts > t_cmd - 2.0) & (frame_ts <= t_cmd)]
        post_mask = (frame_ts > t_cmd) & (frame_ts <= t_cmd + 3.5)
        post_t = frame_ts[post_mask]
        post_m = motion[post_mask]
        if len(pre) < 5 or len(post_m) < 5:
            details.append({**_trial_meta(tr), "onset_s": None, "note": "insufficient frames"})
            continue
        base = np.median(pre)
        spread = np.median(np.abs(pre - base)) + 1e-6
        thresh = base + max(6 * spread, 0.15 * base + 0.2)
        onset = None
        # Require 3 consecutive above-threshold frames to reject flicker.
        above = post_m > thresh
        for i in range(len(above) - 2):
            if above[i] and above[i + 1] and above[i + 2]:
                onset = float(post_t[i] - t_cmd)
                break
        if onset is not None:
            latencies.append(onset)
        details.append({**_trial_meta(tr), "onset_s": round(onset, 3) if onset else None})
    out: dict = {"trials": details, "prior_max_s": PRIOR_LATENCY_MAX_S}
    if latencies:
        out.update({
            "latency_s_median": round(statistics.median(latencies), 3),
            "latency_s_max": round(max(latencies), 3),
            "n_measured": len(latencies),
        })
    else:
        out["error"] = "no onset detected in any trial"
    return out


def _trial_meta(tr: dict) -> dict:
    return {"index": tr.get("index"), "axis": tr.get("axis"), "value": tr.get("value")}


def measure_rotation(events: list[dict], stack: np.ndarray, frame_ts: np.ndarray) -> dict:
    """Horizontal pan rate during constant rotation.

    A full-turn/similarity method does NOT work in a generative world (turning
    360 degrees does not reproduce the starting view - verified on run_001).
    Instead: cross-correlate column-mean profiles of frames 0.25 s apart to get
    pixels/s, reported as screen-widths/s. Absolute deg/s needs the (unknown)
    FOV, so the calibration constant is the pan rate at the tested setting.
    """
    phase = next((e for e in events if e["kind"] == "phase" and e.get("name") == "rotation"), None)
    t_start = next((e["t"] for e in events if e["kind"] == "rotation_start"), None)
    t_end = next((e["t"] for e in events if e["kind"] == "rotation_end"), None)
    if not phase or t_start is None or t_end is None or len(frame_ts) == 0:
        return {"error": "no rotation phase recorded"}
    deg_setting = phase["deg"]

    # Skip 2.5 s (action latency) at the start and 0.5 s at the end.
    mask = (frame_ts > t_start + 2.5) & (frame_ts < t_end - 0.5)
    window = stack[mask]
    wts = frame_ts[mask]
    if len(window) < 60:
        return {"error": f"only {len(window)} frames in rotation window"}

    width = window.shape[2]
    band = window[:, window.shape[1] // 3: 2 * window.shape[1] // 3, :]  # textured mid band
    dt = float(np.median(np.diff(wts)))
    step = max(1, int(round(0.25 / dt)))
    max_shift = width // 3
    rates = []
    for i in range(0, len(band) - step, step):
        a = band[i].mean(axis=0)
        b = band[i + step].mean(axis=0)
        a = a - a.mean()
        b = b - b.mean()
        corr = np.correlate(b, a, mode="full")
        center = len(a) - 1
        shift = int(np.argmax(corr[center - max_shift: center + max_shift + 1])) - max_shift
        rates.append(shift / (step * dt))  # px/s, signed
    rates = np.array(rates)
    moving = rates[np.abs(rates) > 0.05 * width]  # ignore stalled/failed matches
    if len(moving) < 5:
        return {"error": "no consistent pan detected", "deg_setting": deg_setting}
    px_per_s = float(np.median(np.abs(moving)))
    sw_per_s = px_per_s / width
    return {
        "deg_setting": deg_setting,
        "pan_rate_screen_widths_per_s": round(sw_per_s, 4),
        "seconds_per_screen_width": round(1.0 / sw_per_s, 2),
        "deg_per_s_if_fov_90": round(sw_per_s * 90.0, 1),
        "deg_per_s_if_fov_100": round(sw_per_s * 100.0, 1),
        "n_samples": int(len(moving)),
        "note": "unit period of set_rotation_speed_deg is unresolved (needs FOV); "
                "use pan_rate at this setting as the teleop calibration constant "
                "and assume linear scaling in the setting",
    }


def measure_pause(events: list[dict], motion: np.ndarray, frame_ts: np.ndarray) -> dict:
    check = next((e for e in events if e["kind"] == "pause_frame_check"), None)
    reset_ok = next((e for e in events if e["kind"] == "restart_after_reset"), None)
    t_pause = next((e["t"] for e in events if e["kind"] == "message"
                    and e.get("type") == "generation_paused"), None)
    t_resume = next((e["t"] for e in events if e["kind"] == "message"
                     and e.get("type") == "generation_resumed"), None)
    reset_seen = any(e["kind"] == "message" and e.get("type") == "generation_reset" for e in events)

    # Content freeze check: frames may keep flowing during pause (run_001 did),
    # so compare inter-frame motion inside vs outside the pause window.
    frozen = None
    if t_pause and t_resume and len(frame_ts):
        inside = motion[(frame_ts > t_pause) & (frame_ts < t_resume)]
        outside = motion[(frame_ts < t_pause) | (frame_ts > t_resume)]
        if len(inside) > 3 and len(outside) > 10:
            frozen = {
                "motion_during_pause": round(float(np.median(inside)), 2),
                "motion_baseline": round(float(np.median(outside)), 2),
                "content_frozen": bool(np.median(inside) < 0.3 * np.median(outside)),
            }

    # Did the server reject start after reset because staging was cleared?
    reset_t = next((e["t"] for e in events if e["kind"] == "message"
                    and e.get("type") == "generation_reset"), None)
    reset_clears_staging = None
    if reset_t is not None:
        err = next((e for e in events if e["kind"] == "message"
                    and e.get("type") == "command_error" and e["t"] > reset_t), None)
        reset_clears_staging = err["data"].get("reason") if err else False

    return {
        "generation_paused_seen": t_pause is not None,
        "generation_resumed_seen": t_resume is not None,
        "frames_during_4s_pause": check.get("frames_during_pause") if check else None,
        "pause_content": frozen,
        "generation_reset_seen": reset_seen,
        "start_after_reset_without_restaging": reset_ok.get("ok") if reset_ok else None,
        "reset_clears_staging": reset_clears_staging,
    }


def extract_action_enums(events: list[dict]) -> dict:
    """Prefer server-declared capabilities; fall back to the documented schema."""
    caps = next((e for e in events if e["kind"] == "capabilities"), None)
    enums = {
        "movement": ["idle", "forward", "back", "strafe_left", "strafe_right"],
        "look_horizontal": ["idle", "left", "right"],
        "look_vertical": ["idle", "up", "down"],
        "source": "SDK_NOTES.md schema",
    }
    if caps:
        enums["capabilities_raw"] = caps.get("capabilities")
        enums["source"] = "server capabilities logged (see capabilities_raw) + schema"
    return enums


def billing_summary(events: list[dict]) -> dict:
    d = next((e for e in events if e["kind"] == "disconnect"), None)
    return {k: d.get(k) for k in ("billed_seconds", "credits", "dollars")} if d else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("measured.json"))
    args = ap.parse_args()

    events = load_jsonl(args.run_dir / "events.jsonl")
    if not events:
        sys.exit(f"no events.jsonl in {args.run_dir}")
    print(f"loaded {len(events)} events; loading frames...")
    stack, frame_ts = load_gray_frames(args.run_dir)
    print(f"loaded {len(stack)} frames")
    motion = motion_signal(stack) if len(stack) else np.empty((0,))

    chunking = measure_chunks(events)
    measured = {
        "run_dir": str(args.run_dir),
        "chunking": chunking,
        "stream": measure_fps(frame_ts, chunking),
        "action_latency": measure_latency(events, motion, frame_ts),
        "rotation": measure_rotation(events, stack, frame_ts),
        "pause_resume_reset": measure_pause(events, motion, frame_ts),
        "action_enums": extract_action_enums(events),
        "session_billing": billing_summary(events),
    }
    # Convenience top-level constants for downstream code.
    ch = measured["chunking"]
    if "chunk_hz" in ch:
        measured["chunk_hz"] = ch["chunk_hz"]
        measured["frames_per_chunk"] = ch["frames_per_chunk"]
        if "content_fps" in measured["stream"]:
            measured["fps"] = measured["stream"]["content_fps"]
    lat = measured["action_latency"]
    if "latency_s_median" in lat:
        measured["latency_ms"] = round(lat["latency_s_median"] * 1000)
    rot = measured["rotation"]
    if rot.get("pan_rate_screen_widths_per_s"):
        measured["rotation_calibration"] = {
            "deg_setting": rot["deg_setting"],
            "screen_widths_per_s": rot["pan_rate_screen_widths_per_s"],
        }

    args.output.write_text(json.dumps(measured, indent=2) + "\n")
    print(json.dumps(measured, indent=2))
    print(f"\nwrote {args.output}")

    problems = [k for k in ("chunking", "action_latency", "rotation") if measured[k].get("error")]
    if problems:
        print(f"WARNING: sections with errors: {problems} - inspect before committing")


if __name__ == "__main__":
    main()
