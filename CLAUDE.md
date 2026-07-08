# CLAUDE.md

Hackathon project — pivoted 2026-07-08 with ~3 h remaining: **DreamPilot** (repo name stays
VectorVLA) — a frontier multimodal model (cloud VLM) that perceives, reasons, and navigates
*inside* LingBot-World, a photoreal world dreamed in real time by a neural world model served
by Reactor, from plain-English commands, using LingBot's native action API as its body.

**`PIVOT.md` is the source of truth — architecture, 3-hour schedule, gates, risks, budget.
Read it before starting work; tick its checkboxes as items land.** The archived vector-BC
plan (formerly `PLAN.md`) has been removed; its measured findings live on in
`measured.json`.

Related repo: `~/projects/mnemos` — the bachelor-thesis SO-101 manipulation track, where the
original frozen-encoder BC recipe lives on. Paused during the hackathon; don't mix code.

## Reactor / LingBot API facts (source-verified, 2026-07 — full detail in SDK_NOTES.md)

- Python SDK: `reactor-sdk==0.8.0` (import `reactor_sdk`), async, WebRTC transport;
  `Reactor(model_name="lingbot", api_key=os.environ["REACTOR_API_KEY"])` — the SDK reads no
  env vars itself, pass the key explicitly. `await connect()`; wait for READY status
  (`@reactor.on_status(ReactorStatus.READY)`). Billing starts at READY (WAITING is free).
- Stage a world: `upload_file(img)` → `send_command("set_image", {"image": ref})` → wait for
  the `image_accepted` message → `send_command("set_prompt", {"prompt": ...})` →
  `send_command("start", {})`. `start` before both are set returns `command_error`.
- Frames arrive via `@reactor.on_frame` as `(H, W, 3)` uint8 RGB numpy arrays;
  stream is 1664×960, <1 s latency. **Docs say 16 fps but we measured ~40 fps content
  rate** (raw callback rate is higher still — duplicated deliveries during warmup; dedupe by
  content, not arrival). The callback runs synchronously on the SDK's event loop — copy into
  a drop-oldest ring buffer and return; inference happens elsewhere.
- Actions are **three independent axes**, each persistent state applied at chunk boundaries:
  `set_movement` (`idle|forward|back|strafe_left|strafe_right` — **strafe-only, no turning**),
  `set_look_horizontal` (`idle|left|right` — this is how you turn), `set_look_vertical`
  (`idle|up|down`). Send state changes only; to stop you must send `"idle"` explicitly.
  The VLM policy picks ONE single action per pulse (e.g. `turn_left`, `forward`), which
  maps to one active axis with the others idle.
- **Measured (run_001, committed in `measured.json` — all code reads it from
  there):** chunk = 24 video frames ≈ 0.61 s → chunk rate ≈ 1.65 Hz; action-to-effect
  ≈ 1.5 s (1.12–1.57 s). The OSS priors (12 frames / 0.75 s / 1.33 Hz / 16 fps) are stale
  for the current deployment.
- `send_command` when status ≠ READY silently no-ops — gate every send on READY.
  (`chunk_complete` messages carry `active_action` + `frames_emitted` — useful for debugging
  what the model actually applied.)
- Reconnect is **manual**: on a transport drop the SDK preserves the session but does NOT
  auto-reconnect; wire `on_error` → if `err.recoverable`: `await reconnect()`. The 30 s
  recovery window is billed.
- Other commands: `pause` / `resume` / `reset` / `set_seed` / `set_rotation_speed_deg`
  (0–30, default 5; unit unresolved, but calibrated: at setting 15 the view pans ~0.56
  screen-widths/s). **`reset` clears the prompt** — re-send `set_prompt` before `start`.
  **`pause` freezes content but frames keep flowing** (and billing continues). Sessions
  hard-cap at **20 minutes** (watch `session_expiration_changed`).
- **Free server-side recording** (methods on the raw SDK object, not yet wrapped in
  `ReactorSession`): `await reactor.request_clip(N)` (last N seconds) or
  `await reactor.request_recording()` (from session start, server-capped ~5 min) →
  `await reactor.download_clip_as_file(clip, path)`; failures raise `RecordingError`.
  Clips are deleted after 24 h — download same-day. Start a recording on every live run;
  demo insurance is a by-product of testing, not a separate task.

## Hard rules

- **Live sessions cost real money ($11.88/hr = 33 credits/s, billed per wall-clock second
  from READY).** Offline-first: develop and gate the prompt against recorded run_001 frames;
  connect live only for closed-loop runs and demos. Every session must go through
  `dreampilot/reactor_client.py` so the credit meter logs the burn (computed client-side —
  no billing events exist on the message channel). **`pause` does NOT stop the meter — only
  disconnecting does.** Disconnect (non-recoverable) between trials; never leave a session
  running idle. (Layout: the `dreampilot/` package is the live stack — module map in its
  `__init__.py`; `tests/` holds the measurement scripts and the offline gate. Live
  entry point: `python -m dreampilot` or the root `run_agent.py` shim.)
- **Sequential PULSE control loop:** grab latest settled frame → VLM picks ONE action
  (single axis; e.g. `turn_left`, `forward`, `stop`) → send it → sleep `--period` ≈2 s
  (the pulse) → send idle (actions are persistent — the explicit stop ends the pulse) →
  sleep `--settle` 3 s (action-to-effect is 1.5 s + server round-trip) → next frame. Every observation
  postdates the previous pulse completely, or the policy acts on stale motion and
  oscillates. Never move + turn in the same pulse.
- **VLM output is enforced JSON** (tool-calling / structured outputs, not parse-and-pray),
  validated against the action vocabulary in the policy (mapped to the axis enums in
  `dreampilot/actions.py`). On any failure, send nothing — the world is already stopped
  between pulses, so "do nothing" is safe — and retry next tick. Never block the frame
  callback.
- **Arrival:** zero movement on the *first* `arrived`, fire the banner only on the second
  consecutive one. 90 s timeout → `zero_actions()`. After success the runner zeroes actions
  and awaits the next command on stdin — multiple commands per session (the judge demo
  issues a second command in the same world).
- **One frame path everywhere.** The downscale + JPEG-encode that builds the VLM prompt
  lives in `dreampilot/frames.py` and is used by both the offline gate (recorded frames)
  and the live loop. A second copy of that code is a bug even if identical today.
- **Policy memory:** vision = latest frame only; text = the VLM's last 3–5 actions +
  one-line reasonings (cheap hysteresis against turn-left/turn-right hunting). No frame
  history in the prompt.
- **The fallback is a mode, not a rewrite:** the scripted-search fallback (VLM answers only
  "landmark visible? left/center/right?" + a tiny controller) is `ScriptedSearchPolicy` in
  `dreampilot/policy/`, selected with `--mode fallback`. A new behavior is a new `Policy`
  subclass registered in `MODES` — never a runner edit.
- **World prompts are static scene descriptions — no motion verbs.** Motion language in the
  prompt ("the camera pans across") overrides movement commands and silently fights the
  policy.
- `REACTOR_API_KEY` and the VLM key come from the environment; never commit them or bake
  them into scripts. The VLM client is provider-agnostic (one small function behind an
  OpenAI-compatible interface; model name, key, base URL from env) — switchable in one line
  if a key dies at the venue.

## Setup

```bash
uv venv --python 3.12
uv pip install reactor-sdk==0.8.0 openai pillow numpy
```

Install everything in **one resolve** so uv reconciles the SDK's numpy 2.5.1 pin.
Run everything with `.venv/bin/python` (uv venvs have no `pip` binary — use `uv pip`).
The BC-plan deps (torch, torchvision, pygame, open_clip_torch) are no longer needed —
harmless if already installed, don't add them to new environments.
