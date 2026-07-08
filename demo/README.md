# DreamPilot demo kit (2026-07-08)

## Live demo (the real thing)

```bash
VLM_IMAGE_DETAIL=high .venv/bin/python run_agent.py --world village   # or: barn, lighthouse
```

Judge types commands at the `command>` prompt (e.g. the ones in `worlds.json`);
empty line or `q` disconnects (STOPS BILLING). Every session auto-downloads its
server-side recording before disconnecting. `--world barn` is the strongest
judge-drivable world (two distinct landmarks: black barn, gray farmhouse).

## Recorded fallback clips (if wifi/API dies)

| Clip | What it shows |
|---|---|
| `village_maypole_success_19s.mp4` | World born from seed image, agent drives to the maypole, arrives in 19 s |
| `barn_two_commands_both_succeed.mp4` | Judge-demo shape: "walk up to the black barn" (12 s) then "go to the gray two-story farmhouse" (58 s) — includes tree sidestep + re-acquiring the farmhouse behind a building |
| `lighthouse_rocks_success_at_end.mp4` | Honest failure + recovery material: lighthouse episode times out (landmark permanence), then "go to the rocks by the water" succeeds in 21 s at the end |

Full session recordings live in `data/live/run_*/recording_*.mp4`.
Clips on Reactor's side expire 24 h after recording — these local copies are the durable ones.

## Numbers for the pitch (all measured today)

- 5 command→navigation successes, 4 on tape, across 3 worlds (village, lighthouse-coast, farmyard).
- Decision loop: ~0.3 Hz sequential (VLM ~1.5 s + 2 s settle, action-to-effect 1.5 s per `measured.json`).
- 100% valid JSON actions across all live sessions (0 held ticks in 51 live decisions).
- Live cost: ~$0.80 per 4-minute two-command session ($11.88/hr, billed per second from READY).
