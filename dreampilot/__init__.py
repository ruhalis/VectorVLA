"""DreamPilot: a cloud VLM navigating inside LingBot-World via Reactor.

Layout:
    actions.py        shared action vocabulary (the LingBot per-axis enums)
    config.py         repo root, .env, measured.json, worlds.json — one load point
    frames.py         THE single downscale+JPEG+base64 frame path
    vlm.py            provider-agnostic OpenAI-compatible VLM client
    policy/           Policy base + one class per mode (navigator, scripted search)
    reactor_client.py session wrapper: credit meter, READY-gated sends, ring buffer
    runner.py         the ~0.5 Hz sequential control loop and CLI

Run live: python -m dreampilot --world village
Kept intentionally import-light: importing dreampilot must not pull reactor_sdk.
"""
