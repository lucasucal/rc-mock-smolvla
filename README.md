# rc-mock-smolvla

Reality Check (contract v0) submission wrapping LeRobot SmolVLA (`lerobot==0.7.0`).

- Entry point: `policy.smolvla:SmolVLAPolicy` (see `manifest.json`)
- Model (public, ungated): https://huggingface.co/lucaslim/rc-mock-smolvla-ft — mounted at `/weights`
- Self-test: `python3 scripts/selftest.py` (ends with `ALL PASS`)

Local smoke run without weights (development only):

    RC_ALLOW_BASE_MODEL_FALLBACK=1 PYTHONPATH=rc_sdk python3 -m rc_sdk \
        --manifest manifest.json --policy policy.smolvla:SmolVLAPolicy
