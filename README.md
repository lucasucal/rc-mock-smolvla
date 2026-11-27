# rc-mock-smolvla

Reality Check (contract v0) submission wrapping LeRobot SmolVLA (`lerobot==0.7.0`).

- Entry point: `policy.smolvla:SmolVLAPolicy` (see `manifest.json`)
- Model (public, ungated): https://huggingface.co/lucaslim/rc-mock-smolvla-ft — mounted at `/weights`.
  It also carries the SmolVLM2 backbone (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, Apache-2.0) as a
  hub cache in `hub-cache/`; the policy points `HF_HUB_CACHE` there, so it loads fully offline.
- Self-test: `python3 scripts/selftest.py` (ends with `ALL PASS`)

Local smoke run without weights (development only):

    RC_ALLOW_BASE_MODEL_FALLBACK=1 PYTHONPATH=rc_sdk python3 -m rc_sdk \
        --manifest manifest.json --policy policy.smolvla:SmolVLAPolicy
