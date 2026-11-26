"""Self-test for your policy repository: python3 scripts/selftest.py   (ends with ALL PASS)

It checks, without downloading any model:
  1. manifest.json (contract v0, a valid entry_point);
  2. your entry point: a Policy class is imported (with the judged image's loader when rc_runtime
     is importable, otherwise a plain import), and the starter example also loads a tiny generated
     checkpoint and acts when torch and safetensors are installed; a program entry point
     (serve.py) is started and checked over HTTP;
  3. the example servers in examples/ (FastAPI is skipped when it is not installed);
  4. the HTTP contract of the SDK server: /healthz at once, /readyz after load, /act echoes
     episode_id and step, a malformed /act gets 400, a failed load stays not-ready.

It imports rc_sdk from ./rc_sdk (or from RC_SDK_DIR when set). Inside the
published image run `python -I -m rc_runtime --selfcheck --submission /submission` instead.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import socket
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
# Package root is starter/rc_sdk (the installable project). Works from starter/ without pip. Without that
# folder, point RC_SDK_DIR at the bundle's starter/rc_sdk or pip install it; the published image ships rc_sdk.
SDK_DIRS = [d for d in (os.environ.get("RC_SDK_DIR"), os.path.join(ROOT, "rc_sdk")) if d]
for sdk in reversed(SDK_DIRS):
    if os.path.isdir(os.path.join(sdk, "rc_sdk")):
        sys.path.insert(0, sdk)
for candidate in (os.path.join(ROOT, "..", "runtime"),):  # monorepo checkout
    if os.path.isdir(os.path.join(candidate, "rc_runtime")):
        sys.path.append(candidate)

try:
    from rc_sdk.policy import Policy  # noqa: E402
    from rc_sdk.server import PolicyServer  # noqa: E402
except ImportError:
    sys.stderr.write(
        "FAIL rc_sdk not found. The self-test needs the starter SDK: keep rc_sdk/ in your copy,\n"
        "or set RC_SDK_DIR to the participant bundle's starter/rc_sdk, or\n"
        "`pip install <bundle>/starter/rc_sdk`, or run it inside the published image, which ships rc_sdk.\n")
    sys.exit(2)

sys.path.insert(0, HERE)  # the stdlib checkpoint writer shared with make_example_weights.py
from make_example_weights import write_example_weights  # noqa: E402

try:
    from rc_runtime import loader as rt_loader  # noqa: E402
    from rc_runtime.server import RuntimeServer  # noqa: E402
except ImportError:
    rt_loader = None
    RuntimeServer = None

EXAMPLE_ENTRY = "policy.example:ExamplePolicy"  # the only entry point the tiny generated checkpoint fits
ENTRY_POINT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$")
MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
PATH_PART_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")


def entry_kind(value) -> str:
    """'class', 'module' or 'file'; raises ValueError with the reason otherwise (same grammar as rc_runtime)."""
    if not isinstance(value, str) or not value:
        raise ValueError("entry_point must be a non-empty string")
    if ":" in value:
        if ENTRY_POINT_RE.fullmatch(value):
            return "class"
        raise ValueError('a Policy class entry point must look like "package.module:Attr"')
    if "/" in value or value.endswith(".py") or "\\" in value:
        parts = value.split("/")
        if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
            raise ValueError("absolute paths are not allowed; use a path relative to the repository root")
        if "\\" in value or any(p in ("", ".", "..") for p in parts) or not all(PATH_PART_RE.match(p) for p in parts):
            raise ValueError("a program path must stay inside the repository (no '..', '.', empty segments, backslashes)")
        if not value.endswith(".py"):
            raise ValueError("a program path must end in .py")
        return "file"
    if MODULE_RE.fullmatch(value):
        return "module"
    raise ValueError('entry_point must be "package.module:Attr", a dotted module path, or a relative .py path')


class FakePolicy(Policy):
    def load(self) -> None:
        time.sleep(0.3)  # simulate weight load

    def act(self, obs):
        return {"type": "joint_delta", "values": [0.0] * 6, "gripper": 0.0}


class FailedLoadPolicy(Policy):
    def load(self) -> None:
        raise RuntimeError("sensitive diagnostic detail")

    def act(self, obs):
        raise AssertionError("act must not run after failed load")


def request(port: int, method: str, path: str, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"} if body else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read()), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}"), dict(err.headers)


def make_server(policy_class):
    """The judged image's server when rc_runtime is available, else the rc_sdk server."""
    if RuntimeServer is not None:
        def loader():
            policy = policy_class()
            rt_loader.run_load(policy)
            return policy
        return RuntimeServer("127.0.0.1", 0, loader)
    return PolicyServer(0, policy_class())


def check_manifest() -> dict:
    manifest_path = os.path.join(ROOT, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as source:
            manifest = json.load(source)
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        if manifest.get("contract") != "v0":
            raise ValueError('contract must be "v0"')
        entry_kind(manifest.get("entry_point"))
        if "port" in manifest and (type(manifest["port"]) is not int or manifest["port"] != 8080):
            raise ValueError("port must be 8080 or omitted (the image always serves on 8080)")
        weights = manifest.get("weights")
        if weights is not None and not isinstance(weights, dict):
            raise ValueError("weights must be an object when present (placeholder; ignored)")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"FAIL manifest.json: {exc}") from exc
    print("PASS manifest.json is valid (model link is supplied in the dashboard; weights not downloaded)")
    return manifest


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def check_program(root, entry: str, label: str) -> None:
    """Start a program-form entry point for real and assert contract v0 over HTTP (no torch needed)."""
    from pathlib import Path
    root = Path(root)
    with tempfile.TemporaryDirectory() as weights, tempfile.TemporaryDirectory() as scratch:
        write_example_weights(weights, scale=0.1)
        port = free_port()
        env = {k: v for k, v in os.environ.items() if not k.startswith("RC_")}
        env.update(RC_WEIGHTS_DIR=weights, RC_SCRATCH=scratch, RC_POLICY_PORT=str(port), PYTHONUNBUFFERED="1")
        if rt_loader is not None:
            env["PYTHONPATH"] = os.pathsep.join([*SDK_DIRS, os.path.join(ROOT, "..", "runtime")])
            cmd = [sys.executable, "-m", "rc_runtime", "--submission", str(root), "--port", str(port)]
            how = "rc_runtime runs it as __main__"
        else:
            cmd = [sys.executable, entry] if entry.endswith(".py") else [sys.executable, "-m", entry]
            how = "plain python (rc_runtime not importable here)"
        log = tempfile.TemporaryFile(mode="w+")
        proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)

        def fail(message):
            proc.kill()
            proc.wait()
            log.seek(0)
            raise SystemExit(f"FAIL {label}: {message}\n--- program output ---\n{log.read()[-3000:]}")

        try:
            deadline = time.monotonic() + 30
            while True:
                if proc.poll() is not None:
                    fail(f"program exited early with status {proc.returncode}")
                try:
                    status, body, _ = request(port, "GET", "/healthz")
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        fail(f"nothing listening on 0.0.0.0:$RC_POLICY_PORT ({port}) after 30 s")
                    time.sleep(0.1)
            assert status == 200 and body == {}, (status, body)
            print(f"PASS {label}: /healthz 200 {{}} ({how})")
            while True:
                status, body, _ = request(port, "GET", "/readyz")
                if body.get("ready"):
                    break
                if body.get("error") or time.monotonic() > deadline:
                    fail(f"/readyz never became ready: {body}")
                time.sleep(0.1)
            assert status == 200 and body["contract"] == "v0" and body["schema_version"] == 2, body
            assert set(body["capabilities"]) >= {"chunking", "caching"} and body["capabilities"]["chunking"] is True, body
            print(f"PASS {label}: /readyz ready after weights load, contract v0, schema_version 2")
            obs = {"episode_id": "ep-9", "step": 3, "proprio": {"joint_pos": [0.1] * 6}}
            status, body, headers = request(port, "POST", "/act", obs)
            assert status == 200 and body["episode_id"] == "ep-9" and body["step"] == 3, (status, body)
            low = {k.lower(): v for k, v in headers.items()}  # header names are case-insensitive
            assert low.get("x-rc-episode-id") == "ep-9" and low.get("x-rc-step") == "3", headers
            action = body["action"]
            assert action["type"] == "joint_delta" and len(action["values"]) == 6, action
            assert all(abs(v) <= 0.05 for v in action["values"]) and 0.0 <= action["gripper"] <= 1.0, action
            print(f"PASS {label}: /act echoes episode_id/step (body + headers) and returns a valid joint_delta")
            status, body, _ = request(port, "POST", "/act", {"episode_id": "ep-9"})
            assert status == 400 and "step" in body["error"], (status, body)
            print(f"PASS {label}: malformed /act (missing step) -> 400")
        finally:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()


def stage_submission(source) -> "Path":
    import shutil
    from pathlib import Path
    import atexit
    staged = tempfile.mkdtemp(prefix="rc-submission-")
    atexit.register(shutil.rmtree, staged, True)
    root = Path(staged) / "submission"
    shutil.copytree(source, root, ignore=shutil.ignore_patterns(
        "rc_sdk", "Dockerfile", "Dockerfile.legacy", ".git", "__pycache__", "scripts", "examples"))
    return root


def check_example_programs() -> None:
    import importlib.util
    examples = os.path.join(ROOT, "examples")
    if not os.path.isdir(examples):
        return
    for name in sorted(os.listdir(examples)):
        folder = os.path.join(examples, name)
        manifest_path = os.path.join(folder, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path, encoding="utf-8") as source:
            entry = json.load(source)["entry_point"]
        label = f"examples/{name}"
        try:
            kind = entry_kind(entry)
        except ValueError as exc:
            raise SystemExit(f"FAIL {label}: manifest entry_point: {exc}")
        assert kind != "class", f"{label} is expected to be a program"
        if "fastapi" in name and (importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("uvicorn") is None):
            print(f"SKIP {label}: fastapi/uvicorn are not installed here (they are in the judged image)")
            continue
        check_program(stage_submission(folder), entry, label)


def check_entry_point(entry: str) -> None:
    """Resolve the entry point the way the image does, then (if torch is available) load and act
    on a tiny generated checkpoint in a fake /weights."""
    # The judged /submission is your repository without a Dockerfile, .git or any rc_sdk copy
    # (rc_sdk comes from the image). Mimic it so rc_sdk is imported from outside the tree.
    root = stage_submission(ROOT)
    if entry_kind(entry) != "class":
        check_program(root, entry, "your program entry point")
        return
    try:
        if rt_loader is not None:
            reserved = rt_loader.load_reserved_modules(env={})
            policy = rt_loader.resolve_entry(root, entry, reserved)
            print("PASS entry point resolved by rc_runtime.loader (path appended last, shadow checks, Policy resolved)")
        else:
            import importlib
            sys.path.append(str(root))
            module, attr = entry.split(":")
            policy = getattr(importlib.import_module(module), attr)()
            print("SKIP rc_runtime not importable: resolved the entry point with importlib only "
                  "(run `python -I -m rc_runtime --selfcheck` in the image for the loader checks)")
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, ImportError) or "ImportError" in str(exc) or "ModuleNotFoundError" in str(exc):
            print(f"SKIP resolving {entry}: it imports a package that is not installed here ({exc}). The judged image "
                  "has the supported packages (see SUBMIT.md); install them locally (for example torch) or run "
                  "`python -I -m rc_runtime --selfcheck --submission /submission` inside the image.")
            return
        raise
    assert isinstance(policy, Policy), "entry point must resolve to a Policy"
    if entry != EXAMPLE_ENTRY:
        print(f"SKIP load/act of {entry}: the generated tiny /weights checkpoint only fits the starter example "
              f"({EXAMPLE_ENTRY}). Your policy loads the real /weights snapshot; check it with the practice tool "
              "(contestants/PRACTICE.md).")
        return
    try:
        import torch  # noqa: F401
        import safetensors.torch  # noqa: F401
    except ImportError:
        print("SKIP load/act of the example policy: torch and safetensors are not installed here (they are in the judged image)")
        return
    with tempfile.TemporaryDirectory() as weights:
        write_example_weights(weights)  # the same file scripts/make_example_weights.py writes
        os.environ["RC_WEIGHTS_DIR"] = weights
        policy.load()
        action = policy.act({"episode_id": "e", "step": 0, "proprio": {"joint_pos": [0.1] * 6}})
        assert action["type"] == "joint_delta" and len(action["values"]) == 6
        assert all(abs(v) <= 0.05 for v in action["values"]) and 0.0 <= action["gripper"] <= 1.0, action
        print("PASS example policy loads model.safetensors from RC_WEIGHTS_DIR and returns a valid joint_delta")


def main() -> None:
    manifest = check_manifest()
    check_entry_point(manifest["entry_point"])
    check_example_programs()
    server = make_server(FakePolicy)
    port = server.server_address[1]
    t0 = time.monotonic()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.start_load()

    status, body, _ = request(port, "GET", "/healthz")
    elapsed = time.monotonic() - t0
    assert status == 200 and body == {}, (status, body)
    assert elapsed < 0.2, f"healthz took {elapsed:.3f}s (liveness only!)"
    print(f"PASS /healthz 200 {{}} immediately (after {elapsed * 1000:.0f} ms, pre-weights)")

    status, body, _ = request(port, "GET", "/readyz")
    assert status == 200 and body["ready"] is False, (status, body)
    assert body["contract"] == "v0" and body["schema_version"] == 2
    assert body["capabilities"] == {"chunking": True, "caching": False}
    print("PASS /readyz not-ready before load, contract v0, schema_version 2, capabilities")

    time.sleep(0.5)
    status, body, _ = request(port, "GET", "/readyz")
    assert status == 200 and body["ready"] is True, (status, body)
    print("PASS /readyz flips ready after load() completes")

    obs = {"episode_id": "ep-123", "step": 7, "task_id": "t1",
           "images": {"wrist": "PLACEHOLDER", "left": "PLACEHOLDER", "right": "PLACEHOLDER"},
           "proprio": {"joint_pos": [0.0] * 6}}
    status, body, headers = request(port, "POST", "/act", obs)
    assert status == 200, (status, body)
    assert body["episode_id"] == "ep-123" and body["step"] == 7  # verbatim echo
    assert body["action"]["type"] == "joint_delta"
    assert headers.get("X-RC-Episode-Id") == "ep-123" and headers.get("X-RC-Step") == "7"
    print("PASS /act echoes episode_id/step verbatim (body + X-RC-* headers) + action")

    status, body, _ = request(port, "POST", "/act", {"episode_id": "ep-123"})
    assert status == 400 and "step" in body["error"], (status, body)
    print("PASS malformed /act (missing step) -> 400 with readable message")

    server.shutdown()

    failed_server = make_server(FailedLoadPolicy)
    failed_port = failed_server.server_address[1]
    threading.Thread(target=failed_server.serve_forever, daemon=True).start()
    failed_server.start_load()
    for _ in range(100):
        if failed_server.load_error is not None:
            break
        time.sleep(0.01)
    status, body, _ = request(failed_port, "GET", "/readyz")
    assert status == 200 and body["ready"] is False, (status, body)
    assert body["error"] == "policy failed to load", body
    assert "sensitive diagnostic detail" not in json.dumps(body), body
    failed_server.shutdown()
    print("PASS terminal load failure stays not-ready and exposes only bounded error")

    print("ALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
