"""Load-test the real Docker image: start it, hammer it, stop it. No API key, no cost.

    docker build -t jobshop-agent .                      # once
    uv run python scripts/load_test_container.py         # 20 users, 25 requests each
    uv run python scripts/load_test_container.py --users 50 --requests 40

The image runs its own Python and solver; the model is the scripted demo stand-in (the same one ``scripts/demo_server.py`` uses),
mounted from this checkout, so the chat-contention part can be tested too. The numbers describe this machine and a stand-in
model: they say the web layer in the container behaves, not how a real model will perform.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "jobshop-load"
IMAGE = "jobshop-agent"
SERVE = (
    "import sys, uvicorn\n"
    "sys.path.insert(0, '/app/scripts')\n"
    "import demo_server\n"
    "uvicorn.run(demo_server.build_app(), host='0.0.0.0', port=8000, log_level='warning')\n"
)


def sh(*args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, text=True, **kw)


def docker_ready() -> str | None:
    """None if Docker is usable, else what is wrong."""
    if shutil.which("docker") is None:
        return "docker is not installed or not on PATH"
    if sh("docker", "info").returncode != 0:
        return "the Docker daemon is not running (start Docker Desktop)"
    if sh("docker", "image", "inspect", IMAGE).returncode != 0:
        return f"the image '{IMAGE}' does not exist (build it: docker build -t {IMAGE} .)"
    return None


def start(port: int) -> None:
    sh("docker", "rm", "-f", NAME)
    # Bound to loopback only: the app has no login.
    result = sh("docker", "run", "-d", "--rm", "--name", NAME, "-p", f"127.0.0.1:{port}:8000",
                "-v", f"{(ROOT / 'scripts').as_posix()}:/app/scripts:ro", "--entrypoint", "python", IMAGE, "-c", SERVE)
    if result.returncode != 0:
        raise RuntimeError(f"could not start the container: {result.stderr.strip()}")


def wait_healthy(port: int, seconds: float = 60.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def stop() -> None:
    sh("docker", "stop", NAME)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--requests", type=int, default=25)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    problem = docker_ready()
    if problem:
        print(f"Cannot run: {problem}.", file=sys.stderr)
        return 2
    try:
        start(args.port)
        if not wait_healthy(args.port):
            print("The container did not become healthy in 60 s. Its log:\n" + sh("docker", "logs", NAME).stdout[-1500:], file=sys.stderr)
            return 1
        print(f"Load-testing the container on 127.0.0.1:{args.port} ({args.users} users, {args.requests} requests each)\n", flush=True)
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / "load_test.py"), "--url", f"http://127.0.0.1:{args.port}",
                               "--users", str(args.users), "--requests", str(args.requests)], cwd=ROOT)
        return done.returncode
    finally:
        stop()


if __name__ == "__main__":
    raise SystemExit(main())
