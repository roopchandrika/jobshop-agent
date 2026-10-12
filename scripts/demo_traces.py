"""See an agent turn as a trace in Jaeger: start Jaeger, load a trace, print where to look. No API key, no cost.

    uv run python scripts/demo_traces.py                      # make a scripted turn, load it, wait for Enter, stop Jaeger
    uv run python scripts/demo_traces.py --trace logs/traces/<file>.jsonl     # load a trace you already have
    uv run python scripts/demo_traces.py --keep               # leave Jaeger running afterwards
    uv run python scripts/demo_traces.py --open               # also open the browser

Needs Docker. The first run pulls the official ``jaegertracing/jaeger`` image (a few hundred MB). Only metadata leaves the
trace (model, tokens, latency, tool names), as with ``python -m jobshop.agent.trace_export``; Jaeger is bound to loopback only.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "jaeger-demo"
IMAGE = "jaegertracing/jaeger:latest"
UI_PORT, OTLP_PORT = 16686, 4318


def sh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, text=True)


def docker_problem() -> str | None:
    if shutil.which("docker") is None:
        return "docker is not installed or not on PATH"
    if sh("docker", "info").returncode != 0:
        return "the Docker daemon is not running (start Docker Desktop)"
    return None


def start_jaeger() -> None:
    sh("docker", "rm", "-f", NAME)
    result = sh("docker", "run", "-d", "--rm", "--name", NAME, "-p", f"127.0.0.1:{UI_PORT}:16686", "-p", f"127.0.0.1:{OTLP_PORT}:4318", IMAGE)
    if result.returncode != 0:
        raise RuntimeError(f"could not start Jaeger: {result.stderr.strip()[-400:]}")


def wait_ready(seconds: float = 90.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{UI_PORT}/", timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def make_trace(workdir: Path) -> Path:
    """A real trace from a scripted turn: the reference agent plays one scenario through the actual loop and solver."""
    out = workdir / "run"
    code = subprocess.run([sys.executable, "-m", "jobshop.evals", "run", "--oracle", "--only", "sd-01", "--only", "q-04", "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True)
    if code.returncode != 0:
        raise RuntimeError("could not make a demo trace:\n" + code.stdout[-600:] + code.stderr[-600:])
    return next(out.rglob("traces"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", type=Path, nargs="+", help="trace file(s) to load (default: make a scripted turn)")
    parser.add_argument("--keep", action="store_true", help="leave Jaeger running when done")
    parser.add_argument("--open", action="store_true", help="open the Jaeger page in the browser")
    parser.add_argument("--no-wait", action="store_true", help="do not wait for Enter before stopping Jaeger")
    args = parser.parse_args(argv)

    problem = docker_problem()
    if problem:
        print(f"Cannot run: {problem}.", file=sys.stderr)
        return 2
    from jobshop.agent.trace_export import send, to_otlp
    from jobshop.agent.trace_report import read_trace

    started = False
    try:
        print("Starting Jaeger (the first run pulls the image)...", flush=True)
        start_jaeger()
        started = True
        if not wait_ready():
            print("Jaeger did not come up in 90 s. Its log:\n" + sh("docker", "logs", NAME).stdout[-1200:], file=sys.stderr)
            return 1
        files = list(args.trace) if args.trace else sorted(make_trace(Path(tempfile.mkdtemp(prefix="jobshop-traces-"))).glob("*.jsonl"))
        records = []
        for f in files:
            records += read_trace(f)
        status = send(to_otlp(records), f"http://127.0.0.1:{OTLP_PORT}/v1/traces")
        spans = sum(len(s["spans"]) for rs in to_otlp(records)["resourceSpans"] for s in rs["scopeSpans"])
        url = f"http://127.0.0.1:{UI_PORT}/search?service=jobshop-agent"
        print(f"Sent {spans} spans from {len(files)} turn(s) (HTTP {status}).\n\n  Open {url}\n"
              "  Pick a trace: each model call and tool call is a span, with tokens, model and latency.\n", flush=True)
        if args.open:
            webbrowser.open(url)
        if not (args.no_wait or args.keep):
            input("Press Enter to stop Jaeger... ")
        return 0
    finally:
        if started and not args.keep:
            sh("docker", "stop", NAME)
        elif started:
            print(f"Jaeger left running (stop it with: docker stop {NAME})")


if __name__ == "__main__":
    raise SystemExit(main())
