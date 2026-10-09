"""Load-test the web app: reads under concurrency, and what happens when several people chat at once.

    uv run python scripts/load_test.py --spawn                    # start the scripted demo app on a free port and test it
    uv run python scripts/load_test.py --url http://127.0.0.1:8000  # test one that is already running

What it measures, and why these two:

1. **Reads.** N users each fetch the plan state and the health endpoint repeatedly. Latency percentiles and errors.
   The state endpoint takes the store's lock for a moment, so this shows whether readers queue behind each other.
2. **Chat contention.** N users send a chat message at the same instant. The app is built for ONE planner and the
   agent owns the store while it works, so the right answer is: exactly one message accepted (202), the rest told
   to wait (409), nobody hung, no 5xx. This checks that rule under real concurrency, not just in a unit test.
3. **Reads during a turn.** While that one turn is running, readers keep calling the state endpoint. It must stay
   responsive (answer "busy") instead of blocking until the solver finishes.

``--spawn`` uses the scripted demo (canned model, REAL solver), so it costs nothing and needs no key. The numbers
describe this machine, this solver limit and a stand-in model: they say the web layer behaves, not how a real model
will perform. A real model's latency is dominated by the model, not by anything here.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import socket
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


@dataclass
class Latency:
    count: int = 0
    errors: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    statuses: dict[int, int] = field(default_factory=dict)


def summarise(samples: list[tuple[int, float]]) -> Latency:
    times = [s * 1000 for status, s in samples if status < 500 and status != 0]
    statuses: dict[int, int] = {}
    for status, _ in samples:
        statuses[status] = statuses.get(status, 0) + 1
    return Latency(len(samples), sum(1 for status, _ in samples if status == 0 or status >= 500),
                   round(percentile(times, 50), 1), round(percentile(times, 95), 1), round(max(times, default=0.0), 1), statuses)


def request(base: str, method: str, path: str, body: dict | None = None, token: str | None = None, timeout: float = 30.0) -> tuple[int, float, bytes]:
    """One HTTP request on its own connection: (status, seconds, body). Status 0 means the connection itself failed."""
    url = urlparse(base)
    began = time.perf_counter()
    try:
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
        headers = {"Content-Type": "application/json"} if body is not None else {}
        if token:
            headers["X-CSRF-Token"] = token
        conn.request(method, path, json.dumps(body) if body is not None else None, headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, time.perf_counter() - began, data
    except (OSError, http.client.HTTPException):
        return 0, time.perf_counter() - began, b""


def csrf_token(base: str) -> str:
    status, _, page = request(base, "GET", "/")
    match = re.search(rb'name="csrf-token" content="([^"]+)"', page)
    if status != 200 or not match:
        raise RuntimeError(f"could not load the page at {base} (status {status})")
    return match.group(1).decode()


def run_reads(base: str, users: int, per_user: int) -> dict[str, Latency]:
    samples: dict[str, list[tuple[int, float]]] = {"/api/state": [], "/healthz": []}
    lock = threading.Lock()
    barrier = threading.Barrier(users)

    def worker() -> None:
        barrier.wait()
        for _ in range(per_user):
            for path in samples:
                status, took, _ = request(base, "GET", path)
                with lock:
                    samples[path].append((status, took))

    threads = [threading.Thread(target=worker) for _ in range(users)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return {path: summarise(s) for path, s in samples.items()}


@dataclass
class Contention:
    users: int
    accepted: int
    refused_busy: int
    server_errors: int
    turn_finished: bool
    state_during_turn: Latency
    busy_answers: int


def run_chat_contention(base: str, users: int, message: str = "M1 is down until 16:00. What happens?") -> Contention:
    token = csrf_token(base)
    results: list[tuple[int, float, bytes]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(users)

    def sender() -> None:
        barrier.wait()
        r = request(base, "POST", "/api/chat", {"message": message}, token)
        with lock:
            results.append(r)

    # Readers poll the plan state while the one accepted turn runs.
    stop = threading.Event()
    reads: list[tuple[int, float]] = []
    busy = [0]

    def reader() -> None:
        while not stop.is_set():
            status, took, body = request(base, "GET", "/api/state")
            with lock:
                reads.append((status, took))
                if status == 200 and b'"busy":true' in body.replace(b" ", b""):
                    busy[0] += 1
            time.sleep(0.02)

    readers = [threading.Thread(target=reader) for _ in range(3)]
    senders = [threading.Thread(target=sender) for _ in range(users)]
    [t.start() for t in readers + senders]
    [t.join() for t in senders]

    turn_id = next((json.loads(body)["turn_id"] for status, _, body in results if status == 202), None)
    finished = False
    if turn_id is not None:
        deadline = time.time() + 120
        while time.time() < deadline:
            status, _, body = request(base, "GET", f"/api/chat/{turn_id}")
            if status == 200 and json.loads(body)["status"] == "done":
                finished = True
                break
            time.sleep(0.1)
    stop.set()
    [t.join() for t in readers]
    codes = [status for status, _, _ in results]
    return Contention(users, codes.count(202), codes.count(409), sum(1 for c in codes if c == 0 or c >= 500), finished, summarise(reads), busy[0])


def spawn_demo():
    """The scripted demo app on a free port, in this process. Returns (base_url, stop_function)."""
    import importlib.util

    import uvicorn

    spec = importlib.util.spec_from_file_location("demo_server", ROOT / "scripts" / "demo_server.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(module.build_app(solve_seconds=2.0), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        if request(base, "GET", "/healthz")[0] == 200:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("the demo app did not start")

    def stop() -> None:
        server.should_exit = True
        thread.join(timeout=10)

    return base, stop


def render(reads: dict[str, Latency], contention: Contention) -> str:
    lines = ["Reads under concurrency", f"  {'endpoint':<12} {'requests':>9} {'errors':>7} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8}"]
    for path, lat in reads.items():
        lines.append(f"  {path:<12} {lat.count:>9} {lat.errors:>7} {lat.p50_ms:>8} {lat.p95_ms:>8} {lat.max_ms:>8}")
    c = contention
    lines += [
        "",
        f"Chat contention ({c.users} users send at the same instant)",
        f"  accepted (202)          {c.accepted}   (the app serves one planner: exactly 1 is correct)",
        f"  told to wait (409)      {c.refused_busy}",
        f"  failed / server error   {c.server_errors}",
        f"  the accepted turn finished  {c.turn_finished}",
        "",
        "Reads while that turn ran",
        f"  state requests {c.state_during_turn.count}, p50 {c.state_during_turn.p50_ms} ms, p95 {c.state_during_turn.p95_ms} ms, "
        f"max {c.state_during_turn.max_ms} ms; {c.busy_answers} answered 'busy' instead of waiting",
    ]
    return "\n".join(lines)


def verdict(reads: dict[str, Latency], c: Contention) -> list[str]:
    """What is wrong, if anything. Empty means the web layer behaved."""
    problems = []
    for path, lat in reads.items():
        if lat.errors:
            problems.append(f"{path}: {lat.errors} failed or 5xx responses")
    if c.accepted != 1:
        problems.append(f"expected exactly 1 chat message accepted, got {c.accepted}")
    if c.accepted + c.refused_busy != c.users:
        problems.append(f"{c.users - c.accepted - c.refused_busy} chat requests got neither 202 nor 409")
    if c.server_errors:
        problems.append(f"{c.server_errors} chat requests failed or returned 5xx")
    if not c.turn_finished:
        problems.append("the accepted turn never finished")
    if c.state_during_turn.errors:
        problems.append("reads failed while a turn was running")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--url", help="base URL of a running app, e.g. http://127.0.0.1:8000")
    where.add_argument("--spawn", action="store_true", help="start the scripted demo app and test that")
    parser.add_argument("--users", type=int, default=20, help="concurrent users (default 20)")
    parser.add_argument("--requests", type=int, default=25, help="requests per user per endpoint for the read test (default 25)")
    parser.add_argument("--json", action="store_true", help="print the numbers as JSON")
    args = parser.parse_args(argv)
    if args.users < 2 or args.requests < 1:
        parser.error("--users must be at least 2 and --requests at least 1")

    base, stop = (spawn_demo() if args.spawn else (args.url.rstrip("/"), lambda: None))
    try:
        reads = run_reads(base, args.users, args.requests)
        contention = run_chat_contention(base, args.users)
    finally:
        stop()
    problems = verdict(reads, contention)
    if args.json:
        print(json.dumps({"reads": {p: asdict(v) for p, v in reads.items()}, "contention": asdict(contention), "problems": problems}, indent=1))
    else:
        print(render(reads, contention))
        print("\n" + ("OK: the web layer behaved." if not problems else "PROBLEMS:\n  " + "\n  ".join(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
