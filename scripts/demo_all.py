"""The whole project in one command, for a live demo. No API key, no cost.

    uv run python scripts/demo_all.py            # every section, then the web UI
    uv run python scripts/demo_all.py --quick    # a faster pass (a subset of the evals), no web UI
    uv run python scripts/demo_all.py --pause    # wait for Enter between sections, to talk over each one

Each section shows one claim and the evidence for it. The model is a scripted stand-in everywhere here; the solver, the
checks, the guards and the plans are real. For the real-model results see docs/DEMO.md.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> int:
    return subprocess.run([sys.executable, *args], cwd=ROOT).returncode


def section(number: int, title: str, claim: str, pause: bool) -> None:
    print(f"\n{'=' * 100}\n  {number}. {title}\n  {claim}\n{'=' * 100}", flush=True)
    if pause:
        input("  (press Enter to run) ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="run a subset of the evals and skip the web UI")
    parser.add_argument("--pause", action="store_true", help="wait for Enter before each section")
    parser.add_argument("--no-web", action="store_true", help="skip the web UI at the end")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    out = tempfile.mkdtemp(prefix="jobshop-demo-")   # keeps evals/results clean
    started = time.perf_counter()
    failures: list[str] = []

    def step(label: str, code: int) -> None:
        if code != 0:
            failures.append(label)

    section(1, "The assistant cannot change the live plan, even when fully taken over",
            "30 attacks (order notes, planted documents, the planner's own words) against a model that obeys every one, "
            "scored on the store and the answer.", args.pause)
    step("red team", run("-m", "jobshop.evals", "redteam", "--out", out))

    section(2, "Behaviour is measured on state, not on wording",
            "Scenarios scored by checks that read the store and the tool log (scripted reference agent: tests the harness).", args.pause)
    only = ["--only", "simple_downtime", "--only", "injection", "--only", "read_only"] if args.quick else []
    step("evals", run("-m", "jobshop.evals", "run", "--oracle", *only, "--out", out))

    section(3, "Looking things up in plant documents, scored on its own",
            "Keyword search on labelled questions; embeddings beat it on hard paraphrases (6 of 6 against 3 of 6).", args.pause)
    step("retrieval", run("-m", "jobshop.evals", "retrieval", "--method", "bm25"))

    section(4, "No prompt or tool description changes silently",
            "The saved text of everything the model is told is compared with the code.", args.pause)
    step("snapshot", run("-m", "jobshop.evals", "snapshot"))

    print(f"\n{'=' * 100}\n  Done in {time.perf_counter() - started:.0f} s. "
          f"{'All sections passed.' if not failures else 'Sections that failed: ' + ', '.join(failures)}\n{'=' * 100}", flush=True)

    shutil.rmtree(out, ignore_errors=True)       # the scratch results are not kept
    if not (args.quick or args.no_web):
        print(f"\nStarting the web UI with a scripted model and the real solver: http://127.0.0.1:{args.port}   (Ctrl+C to stop)\n"
              "Type anything: the canned turns show a proposal with a late order, a second proposal, and a clarifying question.", flush=True)
        step("web", run("scripts/demo_server.py", "--port", str(args.port)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
