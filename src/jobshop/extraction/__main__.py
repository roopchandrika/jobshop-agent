"""``python -m jobshop.extraction evaluate | extract``.

evaluate   score an extractor on the labelled emails (the rule-based baseline is free; ``--extractor llm`` calls the API)
extract    read one email file (plain text or .eml) and print the checked result a planner would review
"""

from __future__ import annotations

import argparse
import email
import os
import sys
from datetime import datetime
from email import policy
from pathlib import Path

from dotenv import load_dotenv

from jobshop.extraction import evaluate
from jobshop.extraction.baseline import extract_baseline
from jobshop.extraction.llm import extract_with_llm
from jobshop.extraction.validate import validate

DEFAULT_EMAILS = Path("evals/extraction/emails.yaml")
MAX_EMAIL_CHARS = 20_000


def read_email(path: Path) -> str:
    """The text of a plain-text file or a .eml message (headers that matter plus the body), size-capped."""
    raw = path.read_bytes()
    if path.suffix.lower() == ".eml":
        message = email.message_from_bytes(raw, policy=policy.default)
        body = message.get_body(preferencelist=("plain",))
        text = f"Subject: {message['subject'] or ''}\n\n{body.get_content() if body else ''}"
    else:
        text = raw.decode("utf-8", errors="replace")
    return text[:MAX_EMAIL_CHARS]


def _llm_extractor(model: str):
    import anthropic

    client = anthropic.Anthropic()
    usage = {"in": 0, "out": 0}

    def run(text, now, families):
        result = extract_with_llm(client, model, text, now, families)
        usage["in"] += result.input_tokens
        usage["out"] += result.output_tokens
        return result.extraction

    return run, usage


def _evaluate(args: argparse.Namespace) -> int:
    try:
        data = evaluate.load_dataset(args.emails)
    except (OSError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    extractors = {"baseline": lambda t, n, f: extract_baseline(t, n, f)}
    usage = None
    if args.extractor == "llm":
        load_dotenv()
        model = args.model or os.environ.get("ANTHROPIC_MODEL", "")
        if not model or not os.environ.get("ANTHROPIC_API_KEY"):
            print("Configuration error: --extractor llm needs ANTHROPIC_API_KEY and a model (--model or ANTHROPIC_MODEL).", file=sys.stderr)
            return 2
        print(f"Calling {model} once per email ({len(data.emails)} small calls).")
        run, usage = _llm_extractor(model)
        extractors = {f"llm ({model})": run}
    for name, extractor in extractors.items():
        results = evaluate.run_extractor(extractor, data)
        print(evaluate.render(name, results, data))
        print("\n  by kind of email:\n" + evaluate.render_by_tag(results))
    if usage:
        print(f"\ntokens used: {usage['in']} in, {usage['out']} out")
    return 0


def _extract(args: argparse.Namespace) -> int:
    try:
        text = read_email(args.file)
        data = evaluate.load_dataset(args.emails)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    now = datetime.strptime(args.now, "%Y-%m-%d %H:%M") if args.now else data.now
    verdict = validate(extract_baseline(text, now, data.families), text, data.families, now)
    print(f"status: {verdict.status}")
    for k, v in verdict.order.model_dump().items():
        print(f"  {k}: {v if v is not None else '-'}")
    for p in verdict.problems:
        print(f"  ! {p}")
    if verdict.suggested_request:
        print(f"\nYou could ask the assistant:  {verdict.suggested_request}")
    print("\nThis is a proposal read from the email. Check it against the email before acting on it.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jobshop.extraction", description=__doc__.strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="score an extractor on the labelled emails")
    ev.add_argument("--extractor", choices=["baseline", "llm"], default="baseline")
    ev.add_argument("--model", help="model for --extractor llm (default: $ANTHROPIC_MODEL)")
    ev.add_argument("--emails", type=Path, default=DEFAULT_EMAILS)
    ev.set_defaults(func=_evaluate)
    ex = sub.add_parser("extract", help="read one email and print the checked order")
    ex.add_argument("file", type=Path)
    ex.add_argument("--now", help="plant time, YYYY-MM-DD HH:MM (default: the labelled set's clock)")
    ex.add_argument("--emails", type=Path, default=DEFAULT_EMAILS, help="for the list of known families")
    ex.set_defaults(func=_extract)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
