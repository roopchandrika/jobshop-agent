"""Record a run's model responses once, replay them offline for free.

An eval run against a real model costs API credit. Most of what you want to test afterwards is
*our* code (the checks, the tools, the report, a change to the harness), and that does not need the
model to answer again. So:

* ``Recorder`` wraps the real client and saves every response, with a fingerprint of the request
  that produced it, under ``<dir>/<scenario>-<attempt>.json``.
* ``Replayer`` plays those responses back in order. Before each one it compares the fingerprint of
  the request it is *now* being asked to answer with the recorded one. If they differ, the prompt, the
  tool definitions or something a tool returned has changed since the recording, so the model's saved
  answer is no longer an answer to this question. That scenario fails with a clear "stale recording"
  error instead of quietly scoring an answer to a different request.

That staleness check is the useful part: it is a free regression alarm for prompt and tool changes.
It does not say whether the new prompt is *better*; that still needs a real run.

Volatile values (the solver's wall-clock seconds) and request-only markers (``cache_control``) are
removed before fingerprinting, because they differ between runs without changing the request's meaning.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anthropic.types import Message

FORMAT = 1
_VOLATILE_KEYS = {"solve_seconds"}


class StaleRecording(Exception):
    """The request no longer matches the recorded one (or the recording ran out)."""


def _clean(value: Any) -> Any:
    """Drop request-only markers and volatile values, recursively, so equal requests hash equally."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k != "cache_control" and k not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, str) and value[:1] in "{[":
        try:  # a tool result is a JSON string; clean what is inside it too
            return json.dumps(_clean(json.loads(value)), sort_keys=True)
        except ValueError:
            return value
    return value


def fingerprint(request: dict[str, Any]) -> str:
    """A short hash of what the model was asked: system prompt, tools and the whole conversation."""
    body = _clean({"system": request.get("system"), "tools": request.get("tools"), "messages": request.get("messages")})
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


def recording_path(directory: Path, scenario_id: str, attempt: int) -> Path:
    return directory / f"{scenario_id}-{attempt}.json"


class Recorder:
    """Quacks like ``anthropic.Anthropic``; forwards to ``real`` and keeps what came back."""

    def __init__(self, real: Any, path: Path, model: str) -> None:
        self._real, self._path, self._model = real, path, model
        self._steps: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Message:
        response = self._real.messages.create(**kwargs)
        self._steps.append({"fingerprint": fingerprint(kwargs), "response": response.model_dump(mode="json")})
        self.save()  # after every call, so a crash or a stopped run keeps what was paid for
        return response

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({
            "format": FORMAT, "model": self._model, "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "steps": self._steps,
        }, indent=1), encoding="utf-8")


class Replayer:
    """Quacks like ``anthropic.Anthropic``; answers from a recording, never from the network."""

    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f"{path.name}: unsupported recording format {data.get('format')!r}")
        self._name, self.model, self._steps = path.name, data["model"], list(data["steps"])
        self._n = 0
        self.messages = self

    def create(self, **kwargs: Any) -> Message:
        if self._n >= len(self._steps):
            raise StaleRecording(
                f"{self._name}: the agent made call {self._n + 1} but the recording has only {len(self._steps)}; "
                "the conversation now goes differently. Re-record with --record."
            )
        step = self._steps[self._n]
        self._n += 1
        if fingerprint(kwargs) != step["fingerprint"]:
            raise StaleRecording(
                f"{self._name}: call {self._n} is a different request from the recorded one (the system prompt, the "
                "tool definitions or an earlier tool result changed). The saved answer no longer applies. "
                "Re-record with --record."
            )
        return Message.model_validate(step["response"])


def recording_factory(directory: Path, make_real: Callable[[Any], Any], model: str) -> Callable[[Any], Any]:
    """``client_for`` for the runner that records. Counts attempts per scenario, as the runner does not pass them."""
    attempts: dict[str, int] = defaultdict(int)

    def client_for(scenario: Any) -> Recorder:
        attempts[scenario.id] += 1
        return Recorder(make_real(scenario), recording_path(directory, scenario.id, attempts[scenario.id]), model)

    return client_for


def replay_factory(directory: Path) -> Callable[[Any], Replayer]:
    attempts: dict[str, int] = defaultdict(int)

    def client_for(scenario: Any) -> Replayer:
        attempts[scenario.id] += 1
        path = recording_path(directory, scenario.id, attempts[scenario.id])
        if not path.exists():
            raise FileNotFoundError(f"no recording for {scenario.id} run {attempts[scenario.id]}: {path}")
        return Replayer(path)

    return client_for


def available(directory: Path) -> dict[str, int]:
    """Scenario id -> number of recorded runs found in ``directory``."""
    counts: dict[str, int] = defaultdict(int)
    for p in directory.glob("*-*.json"):
        counts[p.stem.rsplit("-", 1)[0]] += 1
    return dict(counts)
