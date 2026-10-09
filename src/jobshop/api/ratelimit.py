"""A limit on how many agent turns the web app will start, so a runaway script or a stuck browser tab cannot spend without end.

The app has no login and only listens on loopback, so there is no "who" to limit; the thing to protect is the paid model API
behind it. Two sliding windows (per minute and per hour) count turns that actually started. A turn that was refused because the
assistant was busy does not count, since it cost nothing.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

PER_MINUTE_VAR = "JOBSHOP_RATE_LIMIT_PER_MIN"
PER_HOUR_VAR = "JOBSHOP_RATE_LIMIT_PER_HOUR"
DEFAULT_PER_MINUTE = 10
DEFAULT_PER_HOUR = 120


@dataclass(frozen=True)
class Limit:
    per_minute: int = DEFAULT_PER_MINUTE   # 0 = no limit in this window
    per_hour: int = DEFAULT_PER_HOUR

    def __post_init__(self) -> None:
        if self.per_minute < 0 or self.per_hour < 0:
            raise ValueError("rate limits cannot be negative (use 0 for no limit)")
        if self.per_minute and self.per_hour and self.per_hour < self.per_minute:
            raise ValueError(f"the hourly limit ({self.per_hour}) is below the per-minute limit ({self.per_minute}), so the minute limit could never be reached")


def limit_from_env(env: Mapping[str, str]) -> Limit:
    def number(name: str, default: int) -> int:
        raw = env.get(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"{name} must be a whole number (0 turns the limit off), got {raw!r}") from None

    return Limit(number(PER_MINUTE_VAR, DEFAULT_PER_MINUTE), number(PER_HOUR_VAR, DEFAULT_PER_HOUR))


class RateLimiter:
    def __init__(self, limit: Limit, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit = limit
        self._clock = clock
        self._starts: deque[float] = deque()
        self._lock = threading.Lock()

    def try_acquire(self) -> float:
        """Count a turn if the budget allows and return 0; otherwise return the seconds to wait (rounded up) and count nothing."""
        now = self._clock()
        with self._lock:
            while self._starts and now - self._starts[0] >= 3600:
                self._starts.popleft()
            wait = 0.0
            for window, allowed in ((60, self.limit.per_minute), (3600, self.limit.per_hour)):
                if not allowed:
                    continue
                recent = [t for t in self._starts if now - t < window]
                if len(recent) >= allowed:
                    wait = max(wait, window - (now - recent[0]))
            if wait > 0:
                return float(max(1, int(wait) + (wait % 1 > 0)))
            self._starts.append(now)
            return 0.0
