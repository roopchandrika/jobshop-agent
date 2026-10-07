"""Approval tokens: the only way a schedule can be committed.

The model must never be able to commit. Two layers guarantee that:

1. ``commit_schedule`` is not offered to the model at all (see the registry), and
2. even if something calls it, it needs a token that only the human-facing layer
   (the CLI prompt or the web Approve button) can mint, because only that layer holds
   the ``ApprovalAuthority``. The token never passes through the model's context.

A token is an HMAC-signed claim of "a human approved *this exact schedule* of *this draft*
against *this committed version*". It is single-use and expires, so a stolen or replayed
token is useless, and a draft that changes after approval cannot ride on an old approval.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable

from jobshop.core.models import Schedule


class ApprovalError(Exception):
    """The token is missing, forged, expired, already used, or for something else."""


def schedule_digest(schedule: Schedule) -> str:
    """Fingerprint of the assignments, so approval is bound to exactly what the human saw."""
    rows = sorted(
        (a.op_id, a.order_id, a.machine_id, a.start, a.end) for a in schedule.assignments
    )
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class ApprovalAuthority:
    def __init__(
        self,
        secret: bytes | None = None,
        ttl_s: float = 300.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        # A fresh random secret per process: tokens cannot outlive the session or be guessed.
        self._secret = secret or secrets.token_bytes(32)
        self._ttl_s = ttl_s
        self._clock = clock
        self._used: set[str] = set()

    def issue(self, *, draft_id: str, base_version: int, schedule_digest: str) -> str:
        """Mint a token. Only the human-facing layer should ever call this."""
        claim = {
            "d": draft_id,
            "v": base_version,
            "s": schedule_digest,
            "exp": self._clock() + self._ttl_s,
            "n": secrets.token_hex(8),
        }
        body = _b64(json.dumps(claim, sort_keys=True).encode())
        return f"{body}.{_b64(self._sign(body))}"

    def consume(
        self, token: str, *, draft_id: str, base_version: int, schedule_digest: str
    ) -> None:
        """Verify the token for this exact commit and mark it used. Raises ApprovalError."""
        try:
            body, signature = token.split(".")
            signature_ok = hmac.compare_digest(_unb64(signature), self._sign(body))
            claim = json.loads(_unb64(body)) if signature_ok else None
        except (ValueError, TypeError, AttributeError):
            raise ApprovalError("approval token is malformed") from None
        if claim is None:
            raise ApprovalError("approval token signature is invalid")
        if self._clock() > claim["exp"]:
            raise ApprovalError("approval token has expired; ask the human to approve again")
        if (claim["d"], claim["v"], claim["s"]) != (draft_id, base_version, schedule_digest):
            raise ApprovalError(
                "approval token was issued for a different draft, version or schedule"
            )
        if claim["n"] in self._used:
            raise ApprovalError("approval token has already been used")
        self._used.add(claim["n"])

    def _sign(self, body: str) -> bytes:
        return hmac.new(self._secret, body.encode(), hashlib.sha256).digest()
