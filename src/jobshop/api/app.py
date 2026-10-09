"""The web API: chat with the agent, look at the plan, approve or reject a proposal.

One planner, one shop, one conversation (multi-user and auth are stated non-goals), so it is
meant to run on the planner's own machine and refuses to listen anywhere else (see ``server``).

The Approve button is the web version of the CLI's "[y/N]". That makes ``/api/approve`` the most
sensitive endpoint in the system, so it is protected the way a bank transfer form would be:

* a per-run CSRF token, readable only by the page this server served (cross-site pages cannot
  read it), required on every state-changing request;
* an ``Origin`` check, and ``Host`` validation against DNS rebinding;
* the page sends the fingerprint of the proposal it displayed, and the server refuses if the
  draft no longer matches, so a person can only approve what they were shown;
* the model has no way to call it: it can only use the tools in the registry, none of which is this.
"""

from __future__ import annotations

import hmac
import json
import logging
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from jobshop.agent.conversation import Conversation
from jobshop.agent.loop import AgentConfig, TurnResult
from jobshop.agent.memory import PreferenceError, PreferenceStore
from jobshop.agent.trace import Tracer
from jobshop.api import views as web
from jobshop.api.ratelimit import Limit, RateLimiter
from jobshop.core.kpis import compute_kpis
from jobshop.tools import views
from jobshop.tools.errors import ToolError
from jobshop.tools.functions import DraftId, Source, ToolContext
from jobshop.tools.human import commit_draft

log = logging.getLogger("jobshop.api")
STATIC = Path(__file__).parent / "static"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")

CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatIn(_In):
    message: str = Field(min_length=1, max_length=2000)


class ApproveIn(_In):
    draft_id: DraftId
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")  # fingerprint of the proposal the person was shown


class RejectIn(_In):
    draft_id: DraftId


class PreferenceIn(_In):
    text: str = Field(min_length=1, max_length=1000)   # the store enforces the real (shorter) limit and says why


class ForgetIn(_In):
    id: int = Field(strict=True, ge=1, le=10**9)   # a real number: not "1", not true, not 1.0


HEARTBEAT_SECONDS = 15.0   # how often an idle event stream sends a comment, so proxies and browsers keep the connection


@dataclass
class Turn:
    id: int
    status: str = "running"  # running | done
    progress: list[str] = field(default_factory=list)  # tool names as they run, for the UI's activity line
    error: str | None = None
    # Anyone waiting for news about this turn (the event stream) sleeps on this and is woken by every change.
    changed: threading.Condition = field(default_factory=threading.Condition, repr=False, compare=False)

    def add_progress(self, label: str) -> None:
        with self.changed:
            self.progress.append(label)
            self.changed.notify_all()

    def finish(self) -> None:
        with self.changed:
            self.status = "done"
            self.changed.notify_all()


class WebState:
    """Everything the API remembers between requests. ``lock`` serialises all use of the store."""

    def __init__(
        self, ctx: ToolContext, client: Any, config: AgentConfig, tracer_path: Path | None,
        preferences: PreferenceStore | None = None,
    ) -> None:
        self.ctx, self.config = ctx, config
        self.preferences = preferences if preferences is not None else PreferenceStore()
        self.lock = threading.Lock()
        self.transcript: list[dict[str, Any]] = []
        self.turn: Turn | None = None
        self._turn_counter = 0
        self.proposal_draft: str | None = None  # the draft the last answer asked the planner to approve
        self.conversation = (
            Conversation(client, ctx, config, Tracer(path=tracer_path, echo=self._on_event), self.preferences)
            if client is not None else None
        )

    def _on_event(self, record: dict[str, Any]) -> None:
        turn = self.turn
        if turn is not None and turn.status == "running" and record["event"] == "tool_call":
            turn.add_progress(record["tool"] + (" (failed)" if record["is_error"] else ""))

    def start_turn(self) -> Turn:
        self._turn_counter += 1
        self.turn = Turn(self._turn_counter)
        return self.turn

    def record_answer(self, result: TurnResult) -> None:
        final = result.final
        usage = {"steps": result.steps, "tokens": result.total_tokens, "cost_usd": result.cost_usd}
        if final is None:
            self.transcript.append({"role": "assistant", "kind": "stopped", "text": result.text or "No answer.", "usage": usage})
            self.proposal_draft = None
            return
        self.transcript.append({
            "role": "assistant", "kind": "clarify" if final.clarifying_question else "answer",
            "text": final.clarifying_question or final.summary, "warnings": final.warnings,
            "draft_id": final.draft_id, "usage": usage,
        })
        self.proposal_draft = final.draft_id if final.needs_approval else None


def create_app(
    ctx: ToolContext,
    client: Any,
    config: AgentConfig,
    *,
    tracer_path: Path | None = None,
    allowed_hosts: tuple[str, ...] = LOOPBACK_HOSTS,
    preferences: PreferenceStore | None = None,
    rate_limit: Limit | None = None,
) -> FastAPI:
    state = WebState(ctx, client, config, tracer_path, preferences)
    limiter = RateLimiter(rate_limit if rate_limit is not None else Limit(0, 0))   # off unless the server asks (see api/server.py)
    csrf_token = secrets.token_urlsafe(32)
    app = FastAPI(title="Job-shop scheduling assistant", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.web = state
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse({"detail": "internal error"}, status_code=500)  # never a traceback

    def guard(request: Request, x_csrf_token: Annotated[str | None, Header()] = None) -> None:
        if not x_csrf_token or not hmac.compare_digest(x_csrf_token, csrf_token):
            raise HTTPException(403, "missing or wrong CSRF token; reload the page")
        origin = request.headers.get("origin")
        if origin is not None and urlparse(origin).hostname not in allowed_hosts:
            raise HTTPException(403, "cross-origin request refused")

    def tool_error(e: ToolError) -> HTTPException:
        return HTTPException(404 if "unknown draft" in str(e) else 409, str(e))

    # -- pages --------------------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8").replace("{{CSRF_TOKEN}}", csrf_token)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """For a container orchestrator or a load balancer: is the process up and able to answer? Takes no lock, shows no data."""
        return {"status": "ok", "model_configured": state.conversation is not None}

    # -- reading ------------------------------------------------------------------------------------

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        base = {
            "model": config.model if state.conversation else None,
            "model_configured": state.conversation is not None,
            "transcript": list(state.transcript),
            "preferences": [{"id": p.id, "text": p.text} for p in state.preferences.list()],
        }
        if not state.lock.acquire(timeout=0.25):  # the agent is mid-turn and owns the store
            return {**base, "busy": True, "progress": list(state.turn.progress) if state.turn else []}
        try:
            committed = ctx.store.committed
            live = views.kpi_view(committed.instance, compute_kpis(committed.instance, committed.schedule), include_orders=False)
            return {
                **base, "busy": False,
                "plant_time": views.fmt(committed.instance, committed.instance.now),
                "live_version": committed.version,
                "kpi_live": live.model_dump(mode="json"),
                "drafts": web.draft_summaries(ctx),
                "proposal": web.proposal(ctx, state.proposal_draft),
            }
        finally:
            state.lock.release()

    @app.get("/api/gantt")
    def get_gantt(source: Source = "committed") -> dict[str, Any]:
        if not state.lock.acquire(timeout=0.25):
            raise HTTPException(409, "the agent is working; try again in a moment")
        try:
            return web.gantt(ctx, source).model_dump(mode="json")
        except ToolError as e:
            raise tool_error(e) from None
        finally:
            state.lock.release()

    # -- the conversation ----------------------------------------------------------------------------

    @app.post("/api/chat", status_code=202, dependencies=[Depends(guard)])
    def post_chat(body: ChatIn) -> dict[str, Any]:
        if state.conversation is None:
            raise HTTPException(503, "No model is configured. Set ANTHROPIC_API_KEY and ANTHROPIC_MODEL, then restart.")
        if not state.lock.acquire(blocking=False):
            raise HTTPException(409, "the assistant is still working on the previous message")
        wait = limiter.try_acquire()
        if wait:
            state.lock.release()
            raise HTTPException(429, f"Too many requests to the assistant (the limit protects the model budget). Try again in {int(wait)} s.",
                                headers={"Retry-After": str(int(wait))})
        turn = state.start_turn()
        state.transcript.append({"role": "user", "text": body.message})

        def work() -> None:
            try:
                state.record_answer(state.conversation.say(body.message))  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001  # a bug must end the turn, not leave the UI waiting forever
                log.exception("turn failed")
                turn.error = "Something went wrong running that request. See the server log."
                state.transcript.append({"role": "assistant", "kind": "stopped", "text": turn.error})
            finally:
                turn.finish()
                state.lock.release()

        threading.Thread(target=work, daemon=True, name=f"turn-{turn.id}").start()
        return {"turn_id": turn.id}

    @app.get("/api/chat/{turn_id}")
    def get_turn(turn_id: int) -> dict[str, Any]:
        turn = state.turn
        if turn is None or turn.id != turn_id:
            raise HTTPException(404, "unknown turn")
        return {"turn_id": turn.id, "status": turn.status, "progress": list(turn.progress), "error": turn.error}

    # -- standing preferences (long-term memory): the planner writes, the model only reads ----------------------

    def preference_list() -> dict[str, Any]:
        return {"preferences": [{"id": p.id, "text": p.text} for p in state.preferences.list()]}

    @app.post("/api/preferences", dependencies=[Depends(guard)])
    def add_preference(body: PreferenceIn) -> dict[str, Any]:
        try:
            state.preferences.add(body.text)
        except PreferenceError as e:
            raise HTTPException(422, str(e)) from None
        return preference_list()

    @app.post("/api/preferences/forget", dependencies=[Depends(guard)])
    def forget_preference(body: ForgetIn) -> dict[str, Any]:
        try:
            state.preferences.remove(body.id)
        except PreferenceError as e:
            raise HTTPException(404, str(e)) from None
        return preference_list()

    @app.get("/api/chat/{turn_id}/events")
    def turn_events(turn_id: int, last_event_id: Annotated[str | None, Header()] = None) -> StreamingResponse:
        """Server-Sent Events for one turn: a ``progress`` event as each tool runs, then ``done``.

        Replaces polling /api/chat/{id}. Each progress event carries an ``id``; a browser that reconnects sends
        ``Last-Event-ID`` and gets only what it missed. Read-only, so no CSRF token is needed (it changes nothing).
        """
        turn = state.turn
        if turn is None or turn.id != turn_id:
            raise HTTPException(404, "unknown turn")
        try:
            sent = max(0, int(last_event_id)) if last_event_id else 0
        except ValueError:
            sent = 0

        def stream():
            nonlocal sent
            while True:
                with turn.changed:
                    idle = turn.changed.wait_for(lambda: len(turn.progress) > sent or turn.status == "done", timeout=HEARTBEAT_SECONDS)
                    batch, done, error = turn.progress[sent:], turn.status == "done", turn.error
                for label in batch:
                    sent += 1
                    yield f"id: {sent}\nevent: progress\ndata: {json.dumps({'tool': label, 'index': sent})}\n\n"
                if done:
                    yield f"event: done\ndata: {json.dumps({'status': 'done', 'error': error})}\n\n"
                    return
                if not idle:
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    # -- the human decision ----------------------------------------------------------------------------

    @app.post("/api/approve", dependencies=[Depends(guard)])
    def approve(body: ApproveIn) -> dict[str, Any]:
        if not state.lock.acquire(blocking=False):
            raise HTTPException(409, "the assistant is working; wait for it to finish before approving")
        try:
            result = commit_draft(ctx, body.draft_id, reviewed_digest=body.digest)
            note = f"The planner approved and committed draft {body.draft_id}; the live plan is now version {result['new_version']}."
            state.transcript.append({"role": "system", "text": f"You approved draft {body.draft_id}. The live plan is now version {result['new_version']}."})
            if state.conversation:
                state.conversation.notify(note)
            state.proposal_draft = None
            return {"new_version": result["new_version"]}
        except ToolError as e:
            raise tool_error(e) from None
        finally:
            state.lock.release()

    @app.post("/api/reject", dependencies=[Depends(guard)])
    def reject(body: RejectIn) -> dict[str, str]:
        if not state.lock.acquire(blocking=False):
            raise HTTPException(409, "the assistant is working; wait for it to finish")
        try:
            ctx.store.discard(body.draft_id)
            state.transcript.append({"role": "system", "text": f"You rejected draft {body.draft_id}. The live plan is unchanged."})
            if state.conversation:
                state.conversation.notify(f"The planner REJECTED draft {body.draft_id} and it was discarded; the live plan is unchanged.")
            if state.proposal_draft == body.draft_id:
                state.proposal_draft = None
            return {"status": "rejected"}
        except ToolError as e:
            raise tool_error(e) from None
        finally:
            state.lock.release()

    return app
