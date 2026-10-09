# The web app in a container. Build and run (the port is published to loopback only, because the app has no login):
#
#   docker build -t jobshop-agent .
#   docker run --rm -p 127.0.0.1:8000:8000 -e ANTHROPIC_API_KEY=... -e ANTHROPIC_MODEL=... jobshop-agent
#
# Without a key the page still shows the plan and chat is disabled. Preferences are kept in the container unless a
# volume is mounted at /home/app/.jobshop.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first, so editing the code does not reinstall them. The project files are copied right after because
# the build backend needs them to exist.
COPY pyproject.toml uv.lock LICENSE ./
RUN uv sync --locked --no-install-project --no-dev

COPY src ./src
COPY evals ./evals
COPY knowledge ./knowledge
RUN uv sync --locked --no-dev

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"

CMD ["python", "-m", "jobshop.api", "--host", "0.0.0.0", "--container"]
