# Generation image — phase 3 only (see docs/blueprint.md's Docker boundary
# section: Docker exists specifically to contain AI, not every piece of
# tooling in this module's orbit). This is NOT the devcontainer
# (.devcontainer/Dockerfile, which is for day-to-day development tooling)
# — this is the minimal runtime image a consumer pulls and runs generation
# in. Phases 1-2 (ISBN lookup, structural preflight) run fine on a host,
# outside this image entirely.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependency layer cached separately from source so a source-only change
# doesn't invalidate the (much slower) dependency install.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Checkpoint DB must live on a volume mounted into the container, not its
# own ephemeral filesystem, or resume across container restarts doesn't
# work (docs/blueprint.md's Orchestration section). Consumers should mount
# a volume at /data — see README.md for the exact `docker run` shape.
ENV PRECIS_CHECKPOINT_DB_PATH=/data/checkpoints.sqlite
RUN useradd --create-home --uid 1000 precis \
    && mkdir -p /data \
    && chown precis:precis /data
VOLUME /data
USER precis

ENTRYPOINT ["precis"]
CMD ["--help"]
