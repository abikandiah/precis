# Generation image — everything that calls a model or feeds web content to
# one (see docs/blueprint.md's Docker boundary). This is NOT the
# devcontainer (.devcontainer/Dockerfile, for day-to-day development) — it's
# the minimal runtime image a consumer runs generation in. create-known-file
# (an Open Library lookup) runs fine on a host, outside this image.
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

# The research cache lives on a volume mounted at /data, so a rerun reuses
# a book's search results instead of paying for them again. Consumers should
# mount a NAMED volume there — see README.md for the `docker run` shape.
#
# The chown below only takes effect for a named volume's first population;
# a host bind mount (`-v ./somedir:/data`) keeps the host directory's own
# ownership, so one not writable by uid 1000 fails writing the cache — use
# a named volume, or `chown 1000:1000` the host directory first.
ENV PRECIS_CACHE_DIR=/data/cache
RUN useradd --create-home --uid 1000 precis \
    && mkdir -p /data \
    && chown precis:precis /data
VOLUME /data
USER precis

ENTRYPOINT ["precis"]
CMD ["--help"]
