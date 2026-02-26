# qortex REST API server
# Bakes in the embedding model and spaCy model so cold starts are instant
# and HF_HUB_OFFLINE=1 always works.

FROM python:3.12-slim AS base

# System deps for sqlite-vec, numpy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy the entire monorepo (sub-packages + main package)
COPY packages/ packages/
COPY src/ src/
COPY pyproject.toml .
COPY README.md .

# Install qortex with all sandbox-relevant extras
# server: starlette + uvicorn
# vec-sqlite: default vector backend
# vec-pgvector: postgres vector backend (when available)
# observability: OTEL tracing
# nlp: spaCy NLP pipeline
# memgraph: neo4j driver for graph backend
# Pin numpy<2 first — thinc binary wheels are compiled against numpy 1.x ABI.
# torch and sentence-transformers work fine with numpy 1.x.
RUN uv pip install --system "numpy<2"

# sqlite-vec 0.1.6 ships broken 32-bit ARM binary — need prerelease >=0.1.7a2
RUN uv pip install --system --prerelease=allow "sqlite-vec>=0.1.7a2"

RUN uv pip install --system \
    "./packages/qortex-observe[otel]" \
    "./packages/qortex-ingest" \
    "./packages/qortex-online[nlp]" \
    "./packages/qortex-learning" \
    ".[server,vec-sqlite,vec-pgvector,observability,nlp,memgraph]"

# Verify sqlite-vec loads correctly (catches arch mismatch at build time)
RUN python -c "\
import sqlite3, sqlite_vec; \
conn = sqlite3.connect(':memory:'); \
conn.enable_load_extension(True); \
sqlite_vec.load(conn); \
print(f'sqlite-vec {sqlite_vec.__version__} OK')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Pre-download the embedding model so it's baked into the image.
# This eliminates the vec.unavailable failure that plagues systemd services.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('all-MiniLM-L6-v2'); \
print(f'Model cached: {m.get_sentence_embedding_dimension()} dims')"

# Runtime config
ENV HF_HUB_OFFLINE=1
ENV QORTEX_VEC=sqlite
EXPOSE 8400

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8400/v1/health')" || exit 1

CMD ["qortex", "serve", "--host", "0.0.0.0", "--port", "8400"]
