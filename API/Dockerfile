# -------- Stage 1: Builder (for heavy deps like numpy, torch, sentence-transformers) --------
FROM python:3.10-slim AS builder

WORKDIR /usr/src/app

# Install build tools required for numpy / torch / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first for layer caching
COPY requirements.txt .

# Build wheels (cached in /wheels to reuse on rebuilds)
RUN pip install --upgrade pip
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# -------- Stage 2: Runtime (lightweight, clean, no dev tools) --------
FROM python:3.10-slim AS runtime

WORKDIR /usr/local/app

# Install runtime dependencies using wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy the application code AFTER dependencies to optimize caching
COPY . .

# Ensure non-root user for Cloud Run security
RUN useradd --create-home appuser
USER appuser

# Expose Cloud Run default port
EXPOSE 8080

# Cloud Run expects Gunicorn-style entrypoint but Uvicorn alone is acceptable for light workloads
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
