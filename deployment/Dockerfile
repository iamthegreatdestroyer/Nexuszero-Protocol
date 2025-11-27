FROM python:3.11-slim-bookworm as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements*.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim-bookworm as runtime

RUN groupadd -r nexuszero && useradd -r -g nexuszero nexuszero

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=nexuszero:nexuszero src/ ./src/
COPY --chown=nexuszero:nexuszero nexuszero-optimizer/ ./nexuszero-optimizer/
COPY --chown=nexuszero:nexuszero config/ ./config/

RUN mkdir -p /app/data /app/logs && chown -R nexuszero:nexuszero /app

ENV PYTHONPATH=/app/src:/app/nexuszero-optimizer/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

USER nexuszero
EXPOSE 8080 8443

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
