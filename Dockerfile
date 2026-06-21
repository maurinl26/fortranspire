# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
# Image runtime du serveur MCP = pipeline GPU + serveur MCP, sans JAX
# (le portage JAX n'est pas un cas d'usage du serveur HTTP).
COPY pyproject.toml .
COPY fortranspire/ ./fortranspire/
RUN pip install --no-cache-dir --prefix=/install -e ".[mcp]"

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Dépendances système minimales (grep pour search_code)
RUN apt-get update && apt-get install -y --no-install-recommends \
    grep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie les packages installés depuis le build stage
COPY --from=builder /install /usr/local

# Copie le code source de l'agent uniquement
COPY fortranspire/ ./fortranspire/

# Variables d'environnement par défaut (overridables via docker-compose)
ENV OLLAMA_BASE_URL=http://ollama:11434 \
    OLLAMA_MODEL=mistral-nemo:12b \
    LLM_TEMPERATURE=0.2 \
    AGENT_MAX_ITERATIONS=15 \
    AGENT_WORKSPACE=/workspace \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Port du serveur MCP (HTTP/SSE)
EXPOSE 8000

# Le workspace est monté en volume depuis l'hôte
VOLUME ["/workspace"]

# Lance le serveur MCP
CMD ["python", "-m", "fortranspire.server"]
