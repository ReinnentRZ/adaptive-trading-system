FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Install build dependencies, curl, ca-certificates, libgomp1, compile TA-Lib C-library
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    wget \
    ca-certificates \
    curl \
    libgomp1 \
    && update-ca-certificates \
    && (curl -fsSL --retry 3 "https://github.com/ta-lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz" -o /tmp/ta-lib-0.4.0-src.tar.gz || \
        curl -fsSL --retry 3 "https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz/download" -o /tmp/ta-lib-0.4.0-src.tar.gz) \
    && tar -xzf /tmp/ta-lib-0.4.0-src.tar.gz -C /tmp \
    && cd /tmp/ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && rm -rf /tmp/* \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages (builds TA-Lib python wrapper)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create a non-root user and prepare workspace directories
RUN useradd -u 1000 -m appuser \
    && mkdir -p /app/data /app/logs /app/models \
    && chown -R appuser:appuser /app \
    && chmod -R 775 /app/data /app/logs

USER appuser

# Copy application configuration, source code, models, and test suite
COPY --chown=appuser:appuser pyproject.toml main.py ./
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser scripts/ scripts/
COPY --chown=appuser:appuser models/ models/
COPY --chown=appuser:appuser tests/ tests/

CMD ["python3", "src/execution/live_bot.py"]
