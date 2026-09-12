FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install build dependencies, curl, ca-certificates, compile TA-Lib C-library, and update CA certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    curl \
    && update-ca-certificates \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf /tmp/ta-lib-0.4.0-src.tar.gz -C /tmp \
    && cd /tmp/ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && rm -rf /tmp/*

WORKDIR /app

# Copy requirements and install python packages (builds TA-Lib python wrapper)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Remove compilation build tools (KEEP ca-certificates and curl for runtime connection stability)
RUN apt-get purge -y --auto-remove build-essential wget \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and prepare workspace permissions
RUN useradd -u 1000 -m appuser \
    && mkdir -p /app/logs /app/models \
    && chown -R appuser:appuser /app \
    && chmod -R 775 /app/logs

USER appuser

# Copy application source code and trained models
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser models/ models/
COPY --chown=appuser:appuser main.py .

CMD ["python", "main.py"]
