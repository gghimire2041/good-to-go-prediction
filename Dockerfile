# G2G Model API Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

# Copy scripts
COPY scripts/ ./scripts/

# Copy static UI (for serving from API at /ui)
COPY pages/ ./pages/

# Create necessary directories
RUN mkdir -p data/raw data/processed logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run via entrypoint to validate artifacts first
CMD ["sh", "scripts/entrypoint.sh"]
