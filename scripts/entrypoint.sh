#!/usr/bin/env sh
set -e

echo "[ENTRYPOINT] Validating artifacts..."
if ! python scripts/validate_artifacts.py; then
  STRICT=$(echo "${VALIDATE_STRICT:-false}" | tr '[:upper:]' '[:lower:]')
  if [ "$STRICT" = "1" ] || [ "$STRICT" = "true" ] || [ "$STRICT" = "yes" ]; then
    echo "[ENTRYPOINT] Validation failed and VALIDATE_STRICT is enabled. Exiting." >&2
    exit 1
  else
    echo "[ENTRYPOINT] Validation failed. Continuing with API startup (the API may train a demo model if needed)." >&2
  fi
fi

echo "[ENTRYPOINT] Starting API..."
exec uvicorn src.g2g_model.api.main:app --host 0.0.0.0 --port 8000
