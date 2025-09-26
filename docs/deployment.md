# Deployment

Docker:
- Build: `docker build -t g2g-model:latest .`
- Run: `docker run -p 8000:8000 g2g-model:latest`

Compose:
- `docker-compose up --build`

Environment:
- `PYTHONPATH=/app/src` is set in the image
- Artifacts loaded from `/app/models` if present

Production tips:
- Run multiple workers (e.g., gunicorn with uvicorn workers) if under high load
- Add auth/rate limiting via FastAPI middleware or proxy
- Add Prometheus metrics via `prometheus-client`

Kubernetes (optional):
- See `deploy/k8s/` for `Deployment` and `Service` manifests

