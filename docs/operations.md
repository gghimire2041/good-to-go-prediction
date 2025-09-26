# Operations

Logging:
- Configurable via `config/config.yaml` (level, path, format)
- API logs through stdlib logging

Health:
- `GET /health` for liveness/readiness
- Docker healthcheck pings `/health`

Monitoring:
- Add Prometheus metrics via `prometheus-client`
- Optionally run alongside Prometheus via `docker-compose` example

Model lifecycle:
- Store new artifacts under `models/`
- Hot-reload by replacing artifacts and restarting container(s)

