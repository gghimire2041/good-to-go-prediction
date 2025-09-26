# Testing

We use `pytest` with a lightweight suite:
- Preprocessor: shape and consistency checks
- Model: quick fit with few iterations; prediction bounds
- API: uses a patched lightweight loader to avoid heavy training

Run:
- `pytest` (quick)
- `pytest -m slow` (if slow tests are added later)

Coverage:
- Enable with `pytest --cov` (requires pytest-cov)

