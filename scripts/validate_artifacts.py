#!/usr/bin/env python3
"""
Validate saved model artifacts (model + preprocessor + metadata) against the runtime environment.

Checks:
- Required files exist
- Loadability of preprocessor and model
- Feature dimension consistency (metadata vs preprocessor)
- Version compatibility (if versions stamped in metadata)

Exit codes:
- 0 on success (valid bundle)
- 1 on validation error
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure src is importable when run from project root or container
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from g2g_model.preprocessing.preprocessor import G2GPreprocessor  # noqa: E402
from g2g_model.models.catboost_model import G2GCatBoostModel  # noqa: E402


def load_metadata(model_path: Path) -> dict:
    meta_path = model_path.with_suffix('.json')
    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata JSON not found: {meta_path}")
    return json.loads(meta_path.read_text())


def validate(model_path: Path, preprocessor_path: Path) -> None:
    errors = []
    warnings = []

    if not model_path.exists():
        errors.append(f"Missing model file: {model_path}")
    if not preprocessor_path.exists():
        errors.append(f"Missing preprocessor file: {preprocessor_path}")
    if errors:
        raise RuntimeError("; ".join(errors))

    # Load metadata
    try:
        metadata = load_metadata(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model metadata: {e}")

    # Load preprocessor
    try:
        pre = G2GPreprocessor.load(str(preprocessor_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessor: {e}")

    # Load model (ensures CatBoost file is valid)
    try:
        _ = G2GCatBoostModel.load(str(model_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Check feature dimensions
    meta_features = metadata.get('feature_names') or []
    pre_features = getattr(pre, 'feature_names', None)
    if pre_features is None:
        # Derive from components
        text_names = getattr(pre, 'text_feature_names', []) or []
        pre_features = (
            text_names
            + pre.config.get('categorical_features', [])
            + pre.config.get('numerical_features', [])
        )

    if len(meta_features) != len(pre_features):
        errors.append(
            f"Feature count mismatch: metadata={len(meta_features)} vs preprocessor={len(pre_features)}"
        )

    # Check versions if available
    try:
        import numpy as _np
        import sklearn as _sk
        import catboost as _cb
    except Exception:
        _np = _sk = _cb = None

    meta_versions = metadata.get('versions', {})
    if meta_versions:
        curr = {
            'numpy': getattr(_np, '__version__', 'unknown') if _np else 'unknown',
            'sklearn': getattr(_sk, '__version__', 'unknown') if _sk else 'unknown',
            'catboost': getattr(_cb, '__version__', 'unknown') if _cb else 'unknown',
        }
        for lib, ver in meta_versions.items():
            if lib in curr and curr[lib] != ver:
                warnings.append(f"Version mismatch for {lib}: expected {ver}, found {curr[lib]}")

    if errors:
        raise RuntimeError("; ".join(errors))

    if warnings:
        print("[WARN] " + "; ".join(warnings))

    print("[OK] Artifact bundle is valid.")


def main():
    parser = argparse.ArgumentParser(description="Validate G2G model artifacts")
    parser.add_argument("--model", default="models/g2g_model.pkl", help="Path to model file (.pkl)")
    parser.add_argument("--preprocessor", default="models/preprocessor.pkl", help="Path to preprocessor file (.pkl)")
    args = parser.parse_args()

    try:
        validate(Path(args.model), Path(args.preprocessor))
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
