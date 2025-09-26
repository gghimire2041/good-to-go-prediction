#!/usr/bin/env python3
"""
Daily batch monitoring:
- Generate 500 synthetic cases
- Run model inference locally (using saved artifacts if available; else demo path)
- Compute data drift (Evidently) vs reference (use previous training data distribution or a cached reference)
- Save results and reports under data/monitoring/<date>
- Exit non-zero if drift exceeds threshold
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from g2g_model.data.data_generator import G2GDataGenerator
from g2g_model.preprocessing.preprocessor import G2GPreprocessor
from g2g_model.models.catboost_model import G2GCatBoostModel
from g2g_model.storage import db as g2gdb

from evidently.report import Report
from evidently.metrics import DataDriftPreset


def load_artifacts(model_path: Path, preprocessor_path: Path):
    model = G2GCatBoostModel.load(str(model_path))
    pre = G2GPreprocessor.load(str(preprocessor_path))
    return model, pre


def get_reference_df(n: int = 1000) -> pd.DataFrame:
    # Generate a reference dataset to compare against (synthetic)
    gen = G2GDataGenerator(random_state=42)
    return gen.generate_data(n_samples=n)


def main():
    out_dir = ROOT / "data" / "monitoring" / datetime.utcnow().strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate daily batch
    gen = G2GDataGenerator(random_state=int(datetime.utcnow().strftime("%Y%m%d")))
    df_new = gen.generate_data(n_samples=int(os.getenv("DAILY_SAMPLES", 5)))
    (out_dir / "batch_raw.csv").write_text(df_new.to_csv(index=False))

    # Also add 5 new cases to local DB and run inference on them
    db_path = os.getenv("G2G_DB_PATH", str(ROOT / "data" / "g2g.db"))
    g2gdb.init_db(db_path)
    df_five = gen.generate_data(n_samples=5)
    cases5 = df_five.drop(columns=["g2g_score"]).to_dict(orient="records")
    targets5 = df_five["g2g_score"].tolist()
    g2gdb.insert_train_cases(db_path, cases5, targets5)

    # Load artifacts
    model_path = ROOT / "models" / "g2g_model.pkl"
    pre_path = ROOT / "models" / "preprocessor.pkl"
    if not (model_path.exists() and pre_path.exists()):
        print("[WARN] Artifacts missing; exiting without drift checks.")
        sys.exit(0)

    model, pre = load_artifacts(model_path, pre_path)

    # Prepare features
    X_new, feature_names = pre.transform(df_new)
    preds = model.predict(X_new)
    df_new_out = df_new.copy()
    df_new_out["g2g_pred"] = preds
    (out_dir / "batch_predictions.csv").write_text(df_new_out.to_csv(index=False))

    # Inference and log the 5 new DB-added rows
    X5, _ = pre.transform(df_five)
    p5 = model.predict(X5)
    for i, row in enumerate(cases5):
        g2gdb.log_inference(
            db_path,
            gid=row.get("gid"),
            payload=row,
            prediction=float(p5[i]),
            confidence=("High" if p5[i] > 0.7 else "Medium" if p5[i] > 0.4 else "Low"),
            explanation_summary=None,
        )

    # Build reference
    df_ref = get_reference_df(n=1000)
    ref_small = df_ref[pre.config['categorical_features'] + pre.config['numerical_features']].copy()
    new_small = df_new[pre.config['categorical_features'] + pre.config['numerical_features']].copy()

    # Evidently data drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_small, current_data=new_small)
    report_path = out_dir / "data_drift_report.html"
    report.save_html(str(report_path))

    summary = report.as_dict()
    (out_dir / "data_drift_summary.json").write_text(json.dumps(summary, indent=2))

    # Compress today's artifacts (optional)
    try:
        if (os.getenv("COMPRESS_REPORT", "1").lower() in {"1", "true", "yes"}):
            base_name = str(out_dir)
            shutil.make_archive(base_name, "gztar", root_dir=str(out_dir))
    except Exception as e:
        print(f"[WARN] Failed to compress report: {e}")

    # Retention cleanup (delete older folders/archives)
    try:
        retain_days = int(os.getenv("RETAIN_DAYS", "30"))
        cutoff = datetime.utcnow() - timedelta(days=retain_days)
        mon_root = ROOT / "data" / "monitoring"
        for child in mon_root.iterdir():
            try:
                name = child.name
                # Handle both folders (YYYY-MM-DD) and archives (YYYY-MM-DD.tar.gz)
                date_str = name.split(".")[0]
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                continue
            if dt < cutoff:
                try:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                    print(f"[INFO] Removed old artifact: {child}")
                except Exception as e:
                    print(f"[WARN] Failed to remove {child}: {e}")
    except Exception as e:
        print(f"[WARN] Retention cleanup failed: {e}")

    # Simple threshold check
    # Look for 'dataset_drift' flag if present
    drift_flag = False
    try:
        drift_flag = bool(summary["metrics"][0]["result"].get("dataset_drift", False))
    except Exception:
        pass

    print(f"[INFO] Data drift detected: {drift_flag}")
    # Exit with code 2 on drift to trigger alerting in CI
    if drift_flag:
        sys.exit(2)


if __name__ == "__main__":
    main()
