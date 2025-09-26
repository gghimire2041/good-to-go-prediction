#!/usr/bin/env python3
"""
Export local SQLite database tables to CSV (and optionally Parquet if available).

Usage:
  python scripts/export_db.py --db data/g2g.db --out-dir exports --format both
"""

import argparse
from pathlib import Path
import sqlite3
import sys
import pandas as pd


def export_table(conn: sqlite3.Connection, table: str, out_dir: Path, fmt: str = "csv") -> None:
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt in ("csv", "both"):
        (out_dir / f"{table}.csv").write_text(df.to_csv(index=False))
    if fmt in ("parquet", "both"):
        try:
            df.to_parquet(out_dir / f"{table}.parquet", index=False)
        except Exception as e:
            print(f"[WARN] Parquet export failed for {table}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/g2g.db", help="Path to SQLite DB")
    ap.add_argument("--out-dir", default="exports", help="Output directory")
    ap.add_argument("--format", choices=["csv", "parquet", "both"], default="csv")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[ERROR] DB not found at {db_path}")
        sys.exit(1)

    with sqlite3.connect(str(db_path)) as conn:
        export_table(conn, "train_data", Path(args.out_dir), args.format)
        export_table(conn, "inference_logs", Path(args.out_dir), args.format)
    print(f"[OK] Exported tables to {args.out_dir}")


if __name__ == "__main__":
    main()

