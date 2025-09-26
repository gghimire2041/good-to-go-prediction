"""
Lightweight local database utilities using SQLite (no external deps).

Tables:
- train_data(id, gid, payload_json, target, created_at)
- inference_logs(id, gid, payload_json, prediction, confidence, explanation_summary, created_at)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def init_db(db_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS train_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gid TEXT,
                payload_json TEXT NOT NULL,
                target REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gid TEXT,
                payload_json TEXT NOT NULL,
                prediction REAL,
                confidence TEXT,
                explanation_summary TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def insert_train_cases(
    db_path: str | Path,
    cases: Iterable[Dict[str, Any]],
    targets: Optional[Iterable[Optional[float]]] = None,
) -> int:
    ts = datetime.utcnow().isoformat()
    db_path = Path(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        if targets is None:
            data = [(c.get("gid"), json.dumps(c), None, ts) for c in cases]
        else:
            data = [(c.get("gid"), json.dumps(c), t, ts) for c, t in zip(cases, targets)]
        cur.executemany(
            "INSERT INTO train_data(gid, payload_json, target, created_at) VALUES (?,?,?,?)",
            data,
        )
        conn.commit()
        return cur.rowcount


def fetch_training_dataframe(db_path: str | Path) -> pd.DataFrame:
    db_path = Path(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query("SELECT gid, payload_json, target FROM train_data", conn)
    if df.empty:
        return df
    payload_df = pd.json_normalize(df["payload_json"].apply(json.loads))
    out = pd.concat([df[["gid", "target"]], payload_df], axis=1)
    return out


def count_train_rows(db_path: str | Path) -> int:
    db_path = Path(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(1) FROM train_data")
        (n,) = cur.fetchone()
    return int(n)


def log_inference(
    db_path: str | Path,
    *,
    gid: Optional[str],
    payload: Dict[str, Any],
    prediction: Optional[float],
    confidence: Optional[str] = None,
    explanation_summary: Optional[str] = None,
) -> None:
    ts = datetime.utcnow().isoformat()
    db_path = Path(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO inference_logs(gid, payload_json, prediction, confidence, explanation_summary, created_at)
            VALUES (?,?,?,?,?,?)
            """,
            (gid, json.dumps(payload), prediction, confidence, explanation_summary, ts),
        )
        conn.commit()
