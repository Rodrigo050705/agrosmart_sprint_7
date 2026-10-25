import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS inferences (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  date TEXT NOT NULL,
  farm_name TEXT NOT NULL,
  plot TEXT NOT NULL,
  input_path TEXT NOT NULL,
  input_sha256 TEXT NOT NULL UNIQUE,
  pred_label TEXT NOT NULL,
  pred_prob REAL NOT NULL,
  probs_json TEXT NOT NULL,
  annotated_path TEXT,
  notes TEXT
);
"""

REQUIRED_COLUMNS = {
    "date": "TEXT NOT NULL DEFAULT ''",
    "farm_name": "TEXT NOT NULL DEFAULT ''",
    "plot": "TEXT NOT NULL DEFAULT ''",
}


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(SCHEMA)
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(inferences);").fetchall()
    }
    for column_name, column_def in REQUIRED_COLUMNS.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE inferences ADD COLUMN {column_name} {column_def};")
    conn.commit()
    return conn

def has_hash(conn: sqlite3.Connection, sha: str) -> bool:
    cur = conn.execute("SELECT 1 FROM inferences WHERE input_sha256=? LIMIT 1;", (sha,))
    return cur.fetchone() is not None

def insert_result(
    conn: sqlite3.Connection,
    created_at: str,
    sampled_date: str,
    farm_name: str,
    plot: str,
    input_path: str,
    sha256: str,
    pred_label: str,
    pred_prob: float,
    probs_json: str,
    annotated_path: str | None,
    notes: str | None,
) -> None:
    conn.execute(
        """INSERT INTO inferences
           (created_at, date, farm_name, plot, input_path, input_sha256, pred_label, pred_prob, probs_json, annotated_path, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
        (
            created_at,
            sampled_date,
            farm_name,
            plot,
            input_path,
            sha256,
            pred_label,
            float(pred_prob),
            probs_json,
            annotated_path,
            notes,
        ),
    )
    conn.commit()
