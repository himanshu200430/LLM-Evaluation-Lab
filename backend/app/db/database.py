import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "evallab.db"

@contextmanager

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
        conn.commit()

    finally:
        conn.close()

def init_db():
    with connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments(
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           created_at TEXT NOT NULL,
           dataset_name TEXT NOT NULL,
           mode TEXT NOT NULL,
           total_questions INTEGER NOT NULL        
        );

        CREATE TABLE IF NOT EXISTS results (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           experiment_id INTEGER NOT NULL REFERENCES experiments(id),
           model TEXT NOT NULL,
           question TEXT NOT NULL,
           expected_answer TEXT NOT NULL,
           answer TEXT,
           exact INTEGER,
           normalized INTEGER,
           correct INTEGER,
           latency_sec REAL,
           tokens_in INTEGER,
           tokens_out INTEGER,
           cost_usd REAL,
           error TEXT
        );
        CREATE TABLE IF NOT EXISTS judgements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id INTEGER NOT NULL REFERENCES results(id),
            judge_model TEXT NOT NULL,
            score INTEGER,
            correct INTEGER,
            reason TEXT,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS faithfulness (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id INTEGER NOT NULL REFERENCES results(id),
            judge_model TEXT NOT NULL,
            context TEXT,
            faithful INTEGER,
            reason TEXT,
            error TEXT
        );
        """)
        