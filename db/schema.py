import os
import sqlite3

DB_PATH = os.getenv("DB_PATH", "app.db")

_DDL = """
CREATE TABLE IF NOT EXISTS users (
    app_user_id TEXT PRIMARY KEY,
    steam_id    TEXT,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_params (
    app_user_id TEXT NOT NULL REFERENCES users(app_user_id) ON DELETE CASCADE,
    param_key   TEXT NOT NULL,
    param_value TEXT NOT NULL,
    updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (app_user_id, param_key)
);

CREATE TABLE IF NOT EXISTS nashi_list (
    app_user_id TEXT    NOT NULL REFERENCES users(app_user_id) ON DELETE CASCADE,
    appid       INTEGER NOT NULL,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (app_user_id, appid)
);

CREATE TABLE IF NOT EXISTS api_cache (
    appid       INTEGER NOT NULL,
    data_type   TEXT    NOT NULL,
    data_json   TEXT    NOT NULL,
    cached_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (appid, data_type)
);

CREATE TABLE IF NOT EXISTS pipeline_jobs (
    app_user_id TEXT    NOT NULL REFERENCES users(app_user_id) ON DELETE CASCADE,
    status      TEXT    NOT NULL DEFAULT 'idle',
    phase       TEXT,
    progress    REAL    NOT NULL DEFAULT 0.0,
    result_json TEXT,
    error_msg   TEXT,
    started_at  DATETIME,
    updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (app_user_id)
);

CREATE INDEX IF NOT EXISTS idx_api_cache_cached_at ON api_cache(cached_at);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(_DDL)
