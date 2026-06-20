import json
import sqlite3
from datetime import datetime, timedelta, timezone

from db.schema import get_conn

_DEFAULT_PARAMS = {
    "review_min": "30",
    "review_max": "300",
    "positive_rate": "0.85",
    "weight_tags_a": "0.7",
    "weight_tags_b": "0.3",
}


# ── ユーザー ──────────────────────────────────────────────────────────────────

def upsert_user(app_user_id: str, steam_id: str | None = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO users (app_user_id, steam_id, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(app_user_id) DO UPDATE SET
                steam_id   = COALESCE(excluded.steam_id, steam_id),
                updated_at = CURRENT_TIMESTAMP
            """,
            (app_user_id, steam_id),
        )


def get_user(app_user_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE app_user_id = ?", (app_user_id,)
        ).fetchone()
    return dict(row) if row else None


# ── パラメータ ────────────────────────────────────────────────────────────────

def save_params(app_user_id: str, params: dict) -> None:
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO user_params (app_user_id, param_key, param_value, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(app_user_id, param_key) DO UPDATE SET
                param_value = excluded.param_value,
                updated_at  = CURRENT_TIMESTAMP
            """,
            [(app_user_id, k, str(v)) for k, v in params.items()],
        )


def load_params(app_user_id: str) -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT param_key, param_value FROM user_params WHERE app_user_id = ?",
            (app_user_id,),
        ).fetchall()
    merged = dict(_DEFAULT_PARAMS)
    merged.update({r["param_key"]: r["param_value"] for r in rows})
    return {
        "review_min":    int(merged["review_min"]),
        "review_max":    int(merged["review_max"]),
        "positive_rate": float(merged["positive_rate"]),
        "weight_tags_a": float(merged["weight_tags_a"]),
        "weight_tags_b": float(merged["weight_tags_b"]),
    }


# ── ナシリスト ────────────────────────────────────────────────────────────────

def add_nashi(app_user_id: str, appid: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO nashi_list (app_user_id, appid) VALUES (?, ?)",
            (app_user_id, appid),
        )


def get_nashi_list(app_user_id: str) -> list[int]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT appid FROM nashi_list WHERE app_user_id = ?", (app_user_id,)
        ).fetchall()
    return [r["appid"] for r in rows]


def count_nashi(app_user_id: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM nashi_list WHERE app_user_id = ?",
            (app_user_id,),
        ).fetchone()
    return row["cnt"]


# ── APIキャッシュ ──────────────────────────────────────────────────────────────

def get_cached(appid: int, data_type: str, ttl_days: int = 7) -> dict | None:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=ttl_days)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT data_json FROM api_cache
            WHERE appid = ? AND data_type = ? AND cached_at >= ?
            """,
            (appid, data_type, cutoff),
        ).fetchone()
    return json.loads(row["data_json"]) if row else None


def set_cache(appid: int, data_type: str, data: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO api_cache (appid, data_type, data_json, cached_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(appid, data_type) DO UPDATE SET
                data_json = excluded.data_json,
                cached_at = CURRENT_TIMESTAMP
            """,
            (appid, data_type, json.dumps(data, ensure_ascii=False)),
        )


# ── パイプラインジョブ ─────────────────────────────────────────────────────────

def update_job(
    app_user_id: str,
    status: str,
    phase: str = "",
    progress: float = 0.0,
    result_json: str | None = None,
    error_msg: str | None = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO pipeline_jobs
                (app_user_id, status, phase, progress, result_json, error_msg,
                 started_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(app_user_id) DO UPDATE SET
                status      = excluded.status,
                phase       = excluded.phase,
                progress    = excluded.progress,
                result_json = COALESCE(excluded.result_json, result_json),
                error_msg   = excluded.error_msg,
                started_at  = CASE WHEN excluded.status = 'running'
                                   THEN CURRENT_TIMESTAMP
                                   ELSE started_at END,
                updated_at  = CURRENT_TIMESTAMP
            """,
            (app_user_id, status, phase, progress, result_json, error_msg),
        )


def get_job(app_user_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM pipeline_jobs WHERE app_user_id = ?", (app_user_id,)
        ).fetchone()
    return dict(row) if row else None


def reset_job(app_user_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM pipeline_jobs WHERE app_user_id = ?", (app_user_id,)
        )
