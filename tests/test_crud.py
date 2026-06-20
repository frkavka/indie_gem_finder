"""TASK-4002: db/crud.py のユニットテスト（一時SQLiteファイルを使用）"""
import os
import tempfile

import pytest

import db.schema as schema
import db.crud as crud


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """テストごとに独立した一時DBを使う"""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setattr(schema, "DB_PATH", db_file)
    monkeypatch.setattr(crud, "get_conn", lambda: schema.get_conn())
    schema.init_db()
    yield


# ── ユーザー ──────────────────────────────────────────────────────────────────

def test_upsert_and_get_user():
    crud.upsert_user("uid-001", steam_id="12345678901234567")
    user = crud.get_user("uid-001")
    assert user is not None
    assert user["steam_id"] == "12345678901234567"


def test_upsert_user_updates_steam_id():
    crud.upsert_user("uid-001")
    crud.upsert_user("uid-001", steam_id="99999999999999999")
    assert crud.get_user("uid-001")["steam_id"] == "99999999999999999"


def test_get_user_returns_none_for_unknown():
    assert crud.get_user("nonexistent") is None


# ── パラメータ ────────────────────────────────────────────────────────────────

def test_load_params_returns_defaults_for_new_user():
    crud.upsert_user("uid-001")
    params = crud.load_params("uid-001")
    assert params["review_min"] == 30
    assert params["review_max"] == 300
    assert abs(params["positive_rate"] - 0.85) < 1e-6


def test_save_and_load_params():
    crud.upsert_user("uid-001")
    crud.save_params("uid-001", {"review_min": 50, "review_max": 500, "positive_rate": 0.9})
    params = crud.load_params("uid-001")
    assert params["review_min"] == 50
    assert params["review_max"] == 500
    assert abs(params["positive_rate"] - 0.9) < 1e-6


def test_save_params_partial_update():
    """一部のパラメータのみ更新しても他のデフォルト値が維持されること"""
    crud.upsert_user("uid-001")
    crud.save_params("uid-001", {"review_min": 100})
    params = crud.load_params("uid-001")
    assert params["review_min"] == 100
    assert params["review_max"] == 300  # デフォルト維持


# ── ナシリスト ────────────────────────────────────────────────────────────────

def test_add_and_count_nashi():
    crud.upsert_user("uid-001")
    crud.add_nashi("uid-001", 12345)
    crud.add_nashi("uid-001", 67890)
    assert crud.count_nashi("uid-001") == 2


def test_add_nashi_ignores_duplicate():
    crud.upsert_user("uid-001")
    crud.add_nashi("uid-001", 12345)
    crud.add_nashi("uid-001", 12345)  # 重複
    assert crud.count_nashi("uid-001") == 1


def test_get_nashi_list_returns_correct_appids():
    crud.upsert_user("uid-001")
    crud.add_nashi("uid-001", 111)
    crud.add_nashi("uid-001", 222)
    result = crud.get_nashi_list("uid-001")
    assert set(result) == {111, 222}


def test_nashi_is_user_scoped():
    """別ユーザーのナシリストは独立していること"""
    crud.upsert_user("uid-001")
    crud.upsert_user("uid-002")
    crud.add_nashi("uid-001", 12345)
    assert crud.count_nashi("uid-002") == 0


# ── APIキャッシュ ──────────────────────────────────────────────────────────────

def test_set_and_get_cache():
    crud.set_cache(12345, "steamspy_tags", {"tags": "Action,RPG"})
    result = crud.get_cached(12345, "steamspy_tags")
    assert result is not None
    assert result["tags"] == "Action,RPG"


def test_cache_returns_none_for_missing():
    assert crud.get_cached(99999, "steamspy_tags") is None


def test_cache_returns_none_after_ttl(monkeypatch):
    """TTL に負の値を指定すると cutoff が未来になり、期限切れとして None を返すこと"""
    crud.set_cache(12345, "steam_about", {"about": "test"})
    # ttl_days=-1 → cutoff = now + 1日（未来）→ cached_at < cutoff → None
    result = crud.get_cached(12345, "steam_about", ttl_days=-1)
    assert result is None


# ── パイプラインジョブ ─────────────────────────────────────────────────────────

def test_update_and_get_job():
    crud.upsert_user("uid-001")
    crud.update_job("uid-001", "running", "テスト中", 0.5)
    job = crud.get_job("uid-001")
    assert job["status"] == "running"
    assert abs(job["progress"] - 0.5) < 1e-6


def test_update_job_complete_with_result():
    crud.upsert_user("uid-001")
    crud.update_job("uid-001", "complete", "完了", 1.0, result_json='{"ranked_a":[]}')
    job = crud.get_job("uid-001")
    assert job["status"] == "complete"
    assert job["result_json"] == '{"ranked_a":[]}'


def test_reset_job():
    crud.upsert_user("uid-001")
    crud.update_job("uid-001", "complete", "完了", 1.0)
    crud.reset_job("uid-001")
    assert crud.get_job("uid-001") is None
