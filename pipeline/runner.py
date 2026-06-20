import json
import logging
import threading

import numpy as np
import pandas as pd

from db.crud import get_job, get_nashi_list, update_job
from pipeline import bert_encoder, fetcher, matcher, projector, vectorizer

logger = logging.getLogger(__name__)

_DATA_CSV = "data/games.csv"


# ── シリアライズ ──────────────────────────────────────────────────────────────

def _serialize_result(
    pool: pd.DataFrame,
    user_vec_tags: np.ndarray,
    user_vec_about: np.ndarray,
) -> str:
    """候補プール（sim_tags・sim_about 付き）とユーザーベクトルを保存する。
    デッキ側がスライダー値でリアルタイム再ソートする。
    """
    export_cols = [
        "appid", "name", "sim_tags", "sim_about",
        "rating_ratio", "tags", "movies", "total_reviews",
    ]
    cols = [c for c in export_cols if c in pool.columns]
    return json.dumps(
        {
            "pool": pool[cols].to_dict(orient="records"),
            "user_vecs": {
                "tags": user_vec_tags.tolist(),
                "about": user_vec_about.tolist(),
            },
        },
        ensure_ascii=False,
    )


def deserialize_result(result_json: str) -> dict:
    data = json.loads(result_json)
    data["user_vecs"]["tags"] = np.array(data["user_vecs"]["tags"])
    data["user_vecs"]["about"] = np.array(data["user_vecs"]["about"])
    return data


# ── パイプライン本体 ──────────────────────────────────────────────────────────

def _run_pipeline(app_user_id: str, params: dict, api_key: str) -> None:
    try:
        # Phase 1: Steam ライブラリ + ウィッシュリスト取得
        update_job(app_user_id, "running", "Steamライブラリ取得中...", 0.05)
        seed_games = fetcher.fetch_seed_games(params["steam_id"], api_key)
        if not seed_games:
            raise ValueError("プレイ履歴もウィッシュリストも見つかりませんでした。")
        seed_appids = [g["appid"] for g in seed_games]
        seed_weights = {g["appid"]: g["weight_score"] for g in seed_games}
        update_job(app_user_id, "running", "Steamライブラリ取得完了", 0.2)

        # Phase 2: SteamSpy + Steam Store データ並列取得
        update_job(app_user_id, "running", "ゲームデータ取得中（APIキャッシュ利用）...", 0.25)
        seed_data = fetcher.fetch_seed_data_parallel(seed_appids, max_workers=3)
        update_job(app_user_id, "running", "APIデータ取得完了", 0.5)

        # Phase 3: TF-IDF ユーザーベクトル
        update_job(app_user_id, "running", "タグベクトル構築中...", 0.55)
        hidden_df, tfidf_vectorizer, tfidf_matrix = vectorizer.load_hidden_gems(
            _DATA_CSV,
            review_min=params["review_min"],
            review_max=params["review_max"],
            positive_rate=params["positive_rate"],
        )
        csv_tags = vectorizer.load_all_tags(_DATA_CSV)
        user_vec_tags = vectorizer.build_user_vector_tags(
            seed_data, tfidf_vectorizer, seed_weights, csv_tags=csv_tags
        )

        # Phase 4: BERT ユーザーベクトル + 原石エンコード
        update_job(app_user_id, "running", "BERTエンコード中...", 0.65)
        model = bert_encoder.load_bert_model()
        hidden_df["about_clean"] = hidden_df["about_the_game"].fillna("").apply(
            lambda x: __import__("re").sub(r"<[^>]+>", " ", str(x)).strip()
        )
        embeddings_about = bert_encoder.encode_hidden_gems(hidden_df["about_clean"].tolist(), model)
        user_vec_about = bert_encoder.build_user_vector_about(seed_data, seed_weights, model)
        update_job(app_user_id, "running", "ベクトル構築完了", 0.8)

        # Phase 5: 候補プール構築（weight は適用しない — デッキがリアルタイム再ソート）
        update_job(app_user_id, "running", "マッチング計算中...", 0.85)
        nashi_appids = get_nashi_list(app_user_id)
        pool = matcher.build_candidate_pool(
            user_vec_tags, user_vec_about, hidden_df, tfidf_matrix, embeddings_about,
            seed_appids, nashi_appids, pool_size=200,
        )

        result_json = _serialize_result(pool, user_vec_tags, user_vec_about)
        update_job(app_user_id, "complete", "完了", 1.0, result_json=result_json)

    except Exception as e:
        logger.exception("パイプラインエラー: %s", e)
        update_job(app_user_id, "error", "エラー", 0.0, error_msg=str(e))


def _run_recompute(app_user_id: str, params: dict, api_key: str) -> None:
    try:
        update_job(app_user_id, "running", "直交射影による再計算中...", 0.1)
        job = get_job(app_user_id)
        if not job or not job["result_json"]:
            raise ValueError("前回の計算結果が見つかりません。最初から実行してください。")

        prev = deserialize_result(job["result_json"])
        user_vec_tags = prev["user_vecs"]["tags"]
        user_vec_about = prev["user_vecs"]["about"]

        nashi_appids = get_nashi_list(app_user_id)

        # ナシゲームのベクトルを取得して直交射影
        update_job(app_user_id, "running", "ナシゲームのベクトル取得中...", 0.2)
        nashi_seed_data = fetcher.fetch_seed_data_parallel(nashi_appids, max_workers=3)

        hidden_df, tfidf_vectorizer, tfidf_matrix = vectorizer.load_hidden_gems(
            _DATA_CSV,
            review_min=params["review_min"],
            review_max=params["review_max"],
            positive_rate=params["positive_rate"],
        )
        model = bert_encoder.load_bert_model()
        hidden_df["about_clean"] = hidden_df["about_the_game"].fillna("").apply(
            lambda x: __import__("re").sub(r"<[^>]+>", " ", str(x)).strip()
        )
        embeddings_about = bert_encoder.encode_hidden_gems(hidden_df["about_clean"].tolist(), model)

        update_job(app_user_id, "running", "直交射影適用中...", 0.5)
        # ナシゲームは原石プール由来なので hidden_df のタグをフォールバックとして使う
        _hidden_tags = dict(zip(hidden_df["appid"], hidden_df["tags"].fillna("")))
        nashi_tag_vecs = np.array([
            tfidf_vectorizer.transform([
                nashi_seed_data.get(a, {}).get("tags", "") or _hidden_tags.get(a, "")
            ]).toarray()[0]
            for a in nashi_appids
        ])
        nashi_about_vecs = np.array([
            model.encode(nashi_seed_data.get(a, {}).get("about", "") or " ", show_progress_bar=False)
            for a in nashi_appids
        ])

        user_vec_tags_v4 = projector.project_out(user_vec_tags, nashi_tag_vecs)
        user_vec_about_v4 = projector.project_out(user_vec_about, nashi_about_vecs)

        update_job(app_user_id, "running", "マッチング再計算中...", 0.8)
        pool = matcher.build_candidate_pool(
            user_vec_tags_v4, user_vec_about_v4, hidden_df, tfidf_matrix, embeddings_about,
            seed_appids=[], nashi_appids=nashi_appids, pool_size=200,
        )

        result_json = _serialize_result(pool, user_vec_tags_v4, user_vec_about_v4)
        update_job(app_user_id, "complete", "再計算完了", 1.0, result_json=result_json)

    except Exception as e:
        logger.exception("再計算エラー: %s", e)
        update_job(app_user_id, "error", "エラー", 0.0, error_msg=str(e))


# ── 公開 API ──────────────────────────────────────────────────────────────────

def start_pipeline(app_user_id: str, params: dict, api_key: str) -> bool:
    """パイプラインをバックグラウンドスレッドで起動する。二重起動時は False を返す。"""
    job = get_job(app_user_id)
    if job and job["status"] == "running":
        return False
    update_job(app_user_id, "running", "開始中...", 0.0)
    t = threading.Thread(target=_run_pipeline, args=(app_user_id, params, api_key), daemon=True)
    t.start()
    return True


def start_recompute(app_user_id: str, params: dict, api_key: str) -> bool:
    """直交射影再計算をバックグラウンドスレッドで起動する。"""
    job = get_job(app_user_id)
    if job and job["status"] == "running":
        return False
    update_job(app_user_id, "running", "再計算開始中...", 0.0)
    t = threading.Thread(target=_run_recompute, args=(app_user_id, params, api_key), daemon=True)
    t.start()
    return True
