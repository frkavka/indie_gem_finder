import json
import logging
import threading

import numpy as np
import pandas as pd
from db.crud import get_job, get_nashi_list, update_job

from pipeline import bert_encoder, fetcher, matcher, precompute, projector, vectorizer

logger = logging.getLogger(__name__)


# ── シリアライズ ──────────────────────────────────────────────────────────────


def _serialize_result(
    pool: pd.DataFrame,
    user_vec_tags: np.ndarray,
    user_vec_about: np.ndarray,
    seed_appids: list[int],
) -> str:
    export_cols = [
        "appid",
        "name",
        "sim_tags",
        "sim_about",
        "rating_ratio",
        "tags",
        "movies",
        "total_reviews",
    ]
    cols = [c for c in export_cols if c in pool.columns]
    return json.dumps(
        {
            "pool": pool[cols].to_dict(orient="records"),
            "user_vecs": {
                "tags": user_vec_tags.tolist(),
                "about": user_vec_about.tolist(),
            },
            "seed_appids": seed_appids,
        },
        ensure_ascii=False,
    )


def deserialize_result(result_json: str) -> dict:
    data = json.loads(result_json)
    data["user_vecs"]["tags"] = np.array(data["user_vecs"]["tags"])
    data["user_vecs"]["about"] = np.array(data["user_vecs"]["about"])
    # seed_appids 導入前に保存された result_json との互換。当時の再計算は
    # シード除外なし（= 空リスト）で動いていたため、挙動としては等価。
    data.setdefault("seed_appids", [])
    return data


# ── パイプライン本体 ──────────────────────────────────────────────────────────


def _run_pipeline(app_user_id: str, params: dict, api_key: str) -> None:
    try:
        # Phase 1: Steam ライブラリ + ウィッシュリスト取得
        update_job(app_user_id, "running", "Fetching Steam library...", 0.05)
        seed_games = fetcher.fetch_seed_games(params["steam_id"], api_key)
        if not seed_games:
            raise ValueError("No play history or wishlist found for this Steam ID.")
        seed_appids = [g["appid"] for g in seed_games]
        seed_weights = {g["appid"]: g["weight_score"] for g in seed_games}
        update_job(app_user_id, "running", "Steam library fetched.", 0.2)

        # Phase 2: タグ対応表 + Steam Store データ並列取得
        # （タグは games.csv 由来の対応表を一次供給源とし、SteamSpy は新作のみ）
        update_job(
            app_user_id, "running", "Fetching game data (using API cache)...", 0.25
        )
        seed_tags_map = precompute.load_seed_tags()
        seed_data = fetcher.fetch_seed_data_parallel(
            seed_appids, max_workers=3, static_tags=seed_tags_map
        )
        update_job(app_user_id, "running", "Game data fetched.", 0.4)

        # Phase 3: 事前計算済みプールをロード
        update_job(app_user_id, "running", "Loading pre-computed gem pool...", 0.45)
        hidden_df, tfidf_vectorizer, tfidf_matrix, embeddings_about = precompute.load()

        # Phase 4: TF-IDF ユーザーベクトル
        update_job(app_user_id, "running", "Building tag vector...", 0.55)
        user_vec_tags = vectorizer.build_user_vector_tags(
            seed_data, tfidf_vectorizer, seed_weights
        )

        # Phase 5: BERT ユーザーベクトル（シードゲームのみ）
        update_job(app_user_id, "running", "Building BERT user vector...", 0.65)
        model = bert_encoder.load_bert_model()
        user_vec_about = bert_encoder.build_user_vector_about(
            seed_data, seed_weights, model
        )
        update_job(app_user_id, "running", "Vectors built.", 0.8)

        # Phase 6: 候補プール構築
        update_job(app_user_id, "running", "Matching...", 0.85)
        nashi_appids = get_nashi_list(app_user_id)
        pool = matcher.build_candidate_pool(
            user_vec_tags,
            user_vec_about,
            hidden_df,
            tfidf_matrix,
            embeddings_about,
            seed_appids,
            nashi_appids,
            pool_size=200,
        )

        result_json = _serialize_result(
            pool, user_vec_tags, user_vec_about, seed_appids
        )
        update_job(app_user_id, "complete", "Done.", 1.0, result_json=result_json)

    except Exception as e:
        logger.exception("Pipeline error: %s", e)
        update_job(app_user_id, "error", "Error", 0.0, error_msg=str(e))


# ── 軽量再計算（V4直交射影） ────────────────────────────────────────────────────


def _lookup_nashi_vectors(
    nashi_appids: list[int],
    hidden_df: pd.DataFrame,
    tfidf_matrix,
    embeddings_about: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """ナシゲームのベクトルを事前計算済み資産から行indexで引く。

    ナシ判定はデッキ（= プール）上でしか発生しないため、appid は原則
    hidden_df に存在する。ただし precompute を再ビルドしてプールの中身が
    変わった場合、旧プール由来のナシ appid が見つからないことがある。
    hidden_df / tfidf_matrix / embeddings_about は precompute.build() が
    同一 DataFrame から出力するため行が整列している前提。
    """
    appid_to_row = {appid: i for i, appid in enumerate(hidden_df["appid"])}

    rows = []
    for appid in nashi_appids:
        if appid not in appid_to_row:
            # プール再ビルドで資産から消えた appid。新プールに再侵入することは
            # あり得ないため、射影方向1本を失うだけでスキップは安全。
            logger.warning(
                "Nashi appid %d not found in precomputed pool; skipping projection.",
                appid,
            )
            continue
        rows.append(appid_to_row[appid])

    if not rows:
        return (
            np.zeros((0, tfidf_matrix.shape[1])),
            np.zeros((0, embeddings_about.shape[1])),
        )

    return tfidf_matrix[rows].toarray(), embeddings_about[rows]


def run_recompute_light(app_user_id: str) -> None:
    """直交射影の再計算を事前計算済みベクトルのみで同期実行する。

    API 呼び出し・BERT モデルのロードを一切行わないため Streamlit Cloud の
    ような非力なサーバーでもサブ秒で完了する。失敗時は例外を送出する
    （呼び出し側の Streamlit ページで表示する）。
    """
    job = get_job(app_user_id)
    if not job or not job["result_json"]:
        raise ValueError("No previous result found. Please run from the beginning.")

    prev = deserialize_result(job["result_json"])
    nashi_appids = get_nashi_list(app_user_id)

    hidden_df, _, tfidf_matrix, embeddings_about = precompute.load()
    nashi_tag_vecs, nashi_about_vecs = _lookup_nashi_vectors(
        nashi_appids, hidden_df, tfidf_matrix, embeddings_about
    )

    user_vec_tags_v4 = projector.project_out(prev["user_vecs"]["tags"], nashi_tag_vecs)
    user_vec_about_v4 = projector.project_out(
        prev["user_vecs"]["about"], nashi_about_vecs
    )

    pool = matcher.build_candidate_pool(
        user_vec_tags_v4,
        user_vec_about_v4,
        hidden_df,
        tfidf_matrix,
        embeddings_about,
        seed_appids=prev["seed_appids"],
        nashi_appids=nashi_appids,
        pool_size=200,
    )

    result_json = _serialize_result(
        pool, user_vec_tags_v4, user_vec_about_v4, prev["seed_appids"]
    )
    update_job(app_user_id, "complete", "Recompute done.", 1.0, result_json=result_json)


# ── 公開 API ──────────────────────────────────────────────────────────────────


def start_pipeline(app_user_id: str, params: dict, api_key: str) -> bool:
    job = get_job(app_user_id)
    if job and job["status"] == "running":
        return False
    update_job(app_user_id, "running", "Starting...", 0.0)
    t = threading.Thread(
        target=_run_pipeline, args=(app_user_id, params, api_key), daemon=True
    )
    t.start()
    return True
