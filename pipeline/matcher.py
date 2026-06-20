import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_candidate_pool(
    user_vec_tags: np.ndarray,
    user_vec_about: np.ndarray,
    hidden_gems_df: pd.DataFrame,
    tfidf_matrix,
    embeddings_about: np.ndarray,
    seed_appids: list[int],
    nashi_appids: list[int],
    pool_size: int = 200,
) -> pd.DataFrame:
    """sim_tags・sim_about を計算した候補プールを返す（weight は適用しない）。
    デッキUIがスライダー値でリアルタイム再ソートするためにスコアを分離して保持する。
    seed_appids と nashi_appids は物理除外する。
    """
    df = hidden_gems_df.copy()
    exclude = set(seed_appids) | set(nashi_appids)
    mask = ~df["appid"].isin(exclude)
    df = df[mask].reset_index(drop=True)

    tfidf_filtered = tfidf_matrix[np.where(mask)[0]]
    embeddings_filtered = embeddings_about[np.where(mask)[0]]

    df["sim_tags"] = cosine_similarity([user_vec_tags], tfidf_filtered)[0]
    df["sim_about"] = cosine_similarity([user_vec_about], embeddings_filtered)[0]

    # pool_size は両スコアの最大値で粗くフィルタして絞る（全件保存を避ける）
    df["_max_sim"] = df[["sim_tags", "sim_about"]].max(axis=1)
    pool = (
        df.sort_values("_max_sim", ascending=False)
        .head(pool_size)
        .drop(columns=["_max_sim"])
        .reset_index(drop=True)
    )
    return pool


def calculate_similarities(
    user_vec_tags: np.ndarray,
    user_vec_about: np.ndarray,
    hidden_gems_df: pd.DataFrame,
    tfidf_matrix,
    embeddings_about: np.ndarray,
    seed_appids: list[int],
    nashi_appids: list[int],
    weight_tags: float,
    top_n: int = 20,
) -> pd.DataFrame:
    """後方互換のためのラッパー。テストで引き続き使用する。"""
    pool = build_candidate_pool(
        user_vec_tags, user_vec_about, hidden_gems_df, tfidf_matrix, embeddings_about,
        seed_appids, nashi_appids, pool_size=top_n * 10,
    )
    pool["final_score"] = pool["sim_tags"] * weight_tags + pool["sim_about"] * (1.0 - weight_tags)
    return pool.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)
