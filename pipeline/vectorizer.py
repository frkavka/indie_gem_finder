import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import norm


def _tag_tokenizer(x: str) -> list[str]:
    # ラムダは pickle 不可のため、モジュールレベルの名前付き関数として定義する
    return x.split(",") if isinstance(x, str) else []


_COLS = [
    "appid", "name", "release_date", "estimated_owners", "peak_ccu",
    "required_age", "price", "discount", "dlc_count", "about_the_game",
    "supported_languages", "full_audio_languages", "reviews", "header_image",
    "website", "support_url", "support_email", "windows", "mac", "linux",
    "metacritic_score", "metacritic_url", "user_score", "positive", "negative",
    "score_rank", "achievements", "recommendations", "notes",
    "average_playtime_forever", "average_playtime_two_weeks",
    "median_playtime_forever", "median_playtime_two_weeks",
    "developers", "publishers", "categories", "genres", "tags",
    "screenshots", "movies",
]


@st.cache_data(show_spinner=False)
def load_all_tags(csv_path: str) -> dict[int, str]:
    """全ゲームの appid→tags_str マップを返す。SteamSpy が取れない場合のフォールバック用。"""
    df = pd.read_csv(csv_path, header=0, names=_COLS, engine="python", on_bad_lines="skip")
    df["appid"] = pd.to_numeric(df["appid"], errors="coerce")
    df = df.dropna(subset=["appid", "tags"])
    df = df[np.isfinite(df["appid"])]
    df["appid"] = df["appid"].astype(np.int64)
    return dict(zip(df["appid"], df["tags"].fillna("")))


@st.cache_data(show_spinner=False)
def load_hidden_gems(
    csv_path: str,
    review_min: int = 30,
    review_max: int = 300,
    positive_rate: float = 0.85,
) -> tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    """原石プールをフィルタリングし、TF-IDF行列を構築して返す。パラメータ変更時のみ再計算。"""
    # Kaggle の games.csv は1行目のヘッダーが壊れているため列名を強制上書きする（org.py 参照）
    df = pd.read_csv(csv_path, header=0, names=_COLS, engine="python", on_bad_lines="skip")

    df["appid"] = pd.to_numeric(df["appid"], errors="coerce")
    df = df.dropna(subset=["appid", "tags"])
    df["appid"] = df["appid"].astype(np.int64)
    df["positive"] = pd.to_numeric(df["positive"], errors="coerce").fillna(0)
    df["negative"] = pd.to_numeric(df["negative"], errors="coerce").fillna(0)
    df["total_reviews"] = df["positive"] + df["negative"]
    df["rating_ratio"] = df.apply(
        lambda r: r["positive"] / r["total_reviews"] if r["total_reviews"] > 0 else 0,
        axis=1,
    )

    hidden = df[
        (df["total_reviews"] >= review_min)
        & (df["total_reviews"] <= review_max)
        & (df["rating_ratio"] >= positive_rate)
        & (df["supported_languages"].str.contains("Japanese", case=False, na=False))
    ].copy()

    if hidden.empty:
        raise ValueError(
            f"指定条件（レビュー {review_min}〜{review_max}件、好評率 {positive_rate*100:.0f}%以上）"
            "に該当するゲームが0件です。条件を緩和してください。"
        )

    vectorizer = TfidfVectorizer(tokenizer=_tag_tokenizer, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(hidden["tags"])
    return hidden.reset_index(drop=True), vectorizer, tfidf_matrix


def build_user_vector_tags(
    seed_data: dict[int, dict],
    vectorizer: TfidfVectorizer,
    seed_weights: dict[int, float],
    csv_tags: dict[int, str] | None = None,
) -> np.ndarray:
    """重み付きTF-IDFユーザーベクトルを構築・L2正規化して返す。
    SteamSpy タグが空の場合は csv_tags（Kaggle CSV）をフォールバックとして使う。
    """
    vec = np.zeros(len(vectorizer.get_feature_names_out()))
    for appid, data in seed_data.items():
        tags = data.get("tags", "")
        if not tags and csv_tags:
            tags = csv_tags.get(int(appid), "")
        if not tags:
            continue
        weight = seed_weights.get(appid, 1.0)
        vec += vectorizer.transform([tags]).toarray()[0] * weight

    n = norm(vec)
    return vec / n if n > 1e-9 else vec
