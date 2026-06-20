"""TASK-4001: matcher.calculate_similarities のユニットテスト"""
import numpy as np
import pandas as pd
import pytest
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.matcher import calculate_similarities


def _build_fixtures(n: int = 10):
    """テスト用のミニ原石プールとベクトルを生成する"""
    tags_list = [f"Tag{i},Indie" for i in range(n)]
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(",") if isinstance(x, str) else [],
        token_pattern=None,
    )
    tfidf_matrix = vectorizer.fit_transform(tags_list)

    df = pd.DataFrame({
        "appid":        list(range(1000, 1000 + n)),
        "name":         [f"Game{i}" for i in range(n)],
        "tags":         tags_list,
        "rating_ratio": [0.9] * n,
        "total_reviews":[100] * n,
        "movies":       [""] * n,
        "about_the_game": [""] * n,
    })

    embeddings_about = np.random.rand(n, 384)
    embeddings_about /= np.linalg.norm(embeddings_about, axis=1, keepdims=True)

    user_vec_tags = np.ones(tfidf_matrix.shape[1])
    user_vec_tags /= norm(user_vec_tags)
    user_vec_about = np.ones(384)
    user_vec_about /= norm(user_vec_about)

    return df, tfidf_matrix, embeddings_about, user_vec_tags, user_vec_about


def test_seed_appids_excluded():
    """seed_appids に含まれる AppID が結果に含まれないこと"""
    df, tfidf, emb, u_tags, u_about = _build_fixtures(10)
    seed_appids = [1000, 1001, 1002]

    result = calculate_similarities(
        u_tags, u_about, df, tfidf, emb,
        seed_appids=seed_appids, nashi_appids=[], weight_tags=0.7,
    )
    assert not any(appid in result["appid"].values for appid in seed_appids)


def test_nashi_appids_excluded():
    """nashi_appids に含まれる AppID が結果に含まれないこと"""
    df, tfidf, emb, u_tags, u_about = _build_fixtures(10)
    nashi_appids = [1005, 1006]

    result = calculate_similarities(
        u_tags, u_about, df, tfidf, emb,
        seed_appids=[], nashi_appids=nashi_appids, weight_tags=0.7,
    )
    assert not any(appid in result["appid"].values for appid in nashi_appids)


def test_top_n_limit():
    """返り値の件数が top_n を超えないこと"""
    df, tfidf, emb, u_tags, u_about = _build_fixtures(10)

    result = calculate_similarities(
        u_tags, u_about, df, tfidf, emb,
        seed_appids=[], nashi_appids=[], weight_tags=0.7, top_n=5,
    )
    assert len(result) <= 5


def test_final_score_is_weighted_blend():
    """final_score = sim_tags * weight + sim_about * (1 - weight) であること"""
    df, tfidf, emb, u_tags, u_about = _build_fixtures(10)
    w = 0.6

    result = calculate_similarities(
        u_tags, u_about, df, tfidf, emb,
        seed_appids=[], nashi_appids=[], weight_tags=w,
    )
    expected = result["sim_tags"] * w + result["sim_about"] * (1.0 - w)
    assert np.allclose(result["final_score"].values, expected.values, atol=1e-6)


def test_results_sorted_descending():
    """結果が final_score 降順でソートされていること"""
    df, tfidf, emb, u_tags, u_about = _build_fixtures(10)

    result = calculate_similarities(
        u_tags, u_about, df, tfidf, emb,
        seed_appids=[], nashi_appids=[], weight_tags=0.5,
    )
    scores = result["final_score"].tolist()
    assert scores == sorted(scores, reverse=True)
