"""TASK-4001: vectorizer.build_user_vector_tags のユニットテスト"""
import numpy as np
import pytest
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.vectorizer import build_user_vector_tags, _tag_tokenizer


def _make_vectorizer(corpus: list[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(tokenizer=_tag_tokenizer, token_pattern=None)
    vec.fit(corpus)
    return vec


def test_output_is_unit_vector():
    """ユーザーベクトルの L2 ノルムが 1.0 ± 1e-6 であること"""
    corpus = ["Action,RPG,Indie", "Puzzle,Casual", "Horror,Survival"]
    vec = _make_vectorizer(corpus)
    seed_data = {
        12345: {"tags": "Action,RPG"},
        67890: {"tags": "Puzzle,Casual"},
    }
    seed_weights = {12345: 5.0, 67890: 2.0}

    result = build_user_vector_tags(seed_data, vec, seed_weights)
    assert abs(norm(result) - 1.0) < 1e-6


def test_empty_tags_skipped():
    """タグが空のシードゲームはスキップされ、非ゼロベクトルが返ること"""
    corpus = ["Action,RPG", "Puzzle"]
    vec = _make_vectorizer(corpus)
    seed_data = {
        12345: {"tags": "Action,RPG"},
        99999: {"tags": ""},           # 空タグ → スキップ
    }
    seed_weights = {12345: 3.0, 99999: 5.0}

    result = build_user_vector_tags(seed_data, vec, seed_weights)
    assert norm(result) > 1e-9


def test_weights_influence_direction():
    """重みが大きいゲームの方向に引っ張られること"""
    corpus = ["Action", "Puzzle"]
    vec = _make_vectorizer(corpus)
    seed_data = {
        1: {"tags": "Action"},
        2: {"tags": "Puzzle"},
    }

    # Action に強い重み
    result_action = build_user_vector_tags(seed_data, vec, {1: 10.0, 2: 1.0})
    # Puzzle に強い重み
    result_puzzle = build_user_vector_tags(seed_data, vec, {1: 1.0, 2: 10.0})

    # sklearn は lowercase=True がデフォルトなので特徴名は小文字になる
    action_idx = list(vec.get_feature_names_out()).index("action")
    puzzle_idx = list(vec.get_feature_names_out()).index("puzzle")

    assert result_action[action_idx] > result_puzzle[action_idx]
    assert result_puzzle[puzzle_idx] > result_action[puzzle_idx]


def test_all_empty_tags_raises():
    """全シードのタグが空の場合はゼロベクトルを黙って返さず ValueError を送出すること。
    サイレントに返すと sim_tags が全件 0.0 になり A/B比較が退化するため（フェイルファスト）。
    """
    corpus = ["Action,RPG"]
    vec = _make_vectorizer(corpus)
    seed_data = {1: {"tags": ""}, 2: {"tags": ""}}
    seed_weights = {1: 1.0, 2: 1.0}

    with pytest.raises(ValueError, match="タグ"):
        build_user_vector_tags(seed_data, vec, seed_weights)
