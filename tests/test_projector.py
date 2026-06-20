"""TASK-4001: projector.project_out のユニットテスト"""
import numpy as np
import pytest
from numpy.linalg import norm

from pipeline.projector import project_out


def _unit(v: np.ndarray) -> np.ndarray:
    return v / norm(v)


def test_output_is_unit_vector():
    """射影後のベクトルは L2 ノルム = 1.0 ± 1e-6 であること"""
    user = _unit(np.array([1.0, 2.0, 3.0]))
    game = np.array([[1.0, 0.0, 0.0]])
    result = project_out(user, game)
    assert abs(norm(result) - 1.0) < 1e-6


def test_similarity_decreases_after_projection():
    """射影後はナシゲームとのコサイン類似度が下がること"""
    # user と nashi を高相関だが完全同一にしない（同一だとゼロ割れフォールバックが発動する）
    user = _unit(np.array([1.0, 1.0, 0.0]))
    nashi_vec = _unit(np.array([1.0, 0.3, 0.0]))  # 方向が近いが同一ではない

    sim_before = np.dot(user, nashi_vec)
    result = project_out(user, np.array([nashi_vec]))
    sim_after = np.dot(result, nashi_vec)

    assert sim_after < sim_before


def test_orthogonal_game_unchanged():
    """ナシゲームと直交するベクトルは射影後も方向が変わらないこと"""
    user = _unit(np.array([0.0, 0.0, 1.0]))
    nashi_vec = np.array([[1.0, 0.0, 0.0]])  # user と直交

    result = project_out(user, nashi_vec)
    assert np.allclose(result, user, atol=1e-6)


def test_multiple_nashi_games():
    """複数のナシゲームを連続射影できること"""
    user = _unit(np.array([1.0, 1.0, 1.0]))
    game_vecs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    result = project_out(user, game_vecs)
    assert abs(norm(result) - 1.0) < 1e-6
    # 射影後は x, y 成分がほぼ 0
    assert abs(result[0]) < 1e-6
    assert abs(result[1]) < 1e-6


def test_near_zero_norm_fallback():
    """射影でノルムが極小になった場合は元のベクトルを返すこと"""
    user = _unit(np.array([1.0, 0.0, 0.0]))
    # user と完全一致するゲームベクトルをすべて射影 → 結果は 0 ベクトルになるはず
    game_vecs = np.array([[1.0, 0.0, 0.0]])
    result = project_out(user, game_vecs)
    # フォールバックにより result はノルム 1 を維持する
    assert abs(norm(result) - 1.0) < 1e-6
