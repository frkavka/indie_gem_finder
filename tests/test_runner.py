import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from pipeline import projector
from pipeline.runner import _lookup_nashi_vectors, _serialize_result, deserialize_result


@pytest.fixture
def pool_assets():
    """appid [10, 20, 30] の3件からなる整列済みミニプール。"""
    hidden_df = pd.DataFrame({
        "appid": [10, 20, 30],
        "name": ["Alpha", "Beta", "Gamma"],
    })
    tfidf_matrix = scipy.sparse.csr_matrix(np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]))
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ])
    return hidden_df, tfidf_matrix, embeddings


def test_lookup_returns_rows_in_nashi_order(pool_assets):
    hidden_df, tfidf_matrix, embeddings = pool_assets

    tag_vecs, about_vecs = _lookup_nashi_vectors([30, 10], hidden_df, tfidf_matrix, embeddings)

    np.testing.assert_array_equal(tag_vecs, [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(about_vecs, [[0.5, 0.5], [1.0, 0.0]])


def test_lookup_empty_nashi_returns_zero_rows_and_projection_is_noop(pool_assets):
    hidden_df, tfidf_matrix, embeddings = pool_assets

    tag_vecs, about_vecs = _lookup_nashi_vectors([], hidden_df, tfidf_matrix, embeddings)

    assert tag_vecs.shape == (0, 4)
    assert about_vecs.shape == (0, 2)
    user_vec = np.array([3.0, 4.0])
    np.testing.assert_allclose(projector.project_out(user_vec, about_vecs), [0.6, 0.8])


def test_lookup_unknown_appid_is_skipped_with_warning(pool_assets, caplog):
    """プール再ビルドで資産から消えた appid は warning を出してスキップする。"""
    hidden_df, tfidf_matrix, embeddings = pool_assets

    with caplog.at_level("WARNING", logger="pipeline.runner"):
        tag_vecs, about_vecs = _lookup_nashi_vectors(
            [99999, 20], hidden_df, tfidf_matrix, embeddings
        )

    np.testing.assert_array_equal(tag_vecs, [[0.0, 1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(about_vecs, [[0.0, 1.0]])
    assert "99999" in caplog.text


def test_serialize_roundtrip_keeps_seed_appids():
    pool = pd.DataFrame({"appid": [10], "name": ["Alpha"], "sim_tags": [0.5], "sim_about": [0.4]})

    result_json = _serialize_result(pool, np.array([1.0, 0.0]), np.array([0.0, 1.0]), [111, 222])
    data = deserialize_result(result_json)

    assert data["seed_appids"] == [111, 222]
    np.testing.assert_array_equal(data["user_vecs"]["tags"], [1.0, 0.0])


def test_deserialize_legacy_result_without_seed_appids():
    legacy = json.dumps({
        "pool": [],
        "user_vecs": {"tags": [1.0], "about": [1.0]},
    })

    data = deserialize_result(legacy)

    assert data["seed_appids"] == []
