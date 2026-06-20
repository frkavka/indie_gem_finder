import numpy as np
import streamlit as st
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer


@st.cache_resource(show_spinner=False)
def load_bert_model() -> SentenceTransformer:
    """all-MiniLM-L6-v2 をシングルトンでロード（サーバー起動時1回のみ）。"""
    return SentenceTransformer("all-MiniLM-L6-v2")


def encode_hidden_gems(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """原石プール全件の About テキストを 384次元にエンコードする。shape: [N, 384]"""
    return model.encode(texts, show_progress_bar=False, batch_size=32)


def build_user_vector_about(
    seed_data: dict[int, dict],
    seed_weights: dict[int, float],
    model: SentenceTransformer,
) -> np.ndarray:
    """重み付きBERTユーザーベクトルを構築・L2正規化して返す。"""
    dim = model.get_sentence_embedding_dimension()
    vec = np.zeros(dim)
    for appid, data in seed_data.items():
        about = data.get("about", "").strip()
        if not about:
            continue
        weight = seed_weights.get(appid, 1.0)
        vec += model.encode(about, show_progress_bar=False) * weight

    n = norm(vec)
    return vec / n if n > 1e-9 else vec
