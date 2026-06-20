"""
Offline pre-compute script for the hidden gem pool.

Saves TF-IDF matrix, BERT embeddings, and game metadata to disk so
the per-user pipeline only needs to encode seed games (not the full pool).

Usage:
    python -m pipeline.precompute [--csv data/games.csv] [--out precomputed/]
                                  [--review-min 30] [--review-max 300]
                                  [--positive-rate 0.85]
"""

import argparse
import json
import logging
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

logger = logging.getLogger(__name__)

_DEFAULT_OUT = "precomputed"
_META_COLS = ["appid", "name", "tags", "rating_ratio", "movies", "total_reviews"]

_lock = threading.Lock()
_cache: dict = {}


# ── TfidfVectorizer の pickle なし保存・復元 ──────────────────────────────────

def _save_tfidf(vec: TfidfVectorizer, out: Path) -> None:
    """vocabulary と idf を JSON + npy に分割して保存する（pickle 不使用）。"""
    (out / "tfidf_vocab.json").write_text(
        json.dumps(vec.vocabulary_, ensure_ascii=False)
    )
    np.save(str(out / "tfidf_idf.npy"), vec.idf_)


def _load_tfidf(out: Path) -> TfidfVectorizer:
    """vocabulary + idf から TfidfVectorizer を再構築する（pickle 不使用）。"""
    from pipeline.vectorizer import _tag_tokenizer

    vocab: dict[str, int] = json.loads((out / "tfidf_vocab.json").read_text())
    idf: np.ndarray = np.load(str(out / "tfidf_idf.npy"))

    vec = TfidfVectorizer(tokenizer=_tag_tokenizer, token_pattern=None)
    vec.vocabulary_ = vocab
    vec.fixed_vocabulary_ = True

    # sklearn 1.5+ では _tfidf は fit まで生成されないため明示的に作る。
    # check_is_fitted は「_ で終わる属性」の存在で判定するため n_features_in_ も必要。
    transformer = TfidfTransformer()
    n = len(idf)
    transformer._idf_diag = scipy.sparse.diags(
        idf, offsets=0, shape=(n, n), format="csr", dtype=np.float64
    )
    transformer.n_features_in_ = n
    vec._tfidf = transformer
    return vec


# ── ビルド ────────────────────────────────────────────────────────────────────

def build(
    csv_path: str = "data/games.csv",
    out_dir: str = _DEFAULT_OUT,
    review_min: int = 30,
    review_max: int = 300,
    positive_rate: float = 0.85,
) -> None:
    """原石プールを事前計算してディスクに保存する。"""
    from pipeline import bert_encoder, vectorizer

    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    logger.info("原石プールをフィルタ中 (review %d–%d, positive_rate %.0f%%)...",
                review_min, review_max, positive_rate * 100)
    hidden_df, tfidf_vec, tfidf_matrix = vectorizer.load_hidden_gems(
        csv_path,
        review_min=review_min,
        review_max=review_max,
        positive_rate=positive_rate,
    )
    logger.info("対象: %d 件", len(hidden_df))

    about_texts = (
        hidden_df["about_the_game"]
        .fillna("")
        .apply(lambda x: re.sub(r"<[^>]+>", " ", str(x)).strip())
        .tolist()
    )

    logger.info("BERTエンコード中...")
    model = bert_encoder.load_bert_model()
    embeddings = bert_encoder.encode_hidden_gems(about_texts, model)

    cols = [c for c in _META_COLS if c in hidden_df.columns]
    hidden_df[cols].to_parquet(out / "pool_meta.parquet", index=False)

    _save_tfidf(tfidf_vec, out)
    scipy.sparse.save_npz(str(out / "tfidf_matrix.npz"), tfidf_matrix)
    np.save(str(out / "bert_embeddings.npy"), embeddings)

    logger.info("保存完了 → %s/ (%d 件)", out_dir, len(hidden_df))


# ── ロード ────────────────────────────────────────────────────────────────────

def load(out_dir: str = _DEFAULT_OUT) -> tuple[
    pd.DataFrame,
    TfidfVectorizer,
    scipy.sparse.csr_matrix,
    np.ndarray,
]:
    """事前計算済みプールをロードする。プロセス内でキャッシュされる。"""
    global _cache
    with _lock:
        if _cache:
            return (
                _cache["hidden_df"],
                _cache["tfidf_vec"],
                _cache["tfidf_matrix"],
                _cache["embeddings"],
            )

        out = Path(out_dir)
        required = [
            out / "pool_meta.parquet",
            out / "tfidf_vocab.json",
            out / "tfidf_idf.npy",
            out / "tfidf_matrix.npz",
            out / "bert_embeddings.npy",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(
                f"事前計算ファイルが見つかりません: {missing}\n"
                "`python -m pipeline.precompute` を先に実行してください。"
            )

        logger.info("事前計算済みプールをロード中 (%s/) ...", out_dir)
        hidden_df = pd.read_parquet(out / "pool_meta.parquet")
        tfidf_vec = _load_tfidf(out)
        tfidf_matrix = scipy.sparse.load_npz(str(out / "tfidf_matrix.npz"))
        embeddings = np.load(str(out / "bert_embeddings.npy"))

        _cache = {
            "hidden_df": hidden_df,
            "tfidf_vec": tfidf_vec,
            "tfidf_matrix": tfidf_matrix,
            "embeddings": embeddings,
        }
        logger.info("ロード完了 (%d 件)", len(hidden_df))

    return _cache["hidden_df"], _cache["tfidf_vec"], _cache["tfidf_matrix"], _cache["embeddings"]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Pre-compute hidden gem pool")
    parser.add_argument("--csv", default="data/games.csv")
    parser.add_argument("--out", default=_DEFAULT_OUT)
    parser.add_argument("--review-min", type=int, default=30)
    parser.add_argument("--review-max", type=int, default=300)
    parser.add_argument("--positive-rate", type=float, default=0.85)
    args = parser.parse_args()

    build(args.csv, args.out, args.review_min, args.review_max, args.positive_rate)
