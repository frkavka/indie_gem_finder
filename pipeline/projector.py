import logging

import numpy as np
from numpy.linalg import norm

logger = logging.getLogger(__name__)


def project_out(user_vec: np.ndarray, game_vecs: np.ndarray) -> np.ndarray:
    """各ナシゲームの方向成分を user_vec から直交射影で除去し、再正規化して返す。

    game_vecs: shape [n_nashi, dim]
    ノルムが 1e-9 未満になった場合は元のベクトルを返し warning を出す。
    """
    result = user_vec.copy()
    for game_vec in game_vecs:
        n = norm(game_vec)
        if n < 1e-9:
            continue
        unit = game_vec / n
        dot = np.dot(result, unit)
        result = result - dot * unit

    final_norm = norm(result)
    if final_norm < 1e-9:
        logger.warning("直交射影後のノルムが極小 (%.2e)。元のベクトルを維持します。", final_norm)
        return user_vec / norm(user_vec)
    return result / final_norm
