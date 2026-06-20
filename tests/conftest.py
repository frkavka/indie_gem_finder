"""
streamlit のキャッシュデコレーターをテスト時に無効化する。
モジュールが import される前に patch しないと decorator が適用されてしまうため
sys.modules に差し込む方式を使う。
"""
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def _make_streamlit_stub() -> ModuleType:
    st = MagicMock()
    st.cache_data = lambda *args, **kwargs: (lambda fn: fn)
    st.cache_resource = lambda *args, **kwargs: (lambda fn: fn)
    return st


# pytest セッション開始前に一度だけ差し込む
@pytest.fixture(autouse=True, scope="session")
def _patch_streamlit():
    sys.modules.setdefault("streamlit", _make_streamlit_stub())
    yield
