import os

import streamlit as st
from dotenv import load_dotenv

from db.crud import load_params, reset_job, save_params, upsert_user
from pipeline.fetcher import validate_steam_id
from pipeline.runner import start_pipeline

load_dotenv()
API_KEY = os.getenv("STEAM_API_KEY", "")

st.set_page_config(page_title="Indie Gem Finder — セットアップ", layout="centered", page_icon="💎")

app_user_id = st.session_state.get("app_user_id")
if not app_user_id:
    st.switch_page("app.py")

st.title("💎 Indie Gem Finder")
st.caption("あなたのSteamプレイ履歴から、埋もれた名作インディーゲームを発見します。")
st.divider()

params = load_params(app_user_id)

# ── Steam ID 入力 ──────────────────────────────────────────────────────────────
user = st.session_state.get("user_steam_id", "")
steam_id_input = st.text_input(
    "🎮 Steam ID（17桁の数字またはユーザー名）",
    value=user,
    placeholder="例: 76561199040391831",
)

st.divider()
st.subheader("🎛️ レコメンドパラメータ")

col1, col2 = st.columns(2)
with col1:
    review_min = st.slider("レビュー件数 下限", 10, 500, params["review_min"], step=10)
    positive_rate = st.slider("好評率 閾値 (%)", 50, 99, int(params["positive_rate"] * 100))
with col2:
    review_max = st.slider("レビュー件数 上限", 50, 5000, params["review_max"], step=50)

if review_min >= review_max:
    st.error("レビュー件数の下限は上限より小さくしてください。")
    st.stop()

st.divider()
st.subheader("⚖️ A/B モデルの初期ブレンド設定")
col_a, col_b = st.columns(2)
with col_a:
    weight_a = st.slider("モデルA タグ比率", 0.0, 1.0, params["weight_tags_a"], step=0.1)
    st.caption(f"タグ {weight_a*100:.0f}% / ポエム {(1-weight_a)*100:.0f}%")
with col_b:
    weight_b = st.slider("モデルB タグ比率", 0.0, 1.0, params["weight_tags_b"], step=0.1)
    st.caption(f"タグ {weight_b*100:.0f}% / ポエム {(1-weight_b)*100:.0f}%")

st.divider()

if st.button("🚀 レコメンドを生成", type="primary", use_container_width=True):
    if not steam_id_input.strip():
        st.error("Steam IDを入力してください。")
        st.stop()

    with st.spinner("Steam IDを確認中..."):
        is_valid, err_msg = validate_steam_id(steam_id_input.strip(), API_KEY)

    if not is_valid:
        st.error(err_msg)
        st.stop()

    new_params = {
        "review_min": review_min,
        "review_max": review_max,
        "positive_rate": positive_rate / 100.0,
        "weight_tags_a": weight_a,
        "weight_tags_b": weight_b,
    }
    save_params(app_user_id, new_params)
    upsert_user(app_user_id, steam_id=steam_id_input.strip())
    st.session_state.user_steam_id = steam_id_input.strip()

    reset_job(app_user_id)
    started = start_pipeline(
        app_user_id,
        {**new_params, "steam_id": steam_id_input.strip()},
        API_KEY,
    )
    if started:
        st.switch_page("pages/02_computing.py")
    else:
        st.warning("計算中です。しばらくお待ちください。")
        st.switch_page("pages/02_computing.py")
