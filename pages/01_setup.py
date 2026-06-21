import os

import streamlit as st
from dotenv import load_dotenv

from db.crud import load_params, reset_job, save_params, upsert_user
from i18n import render_lang_selector, t
from pipeline.fetcher import validate_steam_id
from pipeline.runner import start_pipeline

load_dotenv()
API_KEY = os.getenv("STEAM_API_KEY", "")
IS_PREMIUM = os.getenv("PREMIUM", "false").lower() == "true"

# 事前計算プールのデフォルト値（無料プランで固定される値）
_POOL_DEFAULTS = {"review_min": 30, "review_max": 300, "positive_rate": 0.85}

st.set_page_config(page_title="Indie Gem Finder — Setup", layout="centered", page_icon="💎")

app_user_id = st.session_state.get("app_user_id")
if not app_user_id:
    st.switch_page("app.py")

render_lang_selector()

st.title(t("title"))
st.caption(t("subtitle"))
st.divider()

params = load_params(app_user_id)

user = st.session_state.get("user_steam_id", "")
steam_id_input = st.text_input(
    t("steam_id_label"),
    value=user,
    placeholder=t("steam_id_placeholder"),
)

st.divider()
st.subheader(t("section_params"))

if IS_PREMIUM:
    col1, col2 = st.columns(2)
    with col1:
        review_min = st.slider(t("review_min_label"), 10, 500, params["review_min"], step=10)
        positive_rate = st.slider(t("positive_rate_label"), 50, 99, int(params["positive_rate"] * 100))
    with col2:
        review_max = st.slider(t("review_max_label"), 50, 5000, params["review_max"], step=50)

    if review_min >= review_max:
        st.error(t("review_range_error"))
        st.stop()
else:
    review_min = _POOL_DEFAULTS["review_min"]
    review_max = _POOL_DEFAULTS["review_max"]
    positive_rate = int(_POOL_DEFAULTS["positive_rate"] * 100)
    st.info(t("params_locked_note"))

st.divider()
st.subheader(t("section_ab"))
col_a, col_b = st.columns(2)
with col_a:
    weight_a = st.slider(t("model_a_label"), 0.0, 1.0, params["weight_tags_a"], step=0.1)
    st.caption(t("blend_caption", tag=weight_a * 100, prose=(1 - weight_a) * 100))
with col_b:
    weight_b = st.slider(t("model_b_label"), 0.0, 1.0, params["weight_tags_b"], step=0.1)
    st.caption(t("blend_caption", tag=weight_b * 100, prose=(1 - weight_b) * 100))

st.divider()

if st.button(t("btn_generate"), type="primary", use_container_width=True):
    if not steam_id_input.strip():
        st.error(t("error_no_steam_id"))
        st.stop()

    with st.spinner(t("spinner_validating")):
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
        st.warning(t("warn_already_running"))
        st.switch_page("pages/02_computing.py")
