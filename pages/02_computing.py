import time

import streamlit as st

from db.crud import get_job, reset_job
from i18n import render_lang_selector, t

st.set_page_config(page_title="Indie Gem Finder", layout="centered", page_icon="⚙️")

app_user_id = st.session_state.get("app_user_id")
if not app_user_id:
    st.switch_page("app.py")

render_lang_selector()

st.title(t("computing_title"))
st.caption(t("computing_caption"))

job = get_job(app_user_id)

if not job:
    st.warning(t("warn_no_job"))
    if st.button(t("btn_restart")):
        st.switch_page("pages/01_setup.py")
    st.stop()

if job["status"] == "running":
    progress = job["progress"] or 0.0
    phase = job["phase"] or "..."
    st.progress(progress, text=f"**{phase}**")
    st.caption(t("progress_caption", pct=progress * 100))
    time.sleep(1)
    st.rerun()

elif job["status"] == "complete":
    st.success(t("success_complete"))
    time.sleep(0.5)
    st.switch_page("pages/03_deck.py")

elif job["status"] == "error":
    st.error(t("error_occurred", msg=job["error_msg"]))
    st.caption(t("error_hint"))
    if st.button(t("btn_retry"), use_container_width=True):
        reset_job(app_user_id)
        st.switch_page("pages/01_setup.py")
