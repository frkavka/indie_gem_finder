import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from db.crud import get_job, upsert_user
from db.schema import init_db

load_dotenv()
st.set_page_config(page_title="Indie Gem Finder", layout="wide", page_icon="💎")

init_db()

# ── Cookie から AppUserID を復元 or 新規生成 ──────────────────────────────────
if "app_user_id" not in st.session_state:
    cookie_val = st.query_params.get("uid")
    if cookie_val:
        st.session_state.app_user_id = cookie_val
    else:
        st.session_state.app_user_id = str(uuid.uuid4())

app_user_id = st.session_state.app_user_id
upsert_user(app_user_id)

# ── 状態に応じてページルーティング ────────────────────────────────────────────
job = get_job(app_user_id)

if not job or job["status"] in ("idle", "error"):
    st.switch_page("pages/01_setup.py")
elif job["status"] == "running":
    st.switch_page("pages/02_computing.py")
elif job["status"] == "complete":
    st.switch_page("pages/03_deck.py")
