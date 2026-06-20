import time

import streamlit as st

from db.crud import get_job, reset_job

st.set_page_config(page_title="計算中... — Indie Gem Finder", layout="centered", page_icon="⚙️")

app_user_id = st.session_state.get("app_user_id")
if not app_user_id:
    st.switch_page("app.py")

st.title("⚙️ レコメンドを計算中...")
st.caption("Steam APIとAIベクトル計算を実行しています。このまましばらくお待ちください。")

job = get_job(app_user_id)

if not job:
    st.warning("計算ジョブが見つかりません。")
    if st.button("最初からやり直す"):
        st.switch_page("pages/01_setup.py")
    st.stop()

if job["status"] == "running":
    progress = job["progress"] or 0.0
    phase = job["phase"] or "処理中..."
    st.progress(progress, text=f"**{phase}**")
    st.caption(f"進捗: {progress*100:.0f}%")
    time.sleep(1)
    st.rerun()

elif job["status"] == "complete":
    st.success("計算完了！カードを表示します...")
    time.sleep(0.5)
    st.switch_page("pages/03_deck.py")

elif job["status"] == "error":
    st.error(f"エラーが発生しました: {job['error_msg']}")
    st.caption("パラメータを調整して再度お試しください。")
    if st.button("🔄 最初からやり直す", use_container_width=True):
        reset_job(app_user_id)
        st.switch_page("pages/01_setup.py")
