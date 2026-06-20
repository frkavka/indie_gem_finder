import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from db.crud import (
    add_nashi,
    count_nashi,
    get_job,
    get_nashi_list,
    load_params,
    reset_job,
)
from pipeline.runner import deserialize_result, start_recompute
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("STEAM_API_KEY", "")

st.set_page_config(page_title="Indie Gem Finder", layout="wide", page_icon="💎")

app_user_id = st.session_state.get("app_user_id")
if not app_user_id:
    st.switch_page("app.py")

job = get_job(app_user_id)
if not job or job["status"] != "complete" or not job["result_json"]:
    st.switch_page("app.py")

result = deserialize_result(job["result_json"])
pool_df = pd.DataFrame(result["pool"])  # sim_tags・sim_about 付き候補プール
params = load_params(app_user_id)
nashi_count = count_nashi(app_user_id)

# ── ヘッダー ──────────────────────────────────────────────────────────────────
st.title("⚖️ Indie Gem Finder — A/B デュアルデッキ")
st.caption("左右で異なるAIブレンドを比較しながら原石を発掘しましょう。")

if nashi_count >= 20:
    st.info(f"ナシ判定が {nashi_count} 件溜まりました。好みを反映して再計算できます。")
    if st.button("🔄 好みを反映してもう一度", type="primary", use_container_width=True):
        nashi_appids = get_nashi_list(app_user_id)
        start_recompute(app_user_id, {**params, "steam_id": st.session_state.get("user_steam_id", "")}, API_KEY)
        st.switch_page("pages/02_computing.py")

st.divider()

# ── セッションステート初期化 ──────────────────────────────────────────────────
for side in ["A", "B"]:
    if f"idx_{side}" not in st.session_state:
        st.session_state[f"idx_{side}"] = 0
        st.session_state[f"keep_{side}"] = []
        st.session_state[f"weight_{side}"] = params[f"weight_tags_{side.lower()}"]


def _render_deck(side: str, col, pool_df: pd.DataFrame) -> None:
    with col:
        st.subheader(f"🤖 モデル {side}")

        weight = st.slider(
            f"タグ ↔ ポエム [{side}]",
            0.0, 1.0,
            st.session_state[f"weight_{side}"],
            step=0.1,
            key=f"slider_{side}",
        )
        st.caption(f"タグ {weight*100:.0f}% / ポエム {(1-weight)*100:.0f}%")

        # スライダー変更でデッキリセット
        if st.session_state[f"weight_{side}"] != weight:
            st.session_state[f"weight_{side}"] = weight
            st.session_state[f"idx_{side}"] = 0
            st.rerun()

        # 候補プールをリアルタイム再ソートして top-20 を取り出す
        sorted_df = pool_df.copy()
        sorted_df["final_score"] = sorted_df["sim_tags"] * weight + sorted_df["sim_about"] * (1.0 - weight)
        games = (
            sorted_df.sort_values("final_score", ascending=False)
            .head(20)
            .to_dict(orient="records")
        )

        idx = st.session_state[f"idx_{side}"]
        total = len(games)

        if idx >= total:
            st.success(f"🎉 モデル {side} のデッキを全件確認しました！")
            keeps = st.session_state[f"keep_{side}"]
            if keeps:
                st.write("**❤️ アリリスト**")
                for g in keeps:
                    st.write(f"- [{g['name']}](https://store.steampowered.com/app/{g['appid']}/)")
            else:
                st.info("アリ判定したゲームはありませんでした。")
            if st.button(f"🔄 モデル {side} をやり直す", key=f"reset_{side}", use_container_width=True):
                st.session_state[f"idx_{side}"] = 0
                st.session_state[f"keep_{side}"] = []
                st.rerun()
            return

        game = games[idx]
        appid = int(game["appid"])
        img_url = f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{appid}/header.jpg"
        movies_raw = str(game.get("movies", ""))
        video_url = next(
            (u.strip() for u in movies_raw.split(",") if u.strip().startswith("http")),
            "",
        )

        st.write("---")
        if video_url:
            components.html(
                f"""
                <div style="width:100%;border-radius:8px;overflow:hidden;background:#000">
                  <video src="{video_url}" poster="{img_url}" loop muted playsinline
                         onmouseover="this.play()" onmouseout="this.pause();this.currentTime=0"
                         style="width:100%;display:block;cursor:pointer">
                  </video>
                </div>
                """,
                height=200,
            )
        else:
            st.image(img_url, use_container_width=True)

        st.subheader(game["name"])

        m1, m2 = st.columns(2)
        m1.metric("🤖 マッチ度", f"{game.get('final_score', 0) * 100:.1f}%")
        m2.metric("👍 好評率", f"{game.get('rating_ratio', 0) * 100:.1f}%")

        st.caption(f"**🏷️ タグ:** {game.get('tags', '')}")
        st.write(f"🔗 [Steamで見る](https://store.steampowered.com/app/{appid}/)")
        st.code(str(appid), language="text")
        st.write("---")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("❌ ナシ", key=f"nashi_{side}_{idx}", use_container_width=True):
                add_nashi(app_user_id, appid)
                st.session_state[f"idx_{side}"] += 1
                st.rerun()
        with b2:
            if st.button("❤️ アリ", key=f"ari_{side}_{idx}", type="primary", use_container_width=True):
                st.session_state[f"keep_{side}"].append(game)
                st.session_state[f"idx_{side}"] += 1
                st.rerun()

        st.progress(idx / total, text=f"進行度: {idx + 1} / {total}")


col_a, col_b = st.columns(2)
_render_deck("A", col_a, pool_df)
_render_deck("B", col_b, pool_df)
