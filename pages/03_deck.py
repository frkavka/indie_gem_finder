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
from i18n import render_lang_selector, t
from pipeline.runner import deserialize_result, run_recompute_light

st.set_page_config(page_title="Indie Gem Finder", layout="wide", page_icon="💎")

app_user_id = st.session_state.get("app_user_id")
if not app_user_id:
    st.switch_page("app.py")

job = get_job(app_user_id)
if not job or job["status"] != "complete" or not job["result_json"]:
    st.switch_page("app.py")

render_lang_selector()

result = deserialize_result(job["result_json"])
pool_df = pd.DataFrame(result["pool"])
params = load_params(app_user_id)
nashi_count = count_nashi(app_user_id)

st.title(t("deck_title"))
st.caption(t("deck_caption"))

if nashi_count >= 20:
    st.info(t("nashi_threshold_info", n=nashi_count))
    if st.button(t("btn_recompute"), type="primary", use_container_width=True):
        try:
            with st.spinner(t("recompute_spinner")):
                run_recompute_light(app_user_id)
        except Exception as e:
            st.error(t("error_occurred", msg=str(e)))
            st.stop()
        for side in ["A", "B"]:
            st.session_state[f"idx_{side}"] = 0
            st.session_state[f"keep_{side}"] = []
        st.rerun()

st.divider()

for side in ["A", "B"]:
    if f"idx_{side}" not in st.session_state:
        st.session_state[f"idx_{side}"] = 0
        st.session_state[f"keep_{side}"] = []
        st.session_state[f"weight_{side}"] = params[f"weight_tags_{side.lower()}"]


def _render_deck(side: str, col, pool_df: pd.DataFrame) -> None:
    with col:
        st.subheader(t("model_label", side=side))

        weight = st.slider(
            t("slider_label", side=side),
            0.0, 1.0,
            st.session_state[f"weight_{side}"],
            step=0.1,
            key=f"slider_{side}",
        )
        st.caption(t("blend_caption", tag=weight * 100, prose=(1 - weight) * 100))

        if st.session_state[f"weight_{side}"] != weight:
            st.session_state[f"weight_{side}"] = weight
            st.session_state[f"idx_{side}"] = 0
            st.rerun()

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
            st.success(t("all_seen", side=side))
            keeps = st.session_state[f"keep_{side}"]
            if keeps:
                st.write(t("keep_list_header"))
                for g in keeps:
                    st.write(f"- [{g['name']}](https://store.steampowered.com/app/{g['appid']}/)")
            else:
                st.info(t("no_keeps"))
            if st.button(t("btn_reset_deck", side=side), key=f"reset_{side}", use_container_width=True):
                st.session_state[f"idx_{side}"] = 0
                st.session_state[f"keep_{side}"] = []
                st.rerun()
            st.divider()
            st.markdown(f"#### {t('deck_closing')}")
            components.html(
                """
                <a href="https://buymeacoffee.com/ashoe" target="_blank">
                  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png"
                       alt="Buy Me A Coffee"
                       style="height:50px;width:auto;">
                </a>
                """,
                height=60,
            )
            return

        game = games[idx]
        appid = int(game["appid"])
        img_url = f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{appid}/header.jpg"
        movies_raw = str(game.get("movies", ""))
        video_url = next(
            (u.strip() for u in movies_raw.split(",") if u.strip().startswith("http")),
            "",
        )

        with st.container(height=500, border=False):
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
                components.html(
                    f'<img src="{img_url}" style="width:100%;height:200px;object-fit:cover;border-radius:8px;">',
                    height=210,
                )

            st.subheader(game["name"])

            m1, m2 = st.columns(2)
            m1.metric(t("match_metric"), f"{game.get('final_score', 0) * 100:.1f}%")
            m2.metric(t("rating_metric"), f"{game.get('rating_ratio', 0) * 100:.1f}%")

            st.caption(t("tags_label", tags=game.get("tags", "")))
            st.write(t("steam_link", appid=appid))
            st.code(str(appid), language="text")

        st.divider()
        b1, b2 = st.columns(2)
        with b1:
            if st.button(t("btn_nashi"), key=f"nashi_{side}_{idx}", use_container_width=True):
                add_nashi(app_user_id, appid)
                st.session_state[f"idx_{side}"] += 1
                st.rerun()
        with b2:
            if st.button(t("btn_ari"), key=f"ari_{side}_{idx}", type="primary", use_container_width=True):
                st.session_state[f"keep_{side}"].append(game)
                st.session_state[f"idx_{side}"] += 1
                st.rerun()

        st.progress(idx / total, text=t("progress_text", idx=idx + 1, total=total))


col_a, col_b = st.columns(2)
_render_deck("A", col_a, pool_df)
_render_deck("B", col_b, pool_df)
