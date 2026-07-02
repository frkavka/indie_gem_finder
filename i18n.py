import streamlit as st

_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # ── setup ──────────────────────────────────────────────────────────
        "page_title_setup": "Indie Gem Finder — Setup",
        "title": "💎 Indie Gem Finder",
        "subtitle": "Discover hidden indie gems based on your Steam play history.",
        "steam_id_label": "🎮 Steam ID (17-digit number or username)",
        "steam_id_placeholder": "e.g. 76561199040391831",
        "section_params": "🎛️ Recommendation Parameters",
        "review_min_label": "Min review count",
        "review_max_label": "Max review count",
        "positive_rate_label": "Min positive rate (%)",
        "review_range_error": "Min review count must be less than max.",
        "params_locked_note": "🔒 Filters are fixed to keep server load manageable. The pool covers games with 30–300 reviews and ≥85% positive rating with Japanese localization.",
        "section_ab": "⚖️ A/B Model Initial Blend",
        "model_a_label": "Model A — tag ratio",
        "model_b_label": "Model B — tag ratio",
        "blend_caption": "Tags {tag:.0f}% / Prose {prose:.0f}%",
        "btn_generate": "🚀 Generate Recommendations",
        "error_no_steam_id": "Please enter a Steam ID.",
        "spinner_validating": "Validating Steam ID...",
        "warn_already_running": "Already computing. Please wait.",
        # ── computing ──────────────────────────────────────────────────────
        "page_title_computing": "Computing... — Indie Gem Finder",
        "computing_title": "⚙️ Computing recommendations...",
        "computing_caption": "Running Steam API calls and AI vector matching. Please wait.",
        "warn_no_job": "No computation job found.",
        "btn_restart": "Start over",
        "progress_caption": "Progress: {pct:.0f}%",
        "success_complete": "Done! Loading your deck...",
        "error_occurred": "An error occurred: {msg}",
        "error_hint": "Try adjusting the parameters and running again.",
        "btn_retry": "🔄 Start over",
        # ── deck ───────────────────────────────────────────────────────────
        "page_title_deck": "Indie Gem Finder",
        "deck_title": "⚖️ Indie Gem Finder — A/B Dual Deck",
        "deck_caption": "Compare two AI blends side-by-side to find your hidden gems.",
        "nashi_threshold_info": "You've rated {n} games as 'Nope'. You can recompute to refine results.",
        "btn_recompute": "🔄 Recompute with your taste",
        "model_label": "🤖 Model {side}",
        "slider_label": "Tags ↔ Prose [{side}]",
        "all_seen": "🎉 You've reviewed all cards in Model {side}!",
        "keep_list_header": "**❤️ Liked list**",
        "no_keeps": "You didn't mark any games as liked.",
        "btn_reset_deck": "🔄 Restart Model {side}",
        "deck_closing": "Found any good vibes? Happy gaming! 🎮",
        "match_metric": "🤖 Match",
        "rating_metric": "👍 Positive rate",
        "tags_label": "**🏷️ Tags:** {tags}",
        "steam_link": "🔗 [View on Steam](https://store.steampowered.com/app/{appid}/)",
        "btn_nashi": "❌ Nope",
        "btn_ari": "❤️ Like",
        "progress_text": "Progress: {idx} / {total}",
        # ── language selector ──────────────────────────────────────────────
        "lang_label": "🌐 Language",
    },
    "ja": {
        # ── setup ──────────────────────────────────────────────────────────
        "page_title_setup": "Indie Gem Finder — セットアップ",
        "title": "💎 Indie Gem Finder",
        "subtitle": "あなたのSteamプレイ履歴から、埋もれた名作インディーゲームを発見します。",
        "steam_id_label": "🎮 Steam ID（17桁の数字またはユーザー名）",
        "steam_id_placeholder": "例: 76561199040391831",
        "section_params": "🎛️ レコメンドパラメータ",
        "review_min_label": "レビュー件数 下限",
        "review_max_label": "レビュー件数 上限",
        "positive_rate_label": "好評率 閾値 (%)",
        "review_range_error": "レビュー件数の下限は上限より小さくしてください。",
        "params_locked_note": "🔒 サーバー負荷軽減のため、フィルター条件は固定しています。レビュー件数 30〜300件・好評率 85%以上・日本語対応ゲームを対象にしています。",
        "section_ab": "⚖️ A/B モデルの初期ブレンド設定",
        "model_a_label": "モデルA タグ比率",
        "model_b_label": "モデルB タグ比率",
        "blend_caption": "タグ {tag:.0f}% / ポエム {prose:.0f}%",
        "btn_generate": "🚀 レコメンドを生成",
        "error_no_steam_id": "Steam IDを入力してください。",
        "spinner_validating": "Steam IDを確認中...",
        "warn_already_running": "計算中です。しばらくお待ちください。",
        # ── computing ──────────────────────────────────────────────────────
        "page_title_computing": "計算中... — Indie Gem Finder",
        "computing_title": "⚙️ レコメンドを計算中...",
        "computing_caption": "Steam APIとAIベクトル計算を実行しています。このまましばらくお待ちください。",
        "warn_no_job": "計算ジョブが見つかりません。",
        "btn_restart": "最初からやり直す",
        "progress_caption": "進捗: {pct:.0f}%",
        "success_complete": "計算完了！カードを表示します...",
        "error_occurred": "エラーが発生しました: {msg}",
        "error_hint": "パラメータを調整して再度お試しください。",
        "btn_retry": "🔄 最初からやり直す",
        # ── deck ───────────────────────────────────────────────────────────
        "page_title_deck": "Indie Gem Finder",
        "deck_title": "⚖️ Indie Gem Finder — A/B デュアルデッキ",
        "deck_caption": "左右で異なるAIブレンドを比較しながら原石を発掘しましょう。",
        "nashi_threshold_info": "ナシ判定が {n} 件溜まりました。好みを反映して再計算できます。",
        "btn_recompute": "🔄 好みを反映してもう一度",
        "model_label": "🤖 モデル {side}",
        "slider_label": "タグ ↔ ポエム [{side}]",
        "all_seen": "🎉 モデル {side} のデッキを全件確認しました！",
        "keep_list_header": "**❤️ アリリスト**",
        "no_keeps": "アリ判定したゲームはありませんでした。",
        "btn_reset_deck": "🔄 モデル {side} をやり直す",
        "deck_closing": "良さげなゲームは見つかった？ 楽しいゲーマーライフを！ 🎮",
        "match_metric": "🤖 マッチ度",
        "rating_metric": "👍 好評率",
        "tags_label": "**🏷️ タグ:** {tags}",
        "steam_link": "🔗 [Steamで見る](https://store.steampowered.com/app/{appid}/)",
        "btn_nashi": "❌ ナシ",
        "btn_ari": "❤️ アリ",
        "progress_text": "進行度: {idx} / {total}",
        # ── language selector ──────────────────────────────────────────────
        "lang_label": "🌐 言語",
    },
}

_SUPPORTED = {"en": "English", "ja": "日本語"}


def _detect_browser_lang() -> str:
    """Accept-Language ヘッダーから初期言語を推定する（Streamlit 1.37+）。"""
    try:
        accept = st.context.headers.get("Accept-Language", "en")
        code = accept.split(",")[0].split("-")[0].lower()
        return code if code in _SUPPORTED else "en"
    except Exception:
        return "en"


def init_lang() -> None:
    """セッション開始時に一度だけ呼ぶ。session_state["lang"] を初期化する。"""
    if "lang" not in st.session_state:
        st.session_state["lang"] = _detect_browser_lang()


def render_lang_selector() -> None:
    """サイドバーに言語セレクタを描画する。全ページで呼ぶ。"""
    init_lang()
    current = st.session_state["lang"]
    options = list(_SUPPORTED.keys())
    labels = list(_SUPPORTED.values())
    selected_label = st.sidebar.selectbox(
        _STRINGS[current]["lang_label"],
        labels,
        index=options.index(current),
        key="_lang_selector",
    )
    selected_code = options[labels.index(selected_label)]
    if selected_code != current:
        st.session_state["lang"] = selected_code
        st.rerun()


def t(key: str, **kwargs) -> str:
    """現在の言語でキーに対応する文字列を返す。フォーマット引数は kwargs で渡す。"""
    lang = st.session_state.get("lang", "en")
    s = _STRINGS.get(lang, _STRINGS["en"]).get(key, key)
    return s.format(**kwargs) if kwargs else s
