import time

import numpy as np
import pandas as pd
import requests
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("1. Kaggleデータから基礎カタログ取得...")

# 🏆 正しい40列のヘッダーをこちらで定義して強制上書きする
correct_cols = [
    "appid",
    "name",
    "release_date",
    "estimated_owners",
    "peak_ccu",
    "required_age",
    "price",
    "discount",
    "dlc_count",
    "about_the_game",
    "supported_languages",
    "full_audio_languages",
    "reviews",
    "header_image",
    "website",
    "support_url",
    "support_email",
    "windows",
    "mac",
    "linux",
    "metacritic_score",
    "metacritic_url",
    "user_score",
    "positive",
    "negative",
    "score_rank",
    "achievements",
    "recommendations",
    "notes",
    "average_playtime_forever",
    "average_playtime_two_weeks",
    "median_playtime_forever",
    "median_playtime_two_weeks",
    "developers",
    "publishers",
    "categories",
    "genres",
    "tags",
    "screenshots",
    "movies",
]

# header=0（元の1行目を無視）し、names=correct_cols で上書き
df_kaggle = pd.read_csv(
    "/content/drive/MyDrive/データ分析/データ分析/05_インテグレーションステップ/データ/archive/games.csv",
    header=0,
    names=correct_cols,
    engine="python",
    on_bad_lines="skip",
)

# ==========================================
# 2. データの浄化と原石抽出
# ==========================================
# AppIDとTagsの欠損を排除
df_kaggle["appid"] = pd.to_numeric(df_kaggle["appid"], errors="coerce")
df_kaggle = df_kaggle.dropna(subset=["appid", "tags"])
df_kaggle["appid"] = df_kaggle["appid"].astype(np.int64)

# レビューと好評率（今度こそ正しい列の数値が使われます！）
df_kaggle["positive"] = pd.to_numeric(df_kaggle["positive"], errors="coerce").fillna(0)
df_kaggle["negative"] = pd.to_numeric(df_kaggle["negative"], errors="coerce").fillna(0)
df_kaggle["total_reviews"] = df_kaggle["positive"] + df_kaggle["negative"]
df_kaggle["rating_ratio"] = df_kaggle.apply(
    lambda x: x["positive"] / x["total_reviews"] if x["total_reviews"] > 0 else 0,
    axis=1,
)

print("2. 泥臭いディグを開始。30〜300件の原石を抽出中...")
df_final_hidden = df_kaggle[
    (df_kaggle["total_reviews"] >= 30)
    & (df_kaggle["total_reviews"] <= 300)
    & (df_kaggle["rating_ratio"] >= 0.85)
    & (df_kaggle["supported_languages"].str.contains("Japanese", case=False, na=False))
].copy()
print(f" -> 抽出された本物の原石プール: {len(df_final_hidden)}件")

# ライブラリから抽出！

import time

import pandas as pd
import requests

# ==========================================
# 準備：ご自身のAPIキーとSteam IDを入力してください
# ==========================================
API_KEY = "883FA8F6BF390E55DADDB71BFAA4BF3C"
STEAM_ID = "76561199040391831"

# ==========================================
# 1. 所持ゲームとプレイ時間の取得
# ==========================================
print("プレイ履歴を取得中...")
owned_games_url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
params_owned = {
    "key": API_KEY,
    "steamid": STEAM_ID,
    "include_appinfo": True,
    "format": "json",
}

response = requests.get(owned_games_url, params=params_owned)
data = response.json()
games = data.get("response", {}).get("games", [])

if not games:
    print(
        "ゲームが見つかりませんでした。プロフィールが非公開になっている可能性があります。"
    )
else:
    # ==========================================
    # 2. ノイズ排除（プレイ時間10時間未満を除外）
    # ==========================================
    # playtime_forever は「分」単位なので 600分以上でフィルター
    filtered_games = [g for g in games if g.get("playtime_forever", 0) >= 600]

    # ==========================================
    # 3. プレイ時間降順でソートし、上位10件を取得
    # ==========================================
    top_10_games = sorted(
        filtered_games, key=lambda x: x["playtime_forever"], reverse=True
    )[:10]

    print(f"総所持ゲーム数: {len(games)}本")
    print(f"10時間以上プレイしたゲーム数: {len(filtered_games)}本")
    print("上位10件の実績データを取得します...\n")

    # ==========================================
    # 4. 上位10件に対する実績データの取得
    # ==========================================
    seed_data = []  # 最終的なデータを格納するリスト

    for game in top_10_games:
        appid = game["appid"]
        name = game["name"]
        playtime_hours = round(game["playtime_forever"] / 60, 1)

        print(f"[{name}] (プレイ時間: {playtime_hours}h) のデータを取得中...")

        # プレイヤーの個人の実績解除状況を取得
        achievements_url = (
            "http://api.steampowered.com/ISteamUserStats/GetPlayerAchievements/v0001/"
        )
        params_ach = {"appid": appid, "key": API_KEY, "steamid": STEAM_ID}
        res_ach = requests.get(achievements_url, params=params_ach)

        # グローバルの実績取得率（難易度）を取得
        global_url = "http://api.steampowered.com/ISteamUserStats/GetGlobalAchievementPercentagesForApp/v0002/"
        params_global = {"gameid": appid}
        res_global = requests.get(global_url, params=params_global)

        # 実績機能が存在しないゲームや、取得エラーの場合はスキップ
        if res_ach.status_code != 200 or res_global.status_code != 200:
            print(f"  -> 実績データなし、または取得エラーのためスキップ\n")
            seed_data.append(
                {
                    "appid": appid,
                    "name": name,
                    "playtime_hours": playtime_hours,
                    "unlocked_achievements": 0,
                    "total_achievements": 0,
                    "rarest_achievement_pct": None,  # レア実績なし
                }
            )
            time.sleep(1)  # API制限回避
            continue

        ach_data = res_ach.json().get("playerstats", {})
        global_data = (
            res_global.json().get("achievementpercentages", {}).get("achievements", [])
        )

        # 個人の解除実績リスト
        achievements = ach_data.get("achievements", [])
        unlocked = [a for a in achievements if a["achieved"] == 1]

        # グローバル取得率を辞書型にして検索しやすくする { "実績名": パーセンテージ }
        global_pct_dict = {a["name"]: a["percent"] for a in global_data}

        # 解除した実績の中で、最もグローバル取得率が低い（レアな）ものを探す
        rarest_pct = 100.0
        for ach in unlocked:
            ach_name = ach["apiname"]
            if ach_name in global_pct_dict:
                # ★修正：APIが文字列で返してくる場合があるため、強制的にfloat（小数）に変換する
                pct = float(global_pct_dict[ach_name])
                if pct < rarest_pct:
                    rarest_pct = pct

        if rarest_pct == 100.0:
            rarest_pct = None  # 解除した実績がない場合

        # データをリストに追加
        seed_data.append(
            {
                "appid": appid,
                "name": name,
                "playtime_hours": playtime_hours,
                "unlocked_achievements": len(unlocked),
                "total_achievements": len(achievements),
                "rarest_achievement_pct": rarest_pct,
            }
        )

        # Steam APIへのDoS攻撃判定を避けるため1秒待機
        time.sleep(1)
        print(f"  -> 完了\n")

    # ==========================================
    # 結果をPandasデータフレームにして表示
    # ==========================================
    df_history = pd.DataFrame(seed_data)

    # 達成率を計算して列を追加
    df_history["completion_rate"] = df_history.apply(
        lambda x: (
            round((x["unlocked_achievements"] / x["total_achievements"]) * 100, 1)
            if x["total_achievements"] > 0
            else 0
        ),
        axis=1,
    )

    # 見やすいようにカラムの並び替え
    display_cols = [
        "name",
        "playtime_hours",
        "completion_rate",
        "rarest_achievement_pct",
        "unlocked_achievements",
        "total_achievements",
    ]
    print("\n=== あなたの基礎シード候補（上位10件） ===")
    display(df_history[display_cols])

    # ==========================================
    # 1. 基礎シードに「加点スコア（重み）」を付与する
    # ==========================================
    def calculate_weight(row):
        score = 0
        # ① レア実績スコア (Max 5)
        rare = row["rarest_achievement_pct"]
        if pd.notna(rare):
            if rare < 10.0:
                score += 5
            elif rare < 30.0:
                score += 3
            elif rare < 50.0:
                score += 2

        # ② 全体解除率スコア (Max 3)
        comp = row["completion_rate"]
        if comp >= 70.0:
            score += 3
        elif comp >= 40.0:
            score += 2
        elif comp >= 10.0:
            score += 1

        # ③ プレイ時間スコア (Max 2)
        hours = row["playtime_hours"]
        if hours >= 50.0:
            score += 2
        elif hours >= 10.0:
            score += 1

        return score

    df_history["weight_score"] = df_history.apply(calculate_weight, axis=1)

    print("=== プレイ履歴上の各ゲームの算出スコア（MAX10点） ===")
    display(
        df_history[["name", "weight_score"]].sort_values(
            "weight_score", ascending=False
        )
    )

    # ==========================================
    # 1.5.プレイ履歴からのリストに、ウィッシュリストを追加する
    # ==========================================

    print("📡 Steam APIからウィッシュリストを取得中...")

    # ✨ 廃止された旧URLに代わり、新設された公式のウィッシュリストAPIを使用します
    wishlist_url = "https://api.steampowered.com/IWishlistService/GetWishlist/v1/"
    params_wishlist = {
        "steamid": STEAM_ID,
    }

    wishlist_appids = []

    try:
        # APIを叩く
        res = requests.get(wishlist_url, params=params_wishlist, timeout=10)

        if res.status_code == 200:
            data = res.json()
            # 新APIは "response": {"items": [{"appid": 12345, "priority": 1}, ...]} の構造
            items = data.get("response", {}).get("items", [])

            if items:
                wishlist_appids = [item["appid"] for item in items]
                print(
                    f"✅ ウィッシュリストから {len(wishlist_appids)} 件の未来の欲望を取得しました！"
                )
            else:
                print("⚠️ ウィッシュリストが0件、または取得できませんでした。")
        else:
            print(f"⚠️ サーバーエラー: {res.status_code}")

    except Exception as e:
        print(f"🚨 API通信エラー: {e}")

    # ウィッシュリスト用のデータフレームを作成し、重みを「3.0」で固定
    df_wishlist = pd.DataFrame(
        {
            "appid": wishlist_appids,
            "weight_score": 3.0,  # 優先度：中のフック
        }
    )

    # 合体
    df_integrated_seed = pd.concat([df_history, df_wishlist], ignore_index=True)

    df_seed = df_integrated_seed.sort_values(
        "weight_score", ascending=False
    ).drop_duplicates(subset="appid", keep="first")

    print(f"🔥 プレイ履歴とウィッシュリストの統合完了！シード総数: {len(df_seed)}件")

    # ==========================================
    # 2.SteamSpyを使って基礎シードベクトル作成
    # ==========================================

    print("3. あなたの個人シードから魂のベクトルを構築中...")

    # まず、12万件のKaggleカタログ全体のタグを使って「言葉の辞書（空間）」を作る
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split(",") if isinstance(x, str) else [],
        token_pattern=None,
    )
    vectorizer.fit(df_kaggle["tags"])
    tfidf_matrix_hidden = vectorizer.transform(df_final_hidden["tags"])

    # ベースとなる空のベクトルを用意
    user_profile_vector = np.zeros(len(vectorizer.get_feature_names_out()))
    seed_appids = df_seed["appid"].tolist()

    # ★ ここでSteamSpyが挟まる！
    # 原石プールの中を探すのではなく、APIに直接タグを聞きに行く
    # 「タグの集合体」をベクトル化している
    for app_id, weight in zip(df_seed["appid"], df_seed["weight_score"]):
        try:
            res = requests.get(
                f"https://steamspy.com/api.php?request=appdetails&appid={app_id}",
                timeout=5,
            )
            if res.status_code == 200:
                data = res.json()
                if "tags" in data and isinstance(data["tags"], dict):
                    # SteamSpyから取得したタグをカンマ区切りの文字列にする
                    tags_str = ",".join(data["tags"].keys())
                    # それをベクトル化して重みを掛ける
                    game_vec = vectorizer.transform([tags_str]).toarray()[0]
                    user_profile_vector += game_vec * weight
        except Exception:
            pass
        time.sleep(0.5)

    # 正規化（ベクトルの長さを1に）する
    from numpy.linalg import norm

    if norm(user_profile_vector) > 0:
        user_profile_vector = user_profile_vector / norm(user_profile_vector)

        import re
        import time

        import numpy as np
        import requests

        print(
            "1. 共通ベクトル空間（Tags用）の構築と、APIからのデータ収集を開始します..."
        )

        # ① Tags用ベクトル空間の構築（TagはV1もV2も共通でTF-IDF）
        vectorizer_tags = TfidfVectorizer(
            tokenizer=lambda x: x.split(",") if isinstance(x, str) else [],
            token_pattern=None,
        )
        tfidf_matrix_tags = vectorizer_tags.fit_transform(df_final_hidden["tags"])

        # ② Kaggle側のAboutテキストのお掃除（V1, V2共通の準備）
        df_final_hidden["about_the_game"] = df_final_hidden["about_the_game"].fillna("")
        df_final_hidden["about_clean"] = df_final_hidden["about_the_game"].apply(
            lambda x: re.sub(r"<[^>]+>", " ", str(x)).strip()
        )

        # ③ シードからAPIを使ってデータを集める
        user_vector_tags = np.zeros(len(vectorizer_tags.get_feature_names_out()))
        seed_about_texts = []  # 👈 Aboutの文章をそのまま貯める箱
        seed_weights = []  # 👈 そのゲームの重みを貯める箱

        for app_id, weight in zip(df_seed["appid"], df_seed["weight_score"]):
            # --- A. Tagsの取得 (SteamSpy API) ---
            try:
                res_tags = requests.get(
                    f"https://steamspy.com/api.php?request=appdetails&appid={app_id}",
                    timeout=5,
                )
                if res_tags.status_code == 200:
                    data_tags = res_tags.json()
                    if "tags" in data_tags and isinstance(data_tags["tags"], dict):
                        tags_str = ",".join(data_tags["tags"].keys())
                        user_vector_tags += (
                            vectorizer_tags.transform([tags_str]).toarray()[0] * weight
                        )
            except Exception:
                pass

            # --- B. Aboutの取得 (Steam公式 Store API) ---
            try:
                res_about = requests.get(
                    f"https://store.steampowered.com/api/appdetails?appids={app_id}",
                    timeout=5,
                )
                if res_about.status_code == 200:
                    data_about = res_about.json()
                    if str(app_id) in data_about and data_about[str(app_id)].get(
                        "success"
                    ):
                        about_html = data_about[str(app_id)]["data"].get(
                            "about_the_game", ""
                        )
                        about_text = re.sub(r"<[^>]+>", " ", about_html).strip()
                        if about_text:
                            # 💡 ベクトル化せず、テキストと重みをリストに保存！
                            seed_about_texts.append(about_text)
                            seed_weights.append(weight)
            except Exception:
                pass

            time.sleep(1.0)

        # Tag側だけはここで完成させて正規化しておく
        from numpy.linalg import norm

        if norm(user_vector_tags) > 0:
            user_vector_tags = user_vector_tags / norm(user_vector_tags)

        print(
            f"✅ 共通データの取得完了！ 貯まったAboutテキスト: {len(seed_about_texts)}件"
        )

    import numpy as np
    from numpy.linalg import norm
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    print("【Ver 2.0】情熱（About）をBERTで文脈ベクトル化し、強化版を作成します...")

    # ==========================================
    # 1. About用のBERT空間を構築
    # ==========================================
    # 英語の文脈を理解できる軽量・高速な事前学習済みモデルをロード
    # （※初回実行時のみモデルのダウンロードが走ります）
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("原石プール全件の情熱（テキスト）をBERT空間にマッピング中...")
    # 共通ブロックで作ったお掃除済みテキスト（about_clean）を一気に384次元のベクトルに変換
    embeddings_about = model.encode(df_final_hidden["about_clean"].tolist())

    # ==========================================
    # 2. 貯金しておいたテキストからユーザーベクトルを錬成
    # ==========================================
    # BERTの出力次元数（384次元）の空ベクトルを用意
    user_vector_about_bert = np.zeros(model.get_sentence_embedding_dimension())

    # 共通ブロックで取得済みのテキストと重みを回す（API通信なし！）
    for text, weight in zip(seed_about_texts, seed_weights):
        if text:
            # TF-IDFではなく、文脈を汲み取れるBERTでエンコードして加算
            user_vector_about_bert += model.encode(text) * weight

    # 正規化（長さを1に揃える）
    if norm(user_vector_about_bert) > 0:
        user_vector_about_bert = user_vector_about_bert / norm(user_vector_about_bert)

    # ==========================================
    # 3. マッチング処理とUI用CSV出力
    # ==========================================
    print("類似度を計算し、出力ファイルを作成します...")

    # 類似度計算
    # Tag側は共通ブロックで作ったものをそのまま使用（V1と完全なフェアプレイ）
    df_final_hidden["sim_tags"] = cosine_similarity(
        [user_vector_tags], tfidf_matrix_tags
    )[0]
    # About側はBERT空間での類似度を使用
    df_final_hidden["sim_about"] = cosine_similarity(
        [user_vector_about_bert], embeddings_about
    )[0]

    # シード以外の全件をベースとして切り出し（コピーして大元を汚さない安全策）
    df_base2 = df_final_hidden[~df_final_hidden["appid"].isin(seed_appids)].copy()

    # StreamlitのUIで必要な列だけを抽出（Ver 1.0と完全に同じスキーマ）
    export_cols = [
        "appid",
        "name",
        "sim_tags",
        "sim_about",
        "tags",
        "movies",
        "positive",
        "negative",
        "total_reviews",
        "rating_ratio",
        "about_the_game",
    ]

    # Ver2.0（BERT搭載版）として保存
    df_base2[export_cols].to_csv("recommendation_base_2.csv", index=False)
    print("💾 強化版（BERT搭載）の 'recommendation_base_2.csv' を保存しました！")

    # ↑ここまでが初期状態↑
    # ⇩ここからは、画面UIフィードバックを受けての再計算用⇩

    import numpy as np
    from numpy.linalg import norm
    from sklearn.metrics.pairwise import cosine_similarity

    # ==========================================
    # 🛑 ナシ判定（ブラックリスト）の確認
    # ==========================================
    # ※Ver 3.0のセルで変数 `rejected_appids` が定義されていればそのまま使います
    if "rejected_appids" not in locals():
        rejected_appids = []

        print("⚠️ ナシ判定リストが空です。必要に応じてAppIDを入力してください。")

     # ==========================================
     # 🛑 ナシ判定（ブラックリスト）の入力
     # ==========================================
     # Streamlit画面の「ナシ判定用コピーボタン」からコピーしたAppIDをここに貼り付けてください
     rejected_appids = [
         2696760,
         1154090,
         1261430,
         1372000,
         968800,
         903850,
         3199390,
         3201210,
         2529520,
     ]
    print(
        f"【Ver 4.0】ナシ判定（{len(rejected_appids)}件）を「直交射影」でスマートに除去し、物理的にも除外します..."
    )

    # ==========================================
    # 1. 直交射影による「嫌いな成分の完全無効化」
    # ==========================================
    # 共通ブロックの純粋なDNAベクトルをコピーして使用
    user_vector_tags_v4 = user_vector_tags.copy()
    user_vector_about_v4 = user_vector_about_bert.copy()

    for app_id in rejected_appids:
        idx_list = df_final_hidden[df_final_hidden["appid"] == app_id].index
        if len(idx_list) > 0:
            idx = df_final_hidden.index.get_loc(idx_list[0])

            # --- A. Tags（TF-IDF）空間での直交射影 ---
            game_tag_vec = tfidf_matrix_tags[idx].toarray()[0]
            norm_tag = norm(game_tag_vec)
            if norm_tag > 0:
                # ゲームのベクトルを長さ1（単位ベクトル）に揃える
                game_tag_unit = game_tag_vec / norm_tag
                # 内積（どれくらい成分が被っているか）を計算し、その分だけピンポイントで引く
                dot_product = np.dot(user_vector_tags_v4, game_tag_unit)
                user_vector_tags_v4 -= dot_product * game_tag_unit

            # --- B. About（BERT）空間での直交射影 ---
            game_about_vec = embeddings_about[idx]
            norm_about = norm(game_about_vec)
            if norm_about > 0:
                # ゲームのベクトルを長さ1（単位ベクトル）に揃える
                game_about_unit = game_about_vec / norm_about
                # 内積（どれくらい成分が被っているか）を計算し、その分だけピンポイントで引く
                dot_product = np.dot(user_vector_about_v4, game_about_unit)
                user_vector_about_v4 -= dot_product * game_about_unit

    # 引き算によって短くなったベクトルの「長さ」を1に再正規化
    if norm(user_vector_tags_v4) > 0:
        user_vector_tags_v4 = user_vector_tags_v4 / norm(user_vector_tags_v4)
    if norm(user_vector_about_v4) > 0:
        user_vector_about_v4 = user_vector_about_v4 / norm(user_vector_about_v4)

    # ==========================================
    # 2. 洗練されたベクトルで再マッチング
    # ==========================================
    print("直交射影後の洗練されたベクトルで、全原石との類似度を再計算中...")

    df_final_hidden["sim_tags"] = cosine_similarity(
        [user_vector_tags_v4], tfidf_matrix_tags
    )[0]
    df_final_hidden["sim_about"] = cosine_similarity(
        [user_vector_about_v4], embeddings_about
    )[0]

    # シードを切り出し（大元を汚さない）
    df_base4 = df_final_hidden[~df_final_hidden["appid"].isin(seed_appids)].copy()

    # ==========================================
    # 🛑 3. 【V4のキモ】ナシ判定のゲームをリストから物理的に除外（Drop）
    # ==========================================
    if len(rejected_appids) > 0:
        df_base4 = df_base4[~df_base4["appid"].isin(rejected_appids)].copy()
        print(
            f"🗑️ ナシ判定された {len(rejected_appids)} 件のゲームをリストから完全に除外しました。"
        )

    # StreamlitのUIで必要な列だけを抽出（スキーマは統一）
    export_cols = [
        "appid",
        "name",
        "sim_tags",
        "sim_about",
        "tags",
        "movies",
        "positive",
        "negative",
        "total_reviews",
        "rating_ratio",
        "about_the_game",
    ]

    # Ver4.0（直交射影＋物理Drop版）として保存！
    df_base4[export_cols].to_csv("recommendation_base_4.csv", index=False)
    print("💾 究極の調整版 'recommendation_base_4.csv' を保存しました！")
