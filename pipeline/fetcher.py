import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    import cloudscraper as _cloudscraper
    _steamspy_session = _cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "mobile": False}
    )
except Exception:
    _steamspy_session = requests.Session()

from db.crud import get_cached, set_cache

logger = logging.getLogger(__name__)

_STEAM_BASE = "http://api.steampowered.com"
_STORE_BASE = "https://store.steampowered.com/api"
_STEAMSPY_BASE = "https://steamspy.com/api.php"
_RATE_STEAM = 1.0      # 秒 / リクエスト
_RATE_STEAMSPY = 0.5   # 秒 / リクエスト


# ── 入力バリデーション ─────────────────────────────────────────────────────────

def validate_steam_id(steam_id: str, api_key: str) -> tuple[bool, str]:
    """Steam ID64（17桁）またはvanity URLを検証する。(is_valid, error_msg) を返す。"""
    steam_id = steam_id.strip()

    if not re.match(r"^\d{17}$", steam_id):
        # vanity URL として試みる
        try:
            url = f"{_STEAM_BASE}/ISteamUser/ResolveVanityURL/v0001/"
            r = requests.get(url, params={"key": api_key, "vanityurl": steam_id}, timeout=5)
            data = r.json().get("response", {})
            if data.get("success") == 1:
                return True, ""
            return False, "Steam IDまたはユーザー名が見つかりませんでした。"
        except requests.RequestException:
            return False, "Steam APIへの接続がタイムアウトしました。再度お試しください。"

    # 17桁数字の場合は GetPlayerSummaries で存在確認
    try:
        url = f"{_STEAM_BASE}/ISteamUser/GetPlayerSummaries/v0002/"
        r = requests.get(url, params={"key": api_key, "steamids": steam_id}, timeout=5)
        players = r.json().get("response", {}).get("players", [])
        if not players:
            return False, "このSteam IDは見つかりませんでした。"
        if players[0].get("communityvisibilitystate", 1) < 3:
            return False, "プロフィールが非公開です。公開設定にしてから再度お試しください。"
        return True, ""
    except requests.RequestException:
        return False, "Steam APIへの接続がタイムアウトしました。再度お試しください。"


# ── Steam API: ライブラリ・ウィッシュリスト・実績 ──────────────────────────────

def fetch_owned_games(steam_id: str, api_key: str) -> list[dict]:
    """プレイ時間 >= 600分 のゲームをプレイ時間降順 top10 で返す。"""
    url = f"{_STEAM_BASE}/IPlayerService/GetOwnedGames/v0001/"
    r = requests.get(
        url,
        params={"key": api_key, "steamid": steam_id, "include_appinfo": True, "format": "json"},
        timeout=10,
    )
    r.raise_for_status()
    games = r.json().get("response", {}).get("games", [])
    filtered = [g for g in games if g.get("playtime_forever", 0) >= 600]
    return sorted(filtered, key=lambda g: g["playtime_forever"], reverse=True)[:10]


def fetch_wishlist(steam_id: str, api_key: str) -> list[int]:
    """ウィッシュリストの appid リストを返す。"""
    url = "https://api.steampowered.com/IWishlistService/GetWishlist/v1/"
    try:
        r = requests.get(url, params={"steamid": steam_id}, timeout=10)
        items = r.json().get("response", {}).get("items", [])
        return [item["appid"] for item in items]
    except Exception as e:
        logger.warning("ウィッシュリスト取得失敗: %s", e)
        return []


def fetch_achievements(appid: int, steam_id: str, api_key: str) -> dict:
    """実績データを返す。取得失敗時は空の結果を返す（例外を上げない）。"""
    result = {"rarest_pct": None, "completion_rate": 0.0}
    try:
        ach_url = f"{_STEAM_BASE}/ISteamUserStats/GetPlayerAchievements/v0001/"
        global_url = f"{_STEAM_BASE}/ISteamUserStats/GetGlobalAchievementPercentagesForApp/v0002/"

        r_ach = requests.get(ach_url, params={"appid": appid, "key": api_key, "steamid": steam_id}, timeout=5)
        r_global = requests.get(global_url, params={"gameid": appid}, timeout=5)
        time.sleep(_RATE_STEAM)

        if r_ach.status_code != 200 or r_global.status_code != 200:
            return result

        achievements = r_ach.json().get("playerstats", {}).get("achievements", [])
        global_data = r_global.json().get("achievementpercentages", {}).get("achievements", [])
        global_pct = {a["name"]: float(a["percent"]) for a in global_data}

        unlocked = [a for a in achievements if a["achieved"] == 1]
        if achievements:
            result["completion_rate"] = len(unlocked) / len(achievements)

        rarest = min(
            (global_pct[a["apiname"]] for a in unlocked if a["apiname"] in global_pct),
            default=None,
        )
        result["rarest_pct"] = rarest
    except Exception as e:
        logger.warning("実績取得失敗 appid=%s: %s", appid, e)
    return result


def _calc_weight(playtime_minutes: int, rarest_pct: float | None, completion_rate: float) -> float:
    score = 0.0
    if rarest_pct is not None:
        if rarest_pct < 10.0:
            score += 5
        elif rarest_pct < 30.0:
            score += 3
        elif rarest_pct < 50.0:
            score += 2
    if completion_rate >= 0.70:
        score += 3
    elif completion_rate >= 0.40:
        score += 2
    elif completion_rate >= 0.10:
        score += 1
    hours = playtime_minutes / 60
    if hours >= 50.0:
        score += 2
    elif hours >= 10.0:
        score += 1
    return score


def fetch_seed_games(steam_id: str, api_key: str) -> list[dict]:
    """ライブラリ top10 + ウィッシュリスト を結合し weight_score 付きで返す。"""
    owned = fetch_owned_games(steam_id, api_key)
    wishlist_ids = fetch_wishlist(steam_id, api_key)
    time.sleep(_RATE_STEAM)

    seed = []
    seen = set()

    for g in owned:
        appid = g["appid"]
        ach = fetch_achievements(appid, steam_id, api_key)
        weight = _calc_weight(g["playtime_forever"], ach["rarest_pct"], ach["completion_rate"])
        seed.append({"appid": appid, "name": g.get("name", ""), "weight_score": weight})
        seen.add(appid)

    for appid in wishlist_ids:
        if appid not in seen:
            seed.append({"appid": appid, "name": "", "weight_score": 3.0})

    return seed


# ── SteamSpy + Steam Store API（キャッシュ付き）────────────────────────────────

def fetch_tags_with_cache(appid: int) -> str:
    """SteamSpy からタグを取得（DBキャッシュ TTL=7日）。カンマ区切り文字列を返す。
    Cloudflare でブロックされた場合は空文字を返す（呼び出し元が Kaggle CSV フォールバックを担う）。
    """
    cached = get_cached(appid, "steamspy_tags")
    if cached is not None:
        return cached.get("tags", "")

    try:
        r = _steamspy_session.get(
            _STEAMSPY_BASE, params={"request": "appdetails", "appid": appid}, timeout=10
        )
        time.sleep(_RATE_STEAMSPY)
        if r.status_code == 200:
            data = r.json()
            tags_str = ",".join(data.get("tags", {}).keys()) if isinstance(data.get("tags"), dict) else ""
            set_cache(appid, "steamspy_tags", {"tags": tags_str})
            return tags_str
        logger.debug("SteamSpy HTTP %s appid=%s（Kaggle CSV フォールバックへ）", r.status_code, appid)
    except Exception as e:
        logger.warning("SteamSpy取得失敗 appid=%s: %s", appid, e)
    return ""


def fetch_about_with_cache(appid: int) -> str:
    """Steam Store から About テキストを取得（DBキャッシュ TTL=7日）。HTMLタグ除去済みテキストを返す。"""
    cached = get_cached(appid, "steam_about")
    if cached is not None:
        return cached.get("about", "")

    try:
        r = requests.get(f"{_STORE_BASE}/appdetails", params={"appids": appid}, timeout=5)
        time.sleep(_RATE_STEAM)
        if r.status_code == 200:
            data = r.json().get(str(appid), {})
            if data.get("success"):
                html = data["data"].get("about_the_game", "")
                about = re.sub(r"<[^>]+>", " ", html).strip()
                set_cache(appid, "steam_about", {"about": about})
                return about
    except Exception as e:
        logger.warning("Steam Store取得失敗 appid=%s: %s", appid, e)
    return ""


def fetch_seed_data_parallel(
    seed_appids: list[int],
    max_workers: int = 3,
    static_tags: dict[int, str] | None = None,
) -> dict[int, dict]:
    """ThreadPoolExecutor でタグ + About を並列取得する。

    タグは static_tags（games.csv 由来の対応表）を一次供給源とし、
    載っていない appid（データセット以降の新作など）のみ SteamSpy を叩く。
    SteamSpy は Cloudflare にブロックされる環境があるため依存を最小化する。
    """
    results: dict[int, dict] = {}

    def _fetch_one(appid: int) -> tuple[int, dict]:
        tags = (static_tags or {}).get(appid, "")
        if not tags:
            tags = fetch_tags_with_cache(appid)
        return appid, {
            "tags": tags,
            "about": fetch_about_with_cache(appid),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, appid): appid for appid in seed_appids}
        for future in as_completed(futures):
            appid, data = future.result()
            results[appid] = data

    return results
