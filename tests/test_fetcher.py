"""fetcher.fetch_seed_data_parallel のタグ供給源の優先順位テスト。
ネットワークには一切出ない（SteamSpy/Steam Store はモックする）。
"""
from pipeline import fetcher


def test_static_tags_are_primary_source(monkeypatch):
    """対応表に載っている appid は SteamSpy を叩かないこと。"""
    called: list[int] = []

    def _fake_spy(appid: int) -> str:
        called.append(appid)
        return f"spy-tags-{appid}"

    monkeypatch.setattr(fetcher, "fetch_tags_with_cache", _fake_spy)
    monkeypatch.setattr(fetcher, "fetch_about_with_cache", lambda appid: "about")

    results = fetcher.fetch_seed_data_parallel(
        [111, 222], max_workers=1, static_tags={111: "Action,RPG"}
    )

    assert results[111]["tags"] == "Action,RPG"
    assert results[222]["tags"] == "spy-tags-222"
    assert called == [222]


def test_no_static_tags_falls_through_to_steamspy(monkeypatch):
    """static_tags 未指定（None）でも従来どおり SteamSpy 経路で動くこと。"""
    monkeypatch.setattr(fetcher, "fetch_tags_with_cache", lambda appid: f"spy-tags-{appid}")
    monkeypatch.setattr(fetcher, "fetch_about_with_cache", lambda appid: "about")

    results = fetcher.fetch_seed_data_parallel([333], max_workers=1)

    assert results[333]["tags"] == "spy-tags-333"
