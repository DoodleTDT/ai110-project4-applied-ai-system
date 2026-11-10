import pytest
import requests

from src import config
from src.config import MissingAPIKeyError
from src.database import SongDatabase
from src.external import (USER_AGENT, DeezerClient, DeezerMatch, ExternalAPIError,
                          LastFmClient, comparable, is_alternate_version)


class FakeResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


class FakeSession:
    """Answers requests with handler(url, params) -> data or (data, status); records every call."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout, "headers": headers})
        result = self.handler(url, params or {})
        if isinstance(result, Exception):
            raise result
        data, status = result if isinstance(result, tuple) else (result, 200)
        return FakeResponse(data, status)


@pytest.fixture
def db(tmp_path):
    database = SongDatabase(str(tmp_path / "test.db3"))
    yield database
    database.close()


def lastfm(handler, db=None):
    session = FakeSession(handler)
    return LastFmClient(api_key="test-key", db=db, session=session, min_interval=0), session


def deezer(handler, db=None):
    session = FakeSession(handler)
    return DeezerClient(db=db, session=session, min_interval=0), session


# --- comparable ---

@pytest.mark.parametrize("a, b", [
    ("Bohemian Rhapsody", "bohemian rhapsody"),
    ("Hey Jude (Remastered 2015)", "Hey Jude"),
    ("Song [Live]", "Song"),
    ("Song - 2011 Remaster", "Song"),
    ("The Beatles", "Beatles"),
    ("Don't Stop", "Dont Stop"),
])
def test_comparable_ignores_case_suffixes_and_punctuation(a, b):
    assert comparable(a).replace(" ", "") == comparable(b).replace(" ", "")


def test_comparable_keeps_different_titles_apart():
    assert comparable("Yesterday") != comparable("Yesterday Once More")


# --- Last.fm ---

def test_lastfm_requires_api_key(monkeypatch):
    monkeypatch.setattr(config, "_loaded", True)  # don't read the real env files
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        LastFmClient()


def test_lastfm_reads_api_key_from_env(monkeypatch):
    monkeypatch.setattr(config, "_loaded", True)
    monkeypatch.setenv("LASTFM_API_KEY", "from-env")
    assert LastFmClient(session=FakeSession(lambda u, p: {})).api_key == "from-env"


def test_tag_top_tracks_parses_tracks():
    client, session = lastfm(lambda url, params: {"tracks": {"track": [
        {"name": "Creep", "mbid": "m1", "artist": {"name": "Radiohead", "mbid": "a1"}},
        {"name": "Hurt", "mbid": "", "artist": {"name": "Johnny Cash"}},
    ]}})
    tracks = client.tag_top_tracks(" Rock ", limit=2)

    assert tracks == [
        {"title": "Creep", "artist": "Radiohead", "mbid": "m1"},
        {"title": "Hurt", "artist": "Johnny Cash", "mbid": None},
    ]
    params = session.calls[0]["params"]
    assert params["method"] == "tag.getTopTracks"
    assert params["tag"] == "rock"
    assert params["api_key"] == "test-key"
    assert params["format"] == "json"


def test_requests_use_generic_user_agent_and_timeout():
    client, session = lastfm(lambda url, params: {"tracks": {"track": []}})
    client.tag_top_tracks("rock")
    assert session.calls[0]["headers"] == {"User-Agent": USER_AGENT}
    assert session.calls[0]["timeout"] > 0


def test_single_track_object_is_treated_as_a_list():
    client, _ = lastfm(lambda url, params: {"tracks": {"track": {"name": "Only", "artist": {"name": "One"}}}})
    assert client.tag_top_tracks("rare tag") == [{"title": "Only", "artist": "One", "mbid": None}]


def test_similar_tracks_include_match_score():
    client, session = lastfm(lambda url, params: {"similartracks": {"track": [
        {"name": "Karma Police", "match": "0.87", "artist": {"name": "Radiohead"}},
    ]}})
    [track] = client.similar_tracks("Creep", "Radiohead")
    assert track["match"] == pytest.approx(0.87)
    assert session.calls[0]["params"]["method"] == "track.getSimilar"
    assert session.calls[0]["params"]["track"] == "Creep"


def test_top_tags_are_lowercased_deduped_and_limited():
    client, _ = lastfm(lambda url, params: {"toptags": {"tag": [
        {"name": "Alternative Rock"}, {"name": "alternative rock"}, {"name": "Sad"}, {"name": "90s"},
    ]}})
    assert client.top_tags("Creep", "Radiohead", limit=2) == ["alternative rock", "sad"]


def test_lastfm_not_found_returns_empty_list():
    client, _ = lastfm(lambda url, params: {"error": 6, "message": "Track not found"})
    assert client.top_tags("Made Up", "Nobody") == []
    assert client.similar_tracks("Made Up", "Nobody") == []


@pytest.mark.parametrize("data", [
    {"error": 10, "message": "Invalid API key"},
    {"error": 29, "message": "Rate limit exceeded"},
])
def test_lastfm_errors_raise(data):
    client, _ = lastfm(lambda url, params: data)
    with pytest.raises(ExternalAPIError, match=str(data["error"])):
        client.tag_top_tracks("rock")


def test_network_failure_raises_external_api_error():
    client, _ = lastfm(lambda url, params: requests.ConnectionError("offline"))
    with pytest.raises(ExternalAPIError, match="offline"):
        client.tag_top_tracks("rock")


def test_non_json_response_raises():
    client, _ = lastfm(lambda url, params: (ValueError("not json"), 502))
    with pytest.raises(ExternalAPIError, match="502"):
        client.tag_top_tracks("rock")


def test_lastfm_results_are_cached(db):
    client, session = lastfm(lambda url, params: {"tracks": {"track": [{"name": "A", "artist": {"name": "B"}}]}}, db=db)
    first = client.tag_top_tracks("rock")
    second = client.tag_top_tracks("ROCK")
    assert first == second
    assert len(session.calls) == 1


def test_errors_are_not_cached(db):
    responses = [{"error": 29, "message": "Rate limit exceeded"}, {"tracks": {"track": []}}]
    client, session = lastfm(lambda url, params: responses.pop(0), db=db)
    with pytest.raises(ExternalAPIError):
        client.tag_top_tracks("rock")
    assert client.tag_top_tracks("rock") == []
    assert len(session.calls) == 2


def test_throttle_waits_between_requests():
    waits = []
    client = LastFmClient(api_key="k", session=FakeSession(lambda u, p: {"tracks": {"track": []}}),
                          min_interval=10, sleep=waits.append)
    client.tag_top_tracks("rock")
    client.tag_top_tracks("pop")
    assert len(waits) == 1 and waits[0] > 9


# --- Deezer ---

def deezer_handler(search_results, bpm=128, calls=None):
    """Fake Deezer: /search returns search_results, /track/<id> returns bpm."""
    def handler(url, params):
        if url.endswith("/search"):
            return {"data": search_results(params["q"]) if callable(search_results) else search_results}
        if "/track/" in url:
            return {"id": int(url.rsplit("/", 1)[1]), "bpm": bpm}
        raise AssertionError(f"unexpected url {url}")
    return handler


HEY_JUDE = {"id": 42, "title": "Hey Jude (Remastered 2015)", "artist": {"name": "The Beatles"},
            "preview": "https://cdn.example/hey-jude.mp3"}


def test_deezer_search_returns_match_with_bpm():
    client, session = deezer(deezer_handler([HEY_JUDE], bpm=73.5))
    match = client.search("Hey Jude", "Beatles")

    assert match == DeezerMatch(deezer_id=42, title="Hey Jude (Remastered 2015)", artist="The Beatles",
                                bpm=73.5, preview_url="https://cdn.example/hey-jude.mp3")
    assert session.calls[0]["params"]["q"] == "Beatles Hey Jude"
    assert session.calls[1]["url"].endswith("/track/42")


def test_deezer_bpm_zero_means_missing():
    client, _ = deezer(deezer_handler([HEY_JUDE], bpm=0))
    assert client.search("Hey Jude", "The Beatles").bpm is None


def test_deezer_wrong_artist_is_not_a_match():
    other = {**HEY_JUDE, "artist": {"name": "A Cover Band"}}
    client, _ = deezer(deezer_handler([other]))
    assert client.search("Hey Jude", "The Beatles") is None


def test_deezer_wrong_title_is_not_a_match():
    other = {**HEY_JUDE, "title": "Hey Jude Medley"}
    client, _ = deezer(deezer_handler([other]))
    assert client.search("Hey Jude", "The Beatles") is None


def test_deezer_falls_back_to_artist_track_query():
    results = lambda q: [HEY_JUDE] if q.startswith("artist:") else []
    client, session = deezer(deezer_handler(results))
    assert client.search("Hey Jude", "The Beatles").deezer_id == 42
    assert session.calls[0]["params"]["q"] == "The Beatles Hey Jude"
    assert session.calls[1]["params"]["q"] == 'artist:"The Beatles" track:"Hey Jude"'


def test_deezer_quotes_in_title_are_removed_from_query():
    client, session = deezer(deezer_handler([]))
    client.search('Say "Hello"', "Someone")
    assert '"Hello"' not in session.calls[0]["params"]["q"]


def test_deezer_not_found_is_cached(db):
    client, session = deezer(deezer_handler([]), db=db)
    assert client.search("Made Up Song", "Nobody") is None
    calls = len(session.calls)
    assert client.search("made up song", "NOBODY") is None
    assert len(session.calls) == calls


def test_deezer_match_is_cached(db):
    client, session = deezer(deezer_handler([HEY_JUDE]), db=db)
    first = client.search("Hey Jude", "The Beatles")
    assert client.search("Hey Jude", "The Beatles") == first
    assert len(session.calls) == 2  # one search + one track lookup


# --- version-aware matching ---

def item(title, deezer_id, artist="The Killers"):
    return {"id": deezer_id, "title": title, "artist": {"name": artist}, "preview": f"https://p/{deezer_id}.mp3"}


@pytest.mark.parametrize("title, expected", [
    ("Mr. Brightside (Jacques Lu Cont's Thin White Duke Radio Remix)", True),
    ("Numb - Unshatter Film Soundtrack (Live in São Paulo)", True),
    ("Song (Acoustic)", True),
    ("Song - Radio Edit", True),
    ("Song [Demo]", True),
    ("Mr. Brightside (Jacques Lu Cont's Thin White Duke Mix)", True),
    ("Mr. Brightside (The Lindbergh Palace Dub)", True),
    ("Mr. Brightside (Jacques Lu Cont's Thin White Duke Short Version)", True),
    ("Hey Jude (Remastered 2015)", False),
    ("Song - 2011 Remaster", False),
    ("Song (Remastered Version)", False),
    ("Song (Album Version)", False),
    ("Song (Original Mix)", False),
    ("Live Forever", False),        # "live" in the main title, not a suffix
    ("Song", False),
])
def test_is_alternate_version(title, expected):
    assert is_alternate_version(title) is expected


def test_exact_title_wins_over_earlier_remix():
    results = [item("Mr. Brightside (Radio Remix)", 1), item("Mr. Brightside", 2)]
    client, _ = deezer(deezer_handler(results))
    assert client.search("Mr. Brightside", "The Killers").deezer_id == 2


def test_original_wins_over_live_version_when_no_exact_title():
    results = [item("Numb (Live in São Paulo)", 1, "Linkin Park"), item("Numb (2023 Remaster)", 2, "Linkin Park")]
    client, _ = deezer(deezer_handler(results))
    assert client.search("Numb", "Linkin Park").deezer_id == 2


def test_second_query_is_tried_when_first_has_only_versions():
    def results(query):
        if query.startswith("artist:"):
            return [item("Mr. Brightside", 2)]
        return [item("Mr. Brightside (Radio Remix)", 1)]
    client, session = deezer(deezer_handler(results))
    match = client.search("Mr. Brightside", "The Killers")
    assert match.deezer_id == 2 and not match.alternate_version
    assert [c["params"].get("q", "")[:7] for c in session.calls[:2]] == ["The Kil", "artist:"]


def test_version_is_used_when_deezer_has_nothing_else_but_its_bpm_is_not():
    client, session = deezer(deezer_handler([item("Mr. Brightside (Radio Remix)", 1)], bpm=128))
    match = client.search("Mr. Brightside", "The Killers")
    assert match.deezer_id == 1
    assert match.alternate_version is True
    assert match.bpm is None
    assert not any("/track/" in c["url"] for c in session.calls)


def test_original_match_is_not_flagged_as_alternate():
    client, _ = deezer(deezer_handler([HEY_JUDE]))
    assert client.search("Hey Jude", "The Beatles").alternate_version is False


def test_requested_version_is_kept():
    results = [item("Hurt (Live)", 1, "Johnny Cash"), item("Hurt", 2, "Johnny Cash")]
    client, _ = deezer(deezer_handler(results))
    assert client.search("Hurt (Live)", "Johnny Cash").deezer_id == 1


def test_fresh_preview_url_is_fetched_every_time(db):
    client, session = deezer(lambda url, params: {"id": 42, "preview": "https://cdn.example/new.mp3"}, db=db)
    assert client.fresh_preview_url(42) == "https://cdn.example/new.mp3"
    client.fresh_preview_url(42)
    assert len(session.calls) == 2
    assert session.calls[0]["url"].endswith("/track/42")


def test_fresh_preview_url_missing_is_none():
    client, _ = deezer(lambda url, params: {"id": 42, "preview": ""})
    assert client.fresh_preview_url(42) is None


def test_deezer_error_payload_raises():
    client, _ = deezer(lambda url, params: {"error": {"code": 4, "message": "Quota limit exceeded"}})
    with pytest.raises(ExternalAPIError, match="Quota"):
        client.search("Hey Jude", "The Beatles")
