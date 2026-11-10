import pytest
import requests

from src import config
from src.config import MissingAPIKeyError
from src.database import SongDatabase
from src.external import (USER_AGENT, DeezerClient, DeezerMatch, ExternalAPIError,
                          LastFmClient, comparable)


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
    assert session.calls[0]["params"]["q"] == 'artist:"Beatles" track:"Hey Jude"'
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


def test_deezer_falls_back_to_plain_query():
    results = lambda q: [HEY_JUDE] if not q.startswith("artist:") else []
    client, session = deezer(deezer_handler(results))
    assert client.search("Hey Jude", "The Beatles").deezer_id == 42
    assert session.calls[1]["params"]["q"] == "The Beatles Hey Jude"


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


def test_deezer_error_payload_raises():
    client, _ = deezer(lambda url, params: {"error": {"code": 4, "message": "Quota limit exceeded"}})
    with pytest.raises(ExternalAPIError, match="Quota"):
        client.search("Hey Jude", "The Beatles")
