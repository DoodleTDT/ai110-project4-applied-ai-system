"""End-to-end agent tests with fake Claude, Last.fm and Deezer (no network, no cost)."""

import json
from pathlib import Path

import pytest

from src.database import SongDatabase
from src.external import DeezerMatch, ExternalAPIError
from src.llm import (LLMError, PlaylistWriter, RequestParser, RequestTooLongError, TokenBudgetExceeded,
                     VibeEstimator)
from src.logging_utils import AgentLogger
from src.pipeline import CoverageChecker, PlaylistAgent
from src.recommender import Song, UserProfile, recommend_songs
from src.validation import SongValidator


# --- fakes ---

class FakeParser:
    def __init__(self, profile):
        self.profile = profile
        self.budget = None

    def parse(self, text):
        RequestParser.check_request(text)
        return self.profile


class FakeEstimator:
    def __init__(self, error=None, skip=()):
        self.error = error
        self.skip = set(skip)
        self.budget = None
        self.calls = []

    def estimate_many(self, songs):
        self.calls.append([s["title"] for s in songs])
        if self.error:
            raise self.error
        return [None if s["title"] in self.skip else
                {"genre": "rock", "mood": "sad", "energy": 0.25, "valence": 0.2, "acousticness": 0.7}
                for s in songs]


class FakeWriter:
    def __init__(self, error=None):
        self.error = error
        self.budget = None
        self.items = None

    def write(self, request, items):
        self.items = items
        if self.error:
            raise self.error
        return [f"Reason for {item['title']}" for item in items]


class FakeLastFm:
    def __init__(self, tracks_by_tag, tags=("rock", "sad"), error_tags=()):
        self.tracks_by_tag = tracks_by_tag
        self.tags = list(tags)
        self.error_tags = set(error_tags)
        self.queried = []

    def tag_top_tracks(self, tag, limit=50):
        self.queried.append(tag)
        if tag in self.error_tags:
            raise ExternalAPIError("rate limited")
        return [dict(t) for t in self.tracks_by_tag.get(tag, [])][:limit]

    def top_tags(self, title, artist):
        return list(self.tags)


class FakeDeezer:
    """Knows every song whose title does not start with 'Fake'."""

    def __init__(self, error_titles=()):
        self.error_titles = set(error_titles)
        self.searched = []

    def search(self, title, artist):
        self.searched.append(title)
        if title in self.error_titles:
            raise ExternalAPIError("quota")
        if title.startswith("Fake"):
            return None
        return DeezerMatch(deezer_id=abs(hash((title, artist))) % 10**9, title=title, artist=artist,
                           bpm=80.0, preview_url="https://cdn.example/p.mp3")


def track(title, artist="Band", mbid=None):
    return {"title": title, "artist": artist, "mbid": mbid}


SAD_ROCK = UserProfile(favorite_genres=["rock"], favorite_mood="sad", target_energy=0.25, likes_acoustic=True)


@pytest.fixture
def db(tmp_path):
    database = SongDatabase(str(tmp_path / "test.db3"))
    yield database
    database.close()


def make_agent(db, lastfm=None, deezer=None, estimator=None, writer=None, profile=SAD_ROCK, **kwargs):
    deezer = deezer or FakeDeezer()
    return PlaylistAgent(
        db=db, parser=FakeParser(profile), lastfm=lastfm or FakeLastFm({}), deezer=deezer,
        estimator=estimator or FakeEstimator(), validator=SongValidator(deezer=deezer),
        writer=writer or FakeWriter(), logger=AgentLogger(log_path=None), **kwargs)


def approve_song(db, title, genre="rock", mood="sad", energy=0.25, acousticness=0.7):
    song = Song(id=0, title=title, artist="Local", genre=genre, mood=mood, energy=energy,
                tempo_bpm=90, valence=0.2, acousticness=acousticness)
    return db.approve(db.add_pending(song, verified=True))


# --- CoverageChecker ---

def ranked_with_scores(*scores):
    return [({"id": i}, score, "") for i, score in enumerate(scores)]


def test_coverage_needs_external_when_too_few_good_matches():
    checker = CoverageChecker(threshold=6.0, min_per_genre=0)
    assert checker.needs_external(ranked_with_scores(8, 7, 5), SAD_ROCK, k=3, genre_counts={"rock": 3})
    assert not checker.needs_external(ranked_with_scores(8, 7, 6), SAD_ROCK, k=3, genre_counts={"rock": 3})


def test_coverage_needs_external_when_a_genre_is_thin():
    checker = CoverageChecker(threshold=6.0, min_per_genre=5)
    assert checker.needs_external(ranked_with_scores(9, 9), SAD_ROCK, k=2, genre_counts={"rock": 4})
    assert not checker.needs_external(ranked_with_scores(9, 9), SAD_ROCK, k=2, genre_counts={"rock": 5})


def test_subgenres_count_toward_coverage():
    checker = CoverageChecker(min_per_genre=5)
    assert checker.genre_coverage(SAD_ROCK, {"rock": 2, "metal": 2, "grunge": 1, "pop": 9}) == {"rock": 5}


def test_target_genres_put_the_thinnest_genre_first():
    profile = UserProfile(favorite_genres=["pop", "rock", "jazz"], favorite_mood="happy", target_energy=0.5)
    counts = {"pop": 30, "rock": 4, "metal": 3, "jazz": 2}  # rock coverage = 4 + 3 (metal) = 7
    assert CoverageChecker().target_genres(profile, counts) == ["jazz", "rock", "pop"]


def test_target_genres_keep_request_order_on_ties():
    profile = UserProfile(favorite_genres=["jazz", "folk"], favorite_mood="happy", target_energy=0.5)
    assert CoverageChecker().target_genres(profile, {}) == ["jazz", "folk"]


# --- local only ---

def test_full_local_coverage_skips_external_search(db):
    for i in range(6):
        approve_song(db, f"Local {i}")
    lastfm = FakeLastFm({"rock": [track("New")]})
    estimator = FakeEstimator()
    result = make_agent(db, lastfm=lastfm, estimator=estimator).run("sad rock", k=5)

    assert not result.used_external
    assert lastfm.queried == [] and estimator.calls == []
    assert len(result.entries) == 5
    assert all(e.source == "songs" and not e.pending_review for e in result.entries)
    assert result.entries[0].reason == f"Reason for {result.entries[0].song.title}"


def test_ranking_matches_recommend_songs(db):
    for i, energy in enumerate([0.9, 0.25, 0.5, 0.1, 0.3, 0.7]):
        approve_song(db, f"Local {i}", energy=energy)
    result = make_agent(db, coverage=CoverageChecker(threshold=0, min_per_genre=0)).run("x", k=6)
    expected = recommend_songs(
        {"favorite_genres": ["rock"], "favorite_mood": "sad", "target_energy": 0.25, "likes_acoustic": True},
        [{**vars(s)} for s in db.get_songs()], k=6)
    assert [e.song.title for e in result.entries] == [s["title"] for s, _, _ in expected]


def test_verified_pending_songs_are_candidates_but_unverified_are_not(db):
    approve_song(db, "Approved")
    base = dict(artist="P", genre="rock", mood="sad", energy=0.25, tempo_bpm=90, valence=0.2, acousticness=0.7)
    db.add_pending(Song(id=0, title="Verified Pending", **base), verified=True)
    db.add_pending(Song(id=0, title="Unverified Pending", **base), verified=False)
    db.add_pending(Song(id=0, title="Unchecked Pending", **base), verified=None)

    result = make_agent(db).run("sad rock", k=10)
    titles = {e.song.title: e for e in result.entries}
    assert set(titles) == {"Approved", "Verified Pending"}
    assert titles["Verified Pending"].pending_review and titles["Verified Pending"].source == "pending"


# --- external search ---

def test_empty_database_triggers_search_and_stages_verified_songs(db):
    lastfm = FakeLastFm({"rock": [track("Real One", mbid="m1"), track("Fake Song"), track("Real Two")]})
    result = make_agent(db, lastfm=lastfm).run("sad low-energy rock", k=10)

    assert result.used_external
    assert {s.title for s in result.staged} == {"Real One", "Real Two"}
    assert result.rejected == [{"song": "Fake Song - Band", "reasons": ["not found on Deezer"]}]

    pending = {p.song.title: p for p in db.list_pending()}
    assert set(pending) == {"Real One", "Real Two"}
    staged = pending["Real One"]
    assert staged.verified is True
    assert staged.song.source == "lastfm"
    assert staged.song.mbid == "m1"
    assert staged.song.estimated_fields == ["genre", "mood", "energy", "valence", "acousticness"]
    assert 0 < staged.song.confidence <= 1
    assert db.get_songs() == []  # nothing is approved automatically

    assert {e.song.title for e in result.entries} == {"Real One", "Real Two"}
    assert all(e.pending_review for e in result.entries)


def test_unverified_and_duplicate_songs_never_reach_claude(db):
    approve_song(db, "Already Known")
    lastfm = FakeLastFm({"rock": [track("Already Known", artist="Local"), track("Fake One"),
                                  track("Real (Remastered)"), track("Real")]})
    estimator = FakeEstimator()
    make_agent(db, lastfm=lastfm, estimator=estimator).run("sad rock", k=10)
    assert estimator.calls == [["Real (Remastered)"]]


def test_external_candidates_are_capped(db):
    lastfm = FakeLastFm({"rock": [track(f"Real {i}") for i in range(40)]})
    deezer = FakeDeezer()
    estimator = FakeEstimator()
    make_agent(db, lastfm=lastfm, deezer=deezer, estimator=estimator, max_external_candidates=4).run("x", k=10)
    assert len(deezer.searched) == 4
    assert len(estimator.calls[0]) == 4
    assert len(db.list_pending()) == 4


def test_search_queries_thin_genres_first_then_mood(db):
    for i in range(3):
        approve_song(db, f"Pop {i}", genre="pop")
    profile = UserProfile(favorite_genres=["pop", "jazz"], favorite_mood="sad", target_energy=0.25)
    lastfm = FakeLastFm({})
    make_agent(db, lastfm=lastfm, profile=profile).run("x", k=10)
    assert lastfm.queried == ["jazz", "pop", "sad"]


def test_lastfm_error_on_one_tag_moves_on(db):
    lastfm = FakeLastFm({"sad": [track("Real Sad")]}, error_tags={"rock"})
    result = make_agent(db, lastfm=lastfm).run("x", k=10)
    assert [s.title for s in result.staged] == ["Real Sad"]


def test_deezer_error_rejects_only_that_song(db):
    lastfm = FakeLastFm({"rock": [track("Real A"), track("Real B")]})
    result = make_agent(db, lastfm=lastfm, deezer=FakeDeezer(error_titles={"Real A"})).run("x", k=10)
    assert [s.title for s in result.staged] == ["Real B"]
    assert "could not check on Deezer" in result.rejected[0]["reasons"][0]


def test_estimator_over_budget_stages_nothing_and_still_returns(db):
    approve_song(db, "Local")
    lastfm = FakeLastFm({"rock": [track("Real A")]})
    estimator = FakeEstimator(error=TokenBudgetExceeded("over budget"))
    result = make_agent(db, lastfm=lastfm, estimator=estimator).run("x", k=10)
    assert result.staged == []
    assert db.list_pending() == []
    assert [e.song.title for e in result.entries] == ["Local"]
    assert "over budget" in result.rejected[0]["reasons"][0]


def test_missing_vibe_estimate_rejects_that_song(db):
    lastfm = FakeLastFm({"rock": [track("Real A"), track("Real B")]})
    result = make_agent(db, lastfm=lastfm, estimator=FakeEstimator(skip={"Real A"})).run("x", k=10)
    assert [s.title for s in result.staged] == ["Real B"]


def test_invalid_estimate_is_rejected_by_guardrails(db):
    class BadEstimator(FakeEstimator):
        def estimate_many(self, songs):
            return [{"genre": "rock", "mood": "sad", "energy": 1.7, "valence": 0.2, "acousticness": 0.5}
                    for _ in songs]

    lastfm = FakeLastFm({"rock": [track("Real A")]})
    result = make_agent(db, lastfm=lastfm, estimator=BadEstimator()).run("x", k=10)
    assert result.staged == []
    assert "energy must be" in result.rejected[0]["reasons"][0]


# --- writer fallback ---

@pytest.mark.parametrize("error", [TokenBudgetExceeded("over"), LLMError("down")])
def test_writer_failure_falls_back_to_score_reasons(db, error):
    approve_song(db, "Local")
    result = make_agent(db, writer=FakeWriter(error=error)).run("x", k=5)
    assert result.entries[0].reason.startswith("Score ")
    assert "/9.00" in result.entries[0].reason


def test_writer_gets_score_reasons_and_source(db):
    approve_song(db, "Local")
    writer = FakeWriter()
    make_agent(db, writer=writer).run("x", k=5)
    item = writer.items[0]
    assert item["source"] == "approved"
    assert "genre match" in item["score_reasons"]


# --- request guardrail, budget, logging ---

def test_too_long_request_stops_before_any_work(db):
    lastfm = FakeLastFm({"rock": [track("Real")]})
    with pytest.raises(RequestTooLongError):
        make_agent(db, lastfm=lastfm).run("x" * 501)
    assert lastfm.queried == []


def test_one_budget_is_shared_by_all_modules(db):
    agent = make_agent(db, token_limit=12_345)
    result = agent.run("x", k=5)
    assert agent.parser.budget is agent.estimator.budget is agent.writer.budget
    assert agent.parser.budget.limit == 12_345
    assert result.tokens["limit"] == 12_345


def test_each_run_gets_a_fresh_budget(db):
    agent = make_agent(db)
    agent.run("x")
    first = agent.parser.budget
    agent.run("x")
    assert agent.parser.budget is not first


def test_steps_are_logged_with_a_run_id(db, tmp_path):
    log_path = tmp_path / "logs" / "steps.jsonl"
    lastfm = FakeLastFm({"rock": [track("Real A")]})
    agent = make_agent(db, lastfm=lastfm)
    agent.logger = AgentLogger(str(log_path))
    result = agent.run("sad rock", k=5)

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    steps = [r["step"] for r in records]
    assert steps[:2] == ["request", "parse_request"]
    assert {"search_local_db", "coverage_check", "search_external", "propose_song_addition",
            "external_results", "playlist"} <= set(steps)
    assert all(r["run_id"] == result.run_id for r in records)
    assert "tokens" in records[-1]


# --- with the real Claude modules and a fake Claude client ---

def test_end_to_end_with_real_modules_and_fake_claude(db):
    from types import SimpleNamespace

    def reply(data, inp, out):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(data))],
                               stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=inp, output_tokens=out))

    class Client:
        def __init__(self):
            self.calls = []
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            self.calls.append(kwargs)
            content = kwargs["messages"][0]["content"]
            if "Playlist request" in content:
                return reply({"favorite_genres": ["rock"], "favorite_mood": "sad",
                              "target_energy": 0.2, "likes_acoustic": True}, 900, 150)
            if "Songs (one JSON object per line)" in content and "Listener's request" not in content:
                n = content.count('"index"')
                return reply({"songs": [{"index": i, "genre": "rock", "mood": "sad", "energy": 0.2,
                                         "valence": 0.1, "acousticness": 0.8} for i in range(n)]}, 1500, 600)
            n = content.count('"index"')
            return reply({"reasons": [{"index": i, "reason": f"Reason {i}"} for i in range(n)]}, 1800, 700)

    client = Client()
    deezer = FakeDeezer()
    agent = PlaylistAgent(
        db=db, parser=RequestParser(client=client), lastfm=FakeLastFm({"rock": [track("Real A"), track("Real B")]}),
        deezer=deezer, estimator=VibeEstimator(client=client, db=db), validator=SongValidator(deezer=deezer),
        writer=PlaylistWriter(client=client), logger=AgentLogger(log_path=None))
    result = agent.run("sad low-energy rock for a rainy night", k=5)

    assert len(client.calls) == 3  # parse, one batched vibe call, write
    assert [e.reason for e in result.entries] == ["Reason 0", "Reason 1"]
    assert result.tokens["used"] == 900 + 150 + 1500 + 600 + 1800 + 700
    assert [c["call"] for c in result.tokens["calls"]] == ["parse_request", "estimate_vibe", "write_playlist"]
