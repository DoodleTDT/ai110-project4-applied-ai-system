import json
from types import SimpleNamespace

import pytest

from src.database import SongDatabase
from src.llm import (FALLBACK_BETA, MAX_REQUEST_CHARS, MODEL, MOODS, VIBE_BATCH_SIZE, LLMError,
                     PlaylistWriter, RequestParser, RequestTooLongError, TokenBudget,
                     TokenBudgetExceeded, VibeEstimator)
from src.recommender import UserProfile


def response(data, stop_reason="end_turn", input_tokens=100, output_tokens=50, text=None):
    """A fake Messages API response with one text block."""
    body = text if text is not None else json.dumps(data)
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=body)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                              cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )


class FakeClaude:
    """Records create() calls; returns handler(kwargs) or the next queued response."""

    def __init__(self, *responses, handler=None):
        self.queue = list(responses)
        self.handler = handler
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.handler:
            return self.handler(kwargs)
        return self.queue.pop(0)


PROFILE = {"favorite_genres": ["rock"], "favorite_mood": "sad", "target_energy": 0.2, "likes_acoustic": True}


# --- request shape (shared by every module) ---

def test_calls_use_the_configured_model_and_structured_output():
    client = FakeClaude(response(PROFILE))
    RequestParser(client=client).parse("sad low-energy rock")
    call = client.calls[0]
    assert call["model"] == MODEL
    assert call["output_config"]["effort"] == "low"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["max_tokens"] > 0
    assert "sad low-energy rock" in call["messages"][0]["content"]


def test_fallback_is_sent_for_models_that_support_it():
    client = FakeClaude(response(PROFILE))
    RequestParser(client=client, model="claude-sonnet-5-5").parse("x")
    assert client.calls[0]["betas"] == [FALLBACK_BETA]
    assert client.calls[0]["fallbacks"] == "default"


def test_fallback_is_left_out_for_haiku():
    import anthropic
    client = FakeClaude(response(PROFILE))
    RequestParser(client=client, model="claude-haiku-5-5").parse("x")
    assert client.calls[0]["betas"] is anthropic.omit
    assert client.calls[0]["fallbacks"] is anthropic.omit


def test_writer_uses_medium_effort():
    client = FakeClaude(response({"reasons": [{"index": 0, "reason": "Fits."}]}))
    PlaylistWriter(client=client).write("rock", [{"title": "A", "artist": "B"}])
    assert client.calls[0]["output_config"]["effort"] == "medium"


# --- RequestParser ---

def test_parse_returns_user_profile():
    profile = RequestParser(client=FakeClaude(response(PROFILE))).parse("sad low-energy rock")
    assert isinstance(profile, UserProfile)
    assert profile.favorite_genres == ["rock"]
    assert profile.favorite_mood == "sad"
    assert profile.target_energy == 0.2
    assert profile.likes_acoustic is True


def test_parse_maps_trims_and_dedupes_genres():
    data = {**PROFILE, "favorite_genres": ["hip-hop", "hip-hop", "jazz", "rock", "pop"]}
    profile = RequestParser(client=FakeClaude(response(data))).parse("x")
    assert profile.favorite_genres == ["hip-hop", "jazz", "rock"]


def test_parse_clamps_energy_and_accepts_null_acoustic():
    data = {**PROFILE, "target_energy": 1.4, "likes_acoustic": None}
    profile = RequestParser(client=FakeClaude(response(data))).parse("x")
    assert profile.target_energy == 1.0
    assert profile.likes_acoustic is None


def test_parse_with_no_usable_genre_raises():
    data = {**PROFILE, "favorite_genres": []}
    with pytest.raises(LLMError, match="no usable genre"):
        RequestParser(client=FakeClaude(response(data))).parse("x")


def test_parse_sends_allowed_genre_and_mood_lists():
    client = FakeClaude(response(PROFILE))
    RequestParser(client=client).parse("x")
    schema = client.calls[0]["output_config"]["format"]["schema"]
    assert "metal" in schema["properties"]["favorite_genres"]["items"]["enum"]
    assert schema["properties"]["favorite_mood"]["enum"] == MOODS


# --- guardrail 7: request length ---

def test_request_at_the_limit_is_accepted():
    client = FakeClaude(response(PROFILE))
    RequestParser(client=client).parse("a" * MAX_REQUEST_CHARS)
    assert len(client.calls) == 1


def test_too_long_request_is_rejected_without_calling_claude():
    client = FakeClaude()
    with pytest.raises(RequestTooLongError, match=str(MAX_REQUEST_CHARS)):
        RequestParser(client=client).parse("a" * (MAX_REQUEST_CHARS + 1))
    assert client.calls == []


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_request_is_rejected(text):
    with pytest.raises(ValueError, match="empty"):
        RequestParser(client=FakeClaude()).parse(text)


# --- guardrail 5: bad responses are errors, never partial results ---

def test_max_tokens_cutoff_raises():
    client = FakeClaude(response(PROFILE, stop_reason="max_tokens"))
    with pytest.raises(LLMError, match="max_tokens"):
        RequestParser(client=client).parse("x")


def test_refusal_raises():
    with pytest.raises(LLMError, match="declined"):
        RequestParser(client=FakeClaude(response(PROFILE, stop_reason="refusal"))).parse("x")


def test_invalid_json_raises():
    with pytest.raises(LLMError, match="not valid JSON"):
        RequestParser(client=FakeClaude(response(None, text="{not json"))).parse("x")


def test_api_error_becomes_llm_error():
    import anthropic
    import httpx2 as httpx

    def fail(kwargs):
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))

    with pytest.raises(LLMError, match="failed"):
        RequestParser(client=FakeClaude(handler=fail)).parse("x")


# --- guardrail 6: token budget ---

def test_budget_records_usage_per_call():
    budget = TokenBudget(limit=10_000)
    RequestParser(client=FakeClaude(response(PROFILE, input_tokens=700, output_tokens=120)),
                  budget=budget).parse("x")
    assert budget.used == 820
    assert budget.calls == [{"call": "parse_request", "input_tokens": 700, "output_tokens": 120}]


def test_budget_counts_cache_tokens_as_input():
    budget = TokenBudget()
    budget.record("x", SimpleNamespace(input_tokens=10, output_tokens=5,
                                       cache_creation_input_tokens=100, cache_read_input_tokens=200))
    assert budget.input_tokens == 310


def test_call_over_budget_is_refused_before_calling_claude():
    client = FakeClaude(response(PROFILE))
    with pytest.raises(TokenBudgetExceeded):
        RequestParser(client=client, budget=TokenBudget(limit=500)).parse("x")
    assert client.calls == []


def test_budget_shared_across_modules_runs_out():
    budget = TokenBudget(limit=6_000)
    parser = RequestParser(client=FakeClaude(response(PROFILE, input_tokens=2_000, output_tokens=1_000)),
                           budget=budget)
    parser.parse("x")
    writer = PlaylistWriter(client=FakeClaude(), budget=budget)
    with pytest.raises(TokenBudgetExceeded):
        writer.write("x", [{"title": "A", "artist": "B"}] * 5)


# --- VibeEstimator ---

def vibe_item(index, **overrides):
    item = {"index": index, "genre": "rock", "mood": "sad", "energy": 0.3, "valence": 0.2, "acousticness": 0.6}
    item.update(overrides)
    return item


SONGS = [{"title": f"Song {i}", "artist": "Band", "tags": ["rock", "sad"]} for i in range(3)]


def test_estimate_many_returns_results_in_order():
    data = {"songs": [vibe_item(2, energy=0.9), vibe_item(0), vibe_item(1, genre="Alt-Rock")]}
    results = VibeEstimator(client=FakeClaude(response(data))).estimate_many(SONGS)
    assert [r["energy"] for r in results] == [0.3, 0.3, 0.9]
    assert results[1]["genre"] == "alternative rock"


def test_estimate_many_sends_one_call_per_batch():
    songs = [{"title": f"S{i}", "artist": "B", "tags": []} for i in range(VIBE_BATCH_SIZE + 2)]

    def handler(kwargs):
        count = kwargs["messages"][0]["content"].count('"index"')
        return response({"songs": [vibe_item(i) for i in range(count)]})

    client = FakeClaude(handler=handler)
    results = VibeEstimator(client=client).estimate_many(songs)
    assert len(client.calls) == 2
    assert all(r is not None for r in results)


def test_missing_or_bad_estimates_are_none():
    data = {"songs": [vibe_item(0), vibe_item(7), vibe_item(1, energy="loud")]}
    results = VibeEstimator(client=FakeClaude(response(data))).estimate_many(SONGS)
    assert results[0] is not None
    assert results[1] is None and results[2] is None


def test_repeated_index_keeps_first_estimate():
    data = {"songs": [vibe_item(0, energy=0.1), vibe_item(0, energy=0.9)]}
    results = VibeEstimator(client=FakeClaude(response(data))).estimate_many(SONGS[:1])
    assert results[0]["energy"] == 0.1


def test_song_text_is_sent_as_data_with_tags_limited():
    client = FakeClaude(response({"songs": [vibe_item(0)]}))
    VibeEstimator(client=client).estimate_many([{"title": "T", "artist": "A", "tags": [f"t{i}" for i in range(20)]}])
    content = client.calls[0]["messages"][0]["content"]
    assert '"title": "T"' in content
    assert "t9" in content and "t10" not in content


def test_estimates_are_cached(tmp_path):
    with SongDatabase(str(tmp_path / "t.db3")) as db:
        first = FakeClaude(response({"songs": [vibe_item(0), vibe_item(1)]}))
        VibeEstimator(client=first, db=db).estimate_many(SONGS[:2])

        second = FakeClaude(response({"songs": [vibe_item(0, energy=0.8)]}))
        results = VibeEstimator(client=second, db=db).estimate_many(SONGS)
        assert len(second.calls) == 1
        assert second.calls[0]["messages"][0]["content"].count('"index"') == 1  # only Song 2 sent
        assert [r["energy"] for r in results] == [0.3, 0.3, 0.8]


def test_all_cached_makes_no_call(tmp_path):
    with SongDatabase(str(tmp_path / "t.db3")) as db:
        VibeEstimator(client=FakeClaude(response({"songs": [vibe_item(0)]})), db=db).estimate("Song 0", "Band", [])
        client = FakeClaude()
        assert VibeEstimator(client=client, db=db).estimate("Song 0", "Band", [])["genre"] == "rock"
        assert client.calls == []


# --- PlaylistWriter ---

ITEMS = [{"title": "Creep", "artist": "Radiohead", "genre": "rock", "mood": "sad", "energy": 0.4,
          "source": "approved", "score_reasons": "genre match"},
         {"title": "Hurt", "artist": "Johnny Cash", "genre": "country", "mood": "sad", "energy": 0.2,
          "source": "pending review", "score_reasons": "mood match"}]


def test_write_returns_reasons_in_order():
    data = {"reasons": [{"index": 1, "reason": "Second."}, {"index": 0, "reason": "First."}]}
    assert PlaylistWriter(client=FakeClaude(response(data))).write("sad songs", ITEMS) == ["First.", "Second."]


def test_write_missing_or_blank_reason_is_none():
    data = {"reasons": [{"index": 0, "reason": "  "}, {"index": 5, "reason": "x"}]}
    assert PlaylistWriter(client=FakeClaude(response(data))).write("sad", ITEMS) == [None, None]


def test_write_with_no_songs_makes_no_call():
    client = FakeClaude()
    assert PlaylistWriter(client=client).write("sad", []) == []
    assert client.calls == []


def test_write_max_tokens_grows_with_playlist_size():
    def handler(kwargs):
        return response({"reasons": []})
    client = FakeClaude(handler=handler)
    writer = PlaylistWriter(client=client)
    writer.write("x", ITEMS[:1])
    writer.write("x", ITEMS * 5)
    assert client.calls[1]["max_tokens"] > client.calls[0]["max_tokens"]
