import math

import pytest

from src.external import DeezerMatch, ExternalAPIError
from src.recommender import Song
from src.validation import (EXACT_TITLE_WEIGHT, GENRE_TAG_WEIGHT, MBID_WEIGHT, MEASURED_TEMPO_WEIGHT,
                            MOOD_TAG_WEIGHT, VERIFIED_WEIGHT, SongValidator, ValidationResult)


def make_song(**overrides) -> Song:
    values = dict(id=0, title="Forget Her", artist="Jeff Buckley", genre="rock", mood="sad",
                  energy=0.4, tempo_bpm=91.5, valence=0.2, acousticness=0.5, source="lastfm",
                  mbid="mbid-1", estimated_fields=["energy", "valence", "acousticness"])
    values.update(overrides)
    return Song(**values)


def make_match(**overrides) -> DeezerMatch:
    values = dict(deezer_id=108733474, title="Forget Her", artist="Jeff Buckley", bpm=91.5,
                  preview_url="https://cdn.example/p.mp3")
    values.update(overrides)
    return DeezerMatch(**values)


class FakeDeezer:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def search(self, title, artist):
        self.calls.append((title, artist))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def validator():
    return SongValidator(deezer=FakeDeezer(make_match()))


# --- verify ---

def test_verify_returns_deezer_match(validator):
    assert validator.verify({"title": " Forget Her ", "artist": "Jeff Buckley"}) == make_match()
    assert validator.deezer.calls == [("Forget Her", "Jeff Buckley")]


def test_verify_returns_none_when_not_found():
    assert SongValidator(FakeDeezer(None)).verify({"title": "Made Up", "artist": "Nobody"}) is None


def test_verify_without_title_or_artist_skips_the_api():
    deezer = FakeDeezer(make_match())
    assert SongValidator(deezer).verify({"title": "", "artist": "Jeff Buckley"}) is None
    assert deezer.calls == []


def test_verify_passes_api_errors_on():
    validator = SongValidator(FakeDeezer(error=ExternalAPIError("rate limited")))
    with pytest.raises(ExternalAPIError):
        validator.verify({"title": "Forget Her", "artist": "Jeff Buckley"})


def test_verify_needs_a_deezer_client():
    with pytest.raises(RuntimeError):
        SongValidator().verify({"title": "A", "artist": "B"})


# --- schema ---

def test_valid_song_passes_schema(validator):
    assert validator.schema_errors(make_song()) == []
    assert validator.validate_schema(make_song())


def test_missing_tempo_is_allowed(validator):
    assert validator.validate_schema(make_song(tempo_bpm=None))


def test_genre_alias_is_allowed(validator):
    assert validator.validate_schema(make_song(genre="Hip Hop"))


@pytest.mark.parametrize("overrides, message", [
    ({"title": "  "}, "title is empty"),
    ({"artist": ""}, "artist is empty"),
    ({"genre": "vaporwave"}, "not on the controlled genre list"),
    ({"mood": " "}, "mood is empty"),
    ({"energy": 1.2}, "energy must be"),
    ({"valence": -0.1}, "valence must be"),
    ({"acousticness": math.nan}, "acousticness must be"),
    ({"energy": "high"}, "energy must be"),
    ({"tempo_bpm": 0}, "tempo_bpm must be"),
    ({"tempo_bpm": 999}, "tempo_bpm must be"),
    ({"confidence": 1.5}, "confidence must be"),
    ({"estimated_fields": ["energy", "lyrics"]}, "unknown fields"),
    ({"source": ""}, "source is empty"),
])
def test_schema_errors_are_reported(validator, overrides, message):
    errors = validator.schema_errors(make_song(**overrides))
    assert any(message in error for error in errors), errors
    assert not validator.validate_schema(make_song(**overrides))


def test_all_schema_errors_are_listed_at_once(validator):
    errors = validator.schema_errors(make_song(energy=2, valence=2, genre="vaporwave"))
    assert len(errors) == 3


# --- duplicates ---

def test_same_title_and_artist_is_duplicate(validator):
    assert validator.is_duplicate(make_song(), [make_song(title="FORGET HER", mbid=None)])


@pytest.mark.parametrize("title", ["Forget Her (Remastered 2004)", "Forget Her - Live", "Forget Her [Demo]"])
def test_near_duplicate_titles_are_caught(validator, title):
    assert validator.is_duplicate(make_song(title=title, mbid=None), [make_song(mbid=None)])


def test_leading_the_in_artist_is_ignored(validator):
    existing = [make_song(title="Creep", artist="The Radiohead", mbid=None)]
    assert validator.is_duplicate(make_song(title="Creep", artist="Radiohead", mbid=None), existing)


def test_same_mbid_is_duplicate(validator):
    assert validator.is_duplicate(make_song(title="Other Title"), [make_song(title="Different")])


def test_same_deezer_id_is_duplicate(validator):
    existing = [make_song(title="Different", mbid=None, deezer_id=5)]
    assert validator.is_duplicate(make_song(title="Other", mbid=None, deezer_id=5), existing)


def test_different_song_is_not_duplicate(validator):
    existing = [make_song(title="Grace", mbid="mbid-2"), make_song(artist="Tim Buckley", mbid=None)]
    assert not validator.is_duplicate(make_song(), existing)


def test_missing_ids_do_not_count_as_equal(validator):
    existing = [make_song(title="Grace", mbid=None, deezer_id=None)]
    assert not validator.is_duplicate(make_song(mbid=None, deezer_id=None), existing)


def test_empty_database_has_no_duplicates(validator):
    assert not validator.is_duplicate(make_song(), [])


# --- confidence ---

def test_unverified_song_scores_zero(validator):
    assert validator.confidence(make_song(), None, tags=["rock", "sad"]) == 0.0


def test_fully_backed_song_scores_one(validator):
    assert validator.confidence(make_song(), make_match(), tags=["Rock", "sad"]) == pytest.approx(1.0)


def test_weights_sum_to_one():
    total = (VERIFIED_WEIGHT + EXACT_TITLE_WEIGHT + MBID_WEIGHT + MEASURED_TEMPO_WEIGHT
             + GENRE_TAG_WEIGHT + MOOD_TAG_WEIGHT)
    assert total == pytest.approx(1.0)


def test_verified_only_scores_the_base_weight(validator):
    song = make_song(title="Forget Her", mbid=None, tempo_bpm=None)
    match = make_match(title="Forget Her (Remastered)", bpm=None)
    assert validator.confidence(song, match, tags=[]) == pytest.approx(VERIFIED_WEIGHT)


def test_breakdown_names_each_piece_of_evidence(validator):
    breakdown = validator.confidence_breakdown(make_song(mbid=None), make_match(), tags=["sad"])
    assert breakdown == {
        "verified": VERIFIED_WEIGHT, "exact_title": EXACT_TITLE_WEIGHT, "mbid": 0.0,
        "measured_tempo": MEASURED_TEMPO_WEIGHT, "genre_tag": 0.0, "mood_tag": MOOD_TAG_WEIGHT,
    }


def test_estimated_tempo_does_not_count_as_measured(validator):
    song = make_song(estimated_fields=["tempo_bpm"])
    assert validator.confidence_breakdown(song, make_match(), [])["measured_tempo"] == 0.0


@pytest.mark.parametrize("genre, tags, backed", [
    ("rock", ["rock"], True),
    ("metal", ["rock"], True),            # tag is a parent of the genre
    ("rock", ["heavy metal"], True),      # tag is a subgenre of the genre
    ("rock", ["alt-rock"], True),         # tag alias
    ("rock", ["jazz", "chill"], False),
    ("indie pop", ["rock"], False),
])
def test_genre_tag_support(validator, genre, tags, backed):
    points = validator.confidence_breakdown(make_song(genre=genre), make_match(), tags)["genre_tag"]
    assert points == (GENRE_TAG_WEIGHT if backed else 0.0)


def test_mood_tag_match_ignores_case(validator):
    breakdown = validator.confidence_breakdown(make_song(mood="Sad"), make_match(), ["SAD "])
    assert breakdown["mood_tag"] == MOOD_TAG_WEIGHT


# --- review ---

def test_review_accepts_good_song(validator):
    result = validator.review(make_song(), existing=[], match=make_match(), tags=["rock", "sad"])
    assert isinstance(result, ValidationResult)
    assert result.accepted
    assert result.reasons == []
    assert result.confidence == pytest.approx(1.0)
    assert result.match == make_match()


def test_review_rejects_unverified_song(validator):
    result = validator.review(make_song(), existing=[], match=None)
    assert not result.accepted
    assert result.reasons == ["not found on Deezer"]
    assert result.confidence == 0.0


def test_review_rejects_duplicate(validator):
    result = validator.review(make_song(), existing=[make_song()], match=make_match())
    assert not result.accepted
    assert result.reasons == ["duplicate of a song already in the database"]


def test_review_lists_every_reason(validator):
    result = validator.review(make_song(energy=5, genre="vaporwave"), existing=[make_song()], match=None)
    assert len(result.reasons) == 4
