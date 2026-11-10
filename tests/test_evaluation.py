from pathlib import Path

import pytest

from src.evaluation import (Label, baseline_playlist, cost_usd, energy_in_range, genre_diversity, genre_related,
                            label_profile, load_labels, mean, parse_checks, precision_at_k, song_fits,
                            traceability)
from src.genres import TAXONOMY
from src.recommender import Song, UserProfile, load_songs

ROOT = Path(__file__).parent.parent
SAD_ROCK = Label(id="t", prompt="sad rock", genres=["rock"], moods=["sad", "moody"], energy=[0.0, 0.4])


def song(genre="rock", energy=0.3, deezer_id=1, title="S"):
    return Song(id=0, title=title, artist="A", genre=genre, mood="sad", energy=energy, tempo_bpm=None,
                valence=0.2, acousticness=0.5, deezer_id=deezer_id)


def test_prompt_file_loads_and_uses_allowed_labels():
    labels = load_labels(str(ROOT / "eval" / "prompts.json"))
    assert len(labels) == 8
    assert len({l.id for l in labels}) == 8
    from src.llm import MOODS
    for label in labels:
        assert all(TAXONOMY.is_allowed(g) for g in label.genres), label.id
        assert all(m in MOODS for m in label.moods), label.id
        low, high = label.energy
        assert 0 <= low < high <= 1
        label_profile(label)  # valid UserProfile


@pytest.mark.parametrize("genre, expected", [("rock", True), ("metal", True), ("grunge", True),
                                             ("jazz", False), ("pop", False)])
def test_genre_related_includes_subgenres(genre, expected):
    assert genre_related(genre, ["rock"]) is expected


def test_genre_related_includes_parents():
    assert genre_related("rock", ["metal"])


def test_energy_range_margin():
    assert energy_in_range(0.45, [0.0, 0.4])
    assert not energy_in_range(0.55, [0.0, 0.4])
    assert not energy_in_range(0.45, [0.0, 0.4], margin=0.0)


def test_song_fits_needs_genre_and_energy():
    assert song_fits(song("metal", 0.3), SAD_ROCK)
    assert not song_fits(song("jazz", 0.3), SAD_ROCK)
    assert not song_fits(song("rock", 0.9), SAD_ROCK)


def test_parse_checks():
    good = UserProfile(favorite_genres=["rock"], favorite_mood="moody", target_energy=0.2)
    bad = UserProfile(favorite_genres=["pop"], favorite_mood="happy", target_energy=0.8, likes_acoustic=True)
    assert parse_checks(good, SAD_ROCK) == {"genre": True, "mood": True, "energy": True, "acoustic": None}
    labeled = Label(**{**SAD_ROCK.__dict__, "acoustic": False})
    assert parse_checks(bad, labeled) == {"genre": False, "mood": False, "energy": False, "acoustic": False}


def test_precision_and_traceability():
    songs = [song("rock", 0.3), song("jazz", 0.3, deezer_id=None), song("rock", 0.9), song("metal", 0.1)]
    assert precision_at_k(songs, SAD_ROCK) == 0.5
    assert traceability(songs) == 0.75
    assert precision_at_k([], SAD_ROCK) is None and traceability([]) is None


def test_genre_diversity():
    d = genre_diversity(["pop", "pop", "rock", "Hip Hop"])
    assert d["songs"] == 4 and d["genres"] == 3
    assert d["top_genre"] == "pop" and d["top_share"] == 0.5
    assert 0 < d["entropy"] < 1
    assert genre_diversity(["pop", "pop"])["entropy"] == 0.0
    assert genre_diversity([])["genres"] == 0


def test_even_spread_has_entropy_one():
    assert genre_diversity(["pop", "rock", "jazz"])["entropy"] == pytest.approx(1.0)


def test_baseline_uses_csv_and_label_profile():
    ranked = baseline_playlist(load_songs(str(ROOT / "data" / "songs.csv")), SAD_ROCK, k=5)
    assert len(ranked) == 5
    assert ranked[0][1] >= ranked[-1][1]


def test_cost():
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert cost_usd("unknown-model", 1, 1) is None


def test_mean_skips_none():
    assert mean([1.0, None, 3.0]) == 2.0
    assert mean([None]) is None
