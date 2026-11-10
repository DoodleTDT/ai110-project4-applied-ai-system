from pathlib import Path

import pytest

from src.genres import GenreTaxonomy, TAXONOMY
from src.recommender import load_songs

SONGS_CSV = Path(__file__).parent.parent / "data" / "songs.csv"


# --- canonical ---

@pytest.mark.parametrize("label, expected", [
    ("Pop", "pop"),
    ("  indie   POP ", "indie pop"),
    ("pop indie", "indie pop"),        # word order
    ("hip hop", "hip-hop"),            # hyphen vs space
    ("HipHop", "hip-hop"),             # alias
    ("Alt-Rock", "alternative rock"),  # alias with hyphen
    ("R&B", "r&b"),
    ("rnb", "r&b"),
    ("lo-fi", "lofi"),
])
def test_canonical_maps_spellings_onto_the_list(label, expected):
    assert TAXONOMY.canonical(label) == expected


def test_canonical_keeps_unknown_labels_normalized():
    assert TAXONOMY.canonical("  Vapor Wave ") == "vapor wave"


# --- parent / ancestors ---

def test_parent_of_subgenre():
    assert TAXONOMY.parent("metal") == "rock"
    assert TAXONOMY.parent("Indie Pop") == "pop"


def test_parent_of_top_level_and_unknown_is_none():
    assert TAXONOMY.parent("rock") is None
    assert TAXONOMY.parent("vaporwave") is None


def test_ancestors_are_closest_first():
    assert TAXONOMY.ancestors("grunge") == ["alternative rock", "rock"]


# --- is_allowed ---

def test_is_allowed_accepts_listed_genres_and_aliases():
    assert TAXONOMY.is_allowed("metal")
    assert TAXONOMY.is_allowed("Hip Hop")


def test_is_allowed_rejects_unknown_genres():
    assert not TAXONOMY.is_allowed("vaporwave")
    assert not TAXONOMY.is_allowed("")


def test_every_starter_csv_genre_is_allowed():
    genres = {song["genre"] for song in load_songs(str(SONGS_CSV))}
    assert [genre for genre in genres if not TAXONOMY.is_allowed(genre)] == []


# --- matches ---

def test_matches_exact():
    assert TAXONOMY.matches(["rock"], "rock") == "rock"


def test_matches_parent_favorite():
    assert TAXONOMY.matches(["rock"], "metal") == "rock"


def test_specific_favorite_does_not_match_broader_genre():
    assert TAXONOMY.matches(["metal"], "rock") is None


def test_exact_match_wins_over_parent():
    assert TAXONOMY.matches(["rock", "metal"], "metal") == "metal"


def test_closest_parent_wins():
    assert TAXONOMY.matches(["rock", "alternative rock"], "grunge") == "alternative rock"


def test_matches_returns_none_for_unrelated_genres():
    assert TAXONOMY.matches(["jazz", "pop"], "metal") is None


def test_matches_accepts_unnormalized_favorites():
    assert TAXONOMY.matches(["  ROCK "], "Heavy Metal") == "rock"


# --- construction ---

def test_unknown_parent_is_rejected():
    with pytest.raises(ValueError, match="unknown parent"):
        GenreTaxonomy(parents={"metal": "rock"}, aliases={})


def test_alias_to_unknown_genre_is_rejected():
    with pytest.raises(ValueError, match="unknown genre"):
        GenreTaxonomy(parents={"rock": None}, aliases={"alt": "alternative rock"})


def test_custom_taxonomy():
    taxonomy = GenreTaxonomy(parents={"rock": None, "metal": "rock"}, aliases={})
    assert taxonomy.matches(["rock"], "metal") == "rock"
    assert not taxonomy.is_allowed("pop")
