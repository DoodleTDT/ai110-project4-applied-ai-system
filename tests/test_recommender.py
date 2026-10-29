import math
from pathlib import Path

import pytest

from src.recommender import Song, UserProfile, Recommender, load_songs, score_song, recommend_songs

SONGS_CSV = Path(__file__).parent.parent / "data" / "songs.csv"

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genres=["pop"],
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genres=["pop"],
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# --- Edge cases ---

def make_song_dict(**overrides) -> dict:
    song = {
        "id": 1,
        "title": "Test Song",
        "artist": "Test Artist",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.5,
        "tempo_bpm": 100,
        "valence": 0.5,
        "acousticness": 0.5,
    }
    song.update(overrides)
    return song


def test_likes_acoustic_rejects_non_bool_string():
    with pytest.raises(TypeError):
        UserProfile(
            favorite_genres=["pop"],
            favorite_mood="happy",
            target_energy=0.5,
            # Wrong type on purpose: this test checks that it gets rejected
            likes_acoustic="false",  # pyright: ignore[reportArgumentType]
        )


def test_target_energy_rejects_nan():
    with pytest.raises(ValueError):
        UserProfile(
            favorite_genres=["pop"],
            favorite_mood="happy",
            target_energy=math.nan,
        )


def test_out_of_range_song_energy_never_scores_negative():
    user_prefs = {
        "favorite_genres": ["rock"],
        "favorite_mood": "sad",
        "target_energy": 0.0,
    }
    song = make_song_dict(energy=1.5)

    score, _ = score_song(user_prefs, song)
    assert score >= 0.0


def test_negative_k_returns_empty_list_and_prints_message(capsys):
    user_prefs = {
        "favorite_genres": ["pop"],
        "favorite_mood": "happy",
        "target_energy": 0.5,
    }
    songs = [
        make_song_dict(id=1, title="One"),
        make_song_dict(id=2, title="Two"),
        make_song_dict(id=3, title="Three"),
    ]

    results = recommend_songs(user_prefs, songs, k=-1)
    assert results == []
    assert "positive number" in capsys.readouterr().out


@pytest.mark.parametrize("second_song", [
    # Same id, different song
    Song(id=1, title="Second", artist="B", genre="lofi", mood="chill",
         energy=0.4, tempo_bpm=80, valence=0.6, acousticness=0.9),
    # Different id, same title and artist (case and spacing ignored)
    Song(id=2, title=" first ", artist="a", genre="lofi", mood="chill",
         energy=0.4, tempo_bpm=80, valence=0.6, acousticness=0.9),
])
def test_recommender_rejects_duplicate_songs(second_song):
    songs = [
        Song(id=1, title="First", artist="A", genre="pop", mood="happy",
             energy=0.8, tempo_bpm=120, valence=0.9, acousticness=0.2),
        second_song,
    ]

    with pytest.raises(ValueError, match="duplicate"):
        Recommender(songs)


# --- UserProfile validation ---

def make_user(**overrides) -> UserProfile:
    prefs = {"favorite_genres": ["pop"], "favorite_mood": "happy", "target_energy": 0.5}
    prefs.update(overrides)
    return UserProfile(**prefs)


def test_genres_are_normalized_and_deduped():
    assert make_user(favorite_genres=[" Pop ", "POP"]).favorite_genres == ["pop"]


def test_genre_inner_whitespace_is_collapsed():
    assert make_user(favorite_genres=["indie   pop"]).favorite_genres == ["indie pop"]


def test_single_genre_string_becomes_list():
    # A plain string is accepted for convenience even though the type says List[str]
    user = UserProfile(favorite_genres="pop", favorite_mood="happy", target_energy=0.5)  # pyright: ignore[reportArgumentType]
    assert user.favorite_genres == ["pop"]


def test_three_genres_are_accepted():
    assert make_user(favorite_genres=["pop", "rock", "jazz"]).favorite_genres == ["pop", "rock", "jazz"]


def test_four_genres_are_rejected():
    with pytest.raises(ValueError):
        make_user(favorite_genres=["pop", "rock", "jazz", "funk"])


def test_four_entries_that_dedupe_to_three_are_accepted():
    user = make_user(favorite_genres=["pop", "Pop", "rock", "jazz"])
    assert user.favorite_genres == ["pop", "rock", "jazz"]


@pytest.mark.parametrize("genres", [[], [""], ["  ", ""]])
def test_empty_or_blank_genres_are_rejected(genres):
    with pytest.raises(ValueError):
        make_user(favorite_genres=genres)


def test_genre_order_is_kept():
    assert make_user(favorite_genres=["rock", "pop"]).favorite_genres == ["rock", "pop"]


@pytest.mark.parametrize("given, expected", [(1.7, 1.0), (-0.3, 0.0), (0, 0.0), (1, 1.0)])
def test_target_energy_is_clamped(given, expected):
    assert make_user(target_energy=given).target_energy == expected


def test_target_energy_numeric_string_is_accepted():
    user = UserProfile(favorite_genres=["pop"], favorite_mood="happy", target_energy="0.8")  # pyright: ignore[reportArgumentType]
    assert user.target_energy == 0.8


def test_target_energy_non_numeric_string_is_rejected():
    with pytest.raises(ValueError):
        UserProfile(favorite_genres=["pop"], favorite_mood="happy", target_energy="abc")  # pyright: ignore[reportArgumentType]


# --- Genre matching ---

def genre_points(favorite_genres, song_genre):
    """Score with only genre able to vary: mood never matches, energy is fixed at 2.0."""
    user_prefs = {"favorite_genres": favorite_genres, "favorite_mood": "none", "target_energy": 0.5}
    score, reasons = score_song(user_prefs, make_song_dict(genre=song_genre, energy=0.5))
    return score - 2.0, reasons


def test_exact_genre_match_scores_full_points():
    points, _ = genre_points(["pop"], "pop")
    assert points == pytest.approx(4.0)


def test_favorite_contained_in_song_genre_scores_full_points():
    points, reasons = genre_points(["pop"], "indie pop")
    assert points == pytest.approx(4.0)
    assert 'genre match: "indie pop" contains "pop" (+4.0)' in reasons


def test_specific_favorite_does_not_match_broader_genre():
    points, _ = genre_points(["indie pop"], "pop")
    assert points == pytest.approx(0.0)


def test_partial_word_does_not_match():
    points, _ = genre_points(["rock"], "rockabilly")
    assert points == pytest.approx(0.0)


def test_word_order_does_not_matter():
    points, _ = genre_points(["pop indie"], "indie pop")
    assert points == pytest.approx(4.0)


def test_multiple_matching_favorites_do_not_stack():
    points, _ = genre_points(["pop", "indie pop"], "indie pop")
    assert points == pytest.approx(4.0)


def test_exact_match_wins_the_reason_text():
    _, reasons = genre_points(["pop", "indie pop"], "indie pop")
    assert 'genre match: "indie pop" (+4.0)' in reasons


def test_genre_with_symbols_matches_case_insensitively():
    points, _ = genre_points(["r&b"], "R&B")
    assert points == pytest.approx(4.0)


def test_hyphenated_genre_does_not_match_spaced_genre():
    points, _ = genre_points(["hip-hop"], "hip hop")
    assert points == pytest.approx(0.0)


# --- Mood ---

def test_mood_match_ignores_case_and_spacing():
    user_prefs = {"favorite_genres": ["rock"], "favorite_mood": "Chill ", "target_energy": 0.5}
    _, reasons = score_song(user_prefs, make_song_dict(mood="chill"))
    assert 'mood match: "chill" (+2.0)' in reasons


def test_mood_requires_exact_match():
    user_prefs = {"favorite_genres": ["rock"], "favorite_mood": "happy", "target_energy": 0.5}
    _, reasons = score_song(user_prefs, make_song_dict(mood="very happy"))
    assert not any(reason.startswith("mood match") for reason in reasons)


# --- Energy ---

def energy_only_score(target_energy, song_energy):
    user_prefs = {"favorite_genres": ["rock"], "favorite_mood": "none", "target_energy": target_energy}
    score, _ = score_song(user_prefs, make_song_dict(energy=song_energy))
    return score


def test_exact_energy_scores_full_points():
    assert energy_only_score(0.8, 0.8) == pytest.approx(2.0)


def test_maximum_energy_distance_scores_zero():
    assert energy_only_score(0.0, 1.0) == pytest.approx(0.0)


# --- Acousticness ---

def acoustic_points(likes_acoustic, acousticness):
    """Score with only acousticness able to vary: genre and mood never match, energy is 2.0."""
    user_prefs = {"favorite_genres": ["rock"], "favorite_mood": "none", "target_energy": 0.5}
    if likes_acoustic is not None:
        user_prefs["likes_acoustic"] = likes_acoustic
    score, reasons = score_song(user_prefs, make_song_dict(energy=0.5, acousticness=acousticness))
    return score - 2.0, reasons


@pytest.mark.parametrize("likes_acoustic, acousticness, expected", [
    (True, 0.9, 0.9),
    (False, 0.9, 0.1),
    (True, 0.0, 0.0),
    (True, 1.0, 1.0),
    (False, 0.0, 1.0),
    (False, 1.0, 0.0),
])
def test_acoustic_points_follow_preference(likes_acoustic, acousticness, expected):
    points, _ = acoustic_points(likes_acoustic, acousticness)
    assert points == pytest.approx(expected)


@pytest.mark.parametrize("likes_acoustic, acousticness, expected", [
    (False, 1.5, 0.0),
    (True, 1.5, 1.0),
    (True, -0.5, 0.0),
    (False, -0.5, 1.0),
])
def test_out_of_range_acousticness_is_clamped(likes_acoustic, acousticness, expected):
    points, _ = acoustic_points(likes_acoustic, acousticness)
    assert points == pytest.approx(expected)


def test_missing_acoustic_preference_scores_zero_with_no_reason():
    points, reasons = acoustic_points(None, 0.9)
    assert points == pytest.approx(0.0)
    assert not any(reason.startswith("acousticness") for reason in reasons)


# --- Score totals ---

def test_perfect_song_scores_nine():
    user_prefs = {"favorite_genres": ["pop"], "favorite_mood": "happy",
                  "target_energy": 0.7, "likes_acoustic": True}
    score, _ = score_song(user_prefs, make_song_dict(energy=0.7, acousticness=1.0))
    assert score == pytest.approx(9.0)


def test_opposite_song_scores_zero():
    user_prefs = {"favorite_genres": ["rock"], "favorite_mood": "sad",
                  "target_energy": 0.0, "likes_acoustic": True}
    score, _ = score_song(user_prefs, make_song_dict(energy=1.0, acousticness=0.0))
    assert score == pytest.approx(0.0)


def test_other_features_can_outvote_genre():
    user_prefs = {"favorite_genres": ["pop"], "favorite_mood": "chill",
                  "target_energy": 0.0, "likes_acoustic": True}
    genre_only = make_song_dict(id=1, title="Genre Only", genre="pop", mood="happy",
                                energy=1.0, acousticness=0.0)
    everything_else = make_song_dict(id=2, title="Everything Else", genre="rock", mood="chill",
                                     energy=0.0, acousticness=1.0)

    results = recommend_songs(user_prefs, [genre_only, everything_else], k=2)
    assert [song["title"] for song, _, _ in results] == ["Everything Else", "Genre Only"]


# --- Ranking and k ---

RANKING_PREFS = {"favorite_genres": ["pop"], "favorite_mood": "happy", "target_energy": 0.5}


def test_results_are_sorted_highest_first():
    songs = [
        make_song_dict(id=1, title="Low", genre="rock", mood="sad"),
        make_song_dict(id=2, title="High"),
        make_song_dict(id=3, title="Mid", mood="sad"),
    ]
    results = recommend_songs(RANKING_PREFS, songs, k=3)
    assert [song["title"] for song, _, _ in results] == ["High", "Mid", "Low"]


def test_tie_goes_to_closer_energy():
    # Both score exactly 2.0: Far loses 0.5 energy points but gains 0.5 acoustic points
    user_prefs = {"favorite_genres": ["rock"], "favorite_mood": "none",
                  "target_energy": 0.5, "likes_acoustic": True}
    songs = [
        make_song_dict(id=1, title="Far", energy=0.25, acousticness=0.5),
        make_song_dict(id=2, title="Close", energy=0.5, acousticness=0.0),
    ]
    results = recommend_songs(user_prefs, songs, k=2)
    assert results[0][1] == results[1][1]
    assert [song["title"] for song, _, _ in results] == ["Close", "Far"]


def test_full_tie_goes_to_lower_id():
    songs = [make_song_dict(id=2, title="Second"), make_song_dict(id=1, title="First")]
    results = recommend_songs(RANKING_PREFS, songs, k=2)
    assert [song["id"] for song, _, _ in results] == [1, 2]


def test_k_larger_than_song_list_returns_all_songs():
    songs = [make_song_dict(id=1, title="One"), make_song_dict(id=2, title="Two")]
    assert len(recommend_songs(RANKING_PREFS, songs, k=10)) == 2


def test_k_zero_returns_empty_list():
    assert recommend_songs(RANKING_PREFS, [make_song_dict()], k=0) == []


def test_empty_song_list_returns_empty_list():
    assert recommend_songs(RANKING_PREFS, [], k=5) == []


# --- Recommender class ---

def test_recommend_returns_song_objects():
    user = make_user(target_energy=0.8)
    results = make_small_recommender().recommend(user, k=2)
    assert all(isinstance(song, Song) for song in results)


def test_explanation_includes_max_score():
    user = make_user(target_energy=0.8)
    rec = make_small_recommender()
    assert "/9.00" in rec.explain_recommendation(user, rec.songs[0])


def test_recommender_matches_functional_version():
    song_dicts = load_songs(str(SONGS_CSV))
    user_prefs = {"favorite_genres": ["pop", "lofi"], "favorite_mood": "chill",
                  "target_energy": 0.4, "likes_acoustic": True}

    functional_ids = [song["id"] for song, _, _ in recommend_songs(user_prefs, song_dicts, k=10)]
    oop_results = Recommender([Song(**song) for song in song_dicts]).recommend(UserProfile(**user_prefs), k=10)
    assert [song.id for song in oop_results] == functional_ids


# --- load_songs ---

CSV_HEADER = "id, title, artist, genre, mood, energy, tempo_bpm, valence, acousticness"


def test_load_songs_reads_real_data_file():
    songs = load_songs(str(SONGS_CSV))
    assert len(songs) == 50

    first = songs[0]
    assert first["title"] == "Sunrise City"
    assert first["genre"] == "pop"  # padding stripped
    assert isinstance(first["id"], int)
    for field in ("energy", "tempo_bpm", "valence", "acousticness"):
        assert isinstance(first[field], float)


def test_load_songs_handles_crlf_line_endings(tmp_path):
    csv_file = tmp_path / "songs.csv"
    rows = [CSV_HEADER, "1,  Test,  Artist,  pop,  happy,  0.5,  100,  0.5,  0.5"]
    csv_file.write_bytes("\r\n".join(rows).encode() + b"\r\n")

    songs = load_songs(str(csv_file))
    assert len(songs) == 1
    assert songs[0]["acousticness"] == 0.5


def test_load_songs_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_songs("does/not/exist.csv")


def test_load_songs_bad_number_raises(tmp_path):
    csv_file = tmp_path / "songs.csv"
    csv_file.write_text(CSV_HEADER + "\n" + "1, Test, Artist, pop, happy, high, 100, 0.5, 0.5\n")
    with pytest.raises(ValueError):
        load_songs(str(csv_file))
