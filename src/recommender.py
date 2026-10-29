import csv
import math
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict

# Scoring weights (max total score = 9.0)
GENRE_WEIGHT = 4.0
MOOD_WEIGHT = 2.0
ENERGY_WEIGHT = 2.0
ACOUSTIC_WEIGHT = 1.0

MAX_FAVORITE_GENRES = 3

def _normalize(label: str) -> str:
    """Lowercases a label and collapses surrounding/inner whitespace."""
    return " ".join(str(label).strip().lower().split())

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))

def _check_duplicates(songs: List[Dict]) -> None:
    """Raises ValueError if two songs share an id, or share a title and artist."""
    seen_ids = set()
    seen_tracks = set()
    for song in songs:
        track = (_normalize(song["title"]), _normalize(song["artist"]))
        if song["id"] in seen_ids:
            raise ValueError(f"duplicate song id: {song['id']}")
        if track in seen_tracks:
            raise ValueError(f'duplicate song: "{song["title"]}" by {song["artist"]}')
        seen_ids.add(song["id"])
        seen_tracks.add(track)

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genres: List[str]
    favorite_mood: str
    target_energy: float
    likes_acoustic: Optional[bool] = None

    def __post_init__(self):
        genres: Union[str, List[str]] = self.favorite_genres
        if isinstance(genres, str):
            genres = [genres]

        # Normalize and dedupe while keeping the user's order
        normalized = []
        for genre in genres:
            genre = _normalize(genre)
            if genre and genre not in normalized:
                normalized.append(genre)

        if not normalized:
            raise ValueError("favorite_genres must contain at least one genre")
        if len(normalized) > MAX_FAVORITE_GENRES:
            raise ValueError(
                f"favorite_genres can list at most {MAX_FAVORITE_GENRES} genres, got {len(normalized)}"
            )

        if self.likes_acoustic is not None and not isinstance(self.likes_acoustic, bool):
            raise TypeError(
                f"likes_acoustic must be True, False, or None, got {self.likes_acoustic!r}"
            )

        target_energy = float(self.target_energy)
        if math.isnan(target_energy):
            raise ValueError("target_energy must be a number, got NaN")

        self.favorite_genres = normalized
        self.favorite_mood = _normalize(self.favorite_mood)
        self.target_energy = _clamp(target_energy)

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        _check_duplicates([asdict(song) for song in songs])
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        ranked = recommend_songs(asdict(user), [asdict(song) for song in self.songs], k)
        songs_by_id = {song.id: song for song in self.songs}
        return [songs_by_id[song_dict["id"]] for song_dict, _, _ in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        score, reasons = score_song(asdict(user), asdict(song))
        return f"Score {score:.2f}/9.00: " + "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            row = {key.strip(): value.strip() for key, value in row.items()}
            songs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs

def _match_genre(favorite_genres: List[str], song_genre: str) -> Optional[str]:
    """
    Returns the favorite genre that matches the song's genre, or None.
    A favorite matches when all of its words appear in the song's genre,
    so "pop" matches "indie pop", but "indie pop" does not match "pop".
    """
    song_words = set(song_genre.split())
    if song_genre in favorite_genres:
        return song_genre
    for favorite in favorite_genres:
        if set(favorite.split()) <= song_words:
            return favorite
    return None

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    user = UserProfile(**user_prefs)
    score = 0.0
    reasons = []

    # Genre: full points if any favorite matches; matches never stack
    song_genre = _normalize(song["genre"])
    matched = _match_genre(user.favorite_genres, song_genre)
    if matched == song_genre:
        score += GENRE_WEIGHT
        reasons.append(f'genre match: "{song_genre}" (+{GENRE_WEIGHT:.1f})')
    elif matched:
        score += GENRE_WEIGHT
        reasons.append(f'genre match: "{song_genre}" contains "{matched}" (+{GENRE_WEIGHT:.1f})')

    # Mood: exact match only
    song_mood = _normalize(song["mood"])
    if song_mood == user.favorite_mood:
        score += MOOD_WEIGHT
        reasons.append(f'mood match: "{song_mood}" (+{MOOD_WEIGHT:.1f})')

    # Energy: closer to the target earns more, never below 0
    song_energy = float(song["energy"])
    energy_points = max(0.0, ENERGY_WEIGHT * (1 - abs(song_energy - user.target_energy)))
    score += energy_points
    reasons.append(f"energy {song_energy:.2f} vs target {user.target_energy:.2f} (+{energy_points:.2f})")

    # Acousticness: optional, rewards the end of the scale the user prefers.
    # Clamped so out-of-range data stays within 0 to ACOUSTIC_WEIGHT points.
    if user.likes_acoustic is not None:
        acousticness = float(song["acousticness"])
        clamped = _clamp(acousticness)
        if user.likes_acoustic:
            acoustic_points = ACOUSTIC_WEIGHT * clamped
            preference = "likes acoustic"
        else:
            acoustic_points = ACOUSTIC_WEIGHT * (1 - clamped)
            preference = "prefers non-acoustic"
        score += acoustic_points
        reasons.append(f"acousticness {acousticness:.2f}, {preference} (+{acoustic_points:.2f})")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    if k < 0:
        print(f"Please enter a positive number for k (got {k}).")
        return []

    _check_duplicates(songs)
    target_energy = _clamp(user_prefs["target_energy"])
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, "; ".join(reasons)))

    # Highest score first; ties go to closer energy, then lower id
    scored.sort(key=lambda item: (-item[1], abs(float(item[0]["energy"]) - target_energy), item[0]["id"]))
    return scored[:k]
