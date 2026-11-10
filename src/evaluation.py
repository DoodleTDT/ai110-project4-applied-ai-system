"""
Evaluation metrics for MusicMatchMaker (used by scripts/evaluate.py).

Each test prompt in eval/prompts.json is hand-labeled with the genres and
moods a correct answer may use, the energy range it should fall in, and the
acoustic preference (null = not stated).

Metrics:
    parse accuracy      did RequestParser understand the request?
    precision@k         share of playlist songs that fit the labels
                        (genre related to a labeled genre AND energy in range)
    traceability        share of playlist songs confirmed on Deezer
    unverifiable rate   share of external candidates Deezer could not find,
                        and the same for the starter CSV
    genre diversity     distinct genres, largest-genre share and normalized
                        entropy of the song pool, before and after
    baseline            the original rigid scorer on data/songs.csv with the
                        hand-labeled profile, scored with the same precision rule

Precision uses each song's stored genre and energy. For songs from Last.fm
those values are estimated by Claude, so precision measures agreement with
those estimates, not a listening test.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from src.genres import TAXONOMY, normalize
    from src.recommender import Song, UserProfile, recommend_songs
except ModuleNotFoundError:
    from genres import TAXONOMY, normalize
    from recommender import Song, UserProfile, recommend_songs

ENERGY_MARGIN = 0.1  # songs count as in range this far outside the labeled range

# USD per million input / output tokens
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


@dataclass
class Label:
    id: str
    prompt: str
    genres: List[str]
    moods: List[str]
    energy: Sequence[float]
    acoustic: Optional[bool] = None


def load_labels(path: str) -> List[Label]:
    with open(path, encoding="utf-8") as f:
        return [Label(**item) for item in json.load(f)]


# --- matching rules ---

def genre_related(genre: str, labeled: Iterable[str]) -> bool:
    """True if genre is a labeled genre, a subgenre of one, or a parent of one."""
    genre = TAXONOMY.canonical(genre)
    for wanted in labeled:
        wanted = TAXONOMY.canonical(wanted)
        if TAXONOMY.matches([wanted], genre) or TAXONOMY.matches([genre], wanted):
            return True
    return False


def energy_in_range(energy: float, labeled: Sequence[float], margin: float = ENERGY_MARGIN) -> bool:
    low, high = labeled
    return low - margin <= float(energy) <= high + margin


def song_fits(song: Song, label: Label) -> bool:
    return genre_related(song.genre, label.genres) and energy_in_range(song.energy, label.energy)


# --- parse accuracy ---

def parse_checks(profile: UserProfile, label: Label) -> Dict[str, Optional[bool]]:
    """Per-field parse check. acoustic is None (not scored) when the label doesn't state it."""
    return {
        "genre": any(genre_related(g, label.genres) for g in profile.favorite_genres),
        "mood": normalize(profile.favorite_mood) in {normalize(m) for m in label.moods},
        "energy": energy_in_range(profile.target_energy, label.energy, margin=0.0),
        "acoustic": None if label.acoustic is None else profile.likes_acoustic == label.acoustic,
    }


# --- playlist metrics ---

def precision_at_k(songs: Sequence[Song], label: Label) -> Optional[float]:
    if not songs:
        return None
    return sum(song_fits(s, label) for s in songs) / len(songs)


def traceability(songs: Sequence[Song]) -> Optional[float]:
    """Share of songs with a Deezer track id (confirmed to exist)."""
    if not songs:
        return None
    return sum(s.deezer_id is not None for s in songs) / len(songs)


def label_profile(label: Label) -> UserProfile:
    """The hand-labeled request as a UserProfile (for the rigid baseline)."""
    low, high = label.energy
    return UserProfile(favorite_genres=label.genres[:3], favorite_mood=label.moods[0],
                       target_energy=(low + high) / 2, likes_acoustic=label.acoustic)


def baseline_playlist(csv_songs: List[Dict[str, Any]], label: Label, k: int) -> List[tuple]:
    """The original system: rigid scorer over the 50-song CSV."""
    profile = label_profile(label)
    prefs = {"favorite_genres": profile.favorite_genres, "favorite_mood": profile.favorite_mood,
             "target_energy": profile.target_energy, "likes_acoustic": profile.likes_acoustic}
    return recommend_songs(prefs, csv_songs, k=k)


# --- diversity ---

def genre_diversity(genres: Iterable[str]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for genre in genres:
        genre = TAXONOMY.canonical(genre)
        counts[genre] = counts.get(genre, 0) + 1
    total = sum(counts.values())
    if not total:
        return {"songs": 0, "genres": 0, "top_genre": None, "top_share": None, "entropy": None}
    top_genre, top_count = max(counts.items(), key=lambda item: item[1])
    entropy = -sum((n / total) * math.log(n / total) for n in counts.values())
    normalized = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
    return {"songs": total, "genres": len(counts), "top_genre": top_genre,
            "top_share": round(top_count / total, 3), "entropy": round(normalized, 3)}


# --- cost ---

def cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


@dataclass
class PromptResult:
    id: str
    prompt: str
    parse: Dict[str, Optional[bool]]
    profile: Dict[str, Any]
    used_external: bool
    staged: int
    checked_on_deezer: int
    not_found_on_deezer: int
    precision: Optional[float]
    traceability: Optional[float]
    mean_score: Optional[float]
    baseline_precision: Optional[float]
    baseline_mean_score: Optional[float]
    tokens: Dict[str, Any]
    cost_usd: Optional[float]
    playlist: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
