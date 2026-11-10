"""
Guardrails that run before any song is staged in pending_songs.

SongValidator checks four things:
    verify          the song exists on Deezer (title + artist match)
    validate_schema numbers are in range and the genre is on the controlled list
    is_duplicate    the song is not already known (ignores "(Remastered)" etc.)
    confidence      0-1 score of how well the stored data is backed by evidence

review() runs all of them and returns one ValidationResult, so the agent
(step 6) can stage accepted songs and log why others were rejected.
The database constraints in database.py back these checks up.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

try:
    from src.external import DeezerMatch, comparable
    from src.genres import TAXONOMY, GenreTaxonomy, normalize
    from src.recommender import Song
except ModuleNotFoundError:
    from external import DeezerMatch, comparable
    from genres import TAXONOMY, GenreTaxonomy, normalize
    from recommender import Song

# Confidence weights (sum to 1.0). A song Deezer cannot verify scores 0.
VERIFIED_WEIGHT = 0.5       # Deezer confirms title + artist
EXACT_TITLE_WEIGHT = 0.1    # title matches exactly, not only after stripping "(Remastered)" etc.
MBID_WEIGHT = 0.1           # Last.fm gave a MusicBrainz id: a second source knows the song
MEASURED_TEMPO_WEIGHT = 0.1 # tempo comes from Deezer, not missing or estimated
GENRE_TAG_WEIGHT = 0.1      # the genre is backed by the song's Last.fm tags
MOOD_TAG_WEIGHT = 0.1       # the mood is backed by the song's Last.fm tags

# Fields an LLM (or other estimate) may fill in
ESTIMABLE_FIELDS = {"genre", "mood", "energy", "valence", "acousticness", "tempo_bpm"}

MAX_TEMPO_BPM = 300.0


@dataclass
class ValidationResult:
    """Outcome of review(): whether the song may be staged, and why."""
    accepted: bool
    reasons: List[str]                      # why it was rejected (empty when accepted)
    confidence: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    match: Optional[DeezerMatch] = None


def _in_unit_range(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number) and 0.0 <= number <= 1.0


class SongValidator:
    def __init__(self, deezer=None, taxonomy: GenreTaxonomy = TAXONOMY):
        self.deezer = deezer
        self.taxonomy = taxonomy

    # --- verify ---

    def verify(self, candidate: Dict) -> Optional[DeezerMatch]:
        """
        Looks the candidate (needs "title" and "artist") up on Deezer.
        Returns the match, or None if the song could not be found.
        ExternalAPIError is passed on: "could not check" is not "does not exist".
        """
        if self.deezer is None:
            raise RuntimeError("SongValidator needs a DeezerClient to verify songs")
        title = str(candidate.get("title", "")).strip()
        artist = str(candidate.get("artist", "")).strip()
        if not title or not artist:
            return None
        return self.deezer.search(title, artist)

    # --- schema ---

    def schema_errors(self, song: Song) -> List[str]:
        """Returns every schema problem with the song (empty list = valid)."""
        errors = []
        if not str(song.title).strip():
            errors.append("title is empty")
        if not str(song.artist).strip():
            errors.append("artist is empty")
        if not self.taxonomy.is_allowed(song.genre):
            errors.append(f'genre "{song.genre}" is not on the controlled genre list')
        if not normalize(song.mood):
            errors.append("mood is empty")
        for name in ("energy", "valence", "acousticness"):
            value = getattr(song, name)
            if not _in_unit_range(value):
                errors.append(f"{name} must be a number between 0 and 1, got {value!r}")
        if song.tempo_bpm is not None:
            try:
                tempo = float(song.tempo_bpm)
                if math.isnan(tempo) or not 0 < tempo <= MAX_TEMPO_BPM:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"tempo_bpm must be between 0 and {MAX_TEMPO_BPM:.0f}, got {song.tempo_bpm!r}")
        if song.confidence is not None and not _in_unit_range(song.confidence):
            errors.append(f"confidence must be between 0 and 1, got {song.confidence!r}")
        unknown = set(song.estimated_fields) - ESTIMABLE_FIELDS
        if unknown:
            errors.append(f"estimated_fields has unknown fields: {sorted(unknown)}")
        if not str(song.source).strip():
            errors.append("source is empty")
        return errors

    def validate_schema(self, song: Song) -> bool:
        return not self.schema_errors(song)

    # --- duplicates ---

    def is_duplicate(self, song: Song, existing: Iterable[Song]) -> bool:
        """
        True if an existing song has the same mbid, the same Deezer id, or the
        same title + artist once case, punctuation and suffixes like
        "(Remastered 2011)" or "- Live" are ignored.
        """
        key = (comparable(song.title), comparable(song.artist))
        for other in existing:
            if song.mbid and other.mbid and song.mbid == other.mbid:
                return True
            if song.deezer_id and other.deezer_id and song.deezer_id == other.deezer_id:
                return True
            if (comparable(other.title), comparable(other.artist)) == key:
                return True
        return False

    # --- confidence ---

    def _genre_backed_by(self, genre: str, tags: Iterable[str]) -> bool:
        """A tag backs the genre if it is the genre, a parent of it, or a subgenre of it."""
        genre = self.taxonomy.canonical(genre)
        family = {genre, *self.taxonomy.ancestors(genre)}
        for tag in tags:
            tag_genre = self.taxonomy.canonical(tag)
            if tag_genre in family or genre in self.taxonomy.ancestors(tag_genre):
                return True
        return False

    def confidence_breakdown(self, song: Song, match: Optional[DeezerMatch],
                             tags: Iterable[str] = ()) -> Dict[str, float]:
        """Points earned per piece of evidence (see the weights at the top of this file)."""
        if match is None:
            return {"verified": 0.0}
        tags = [normalize(tag) for tag in tags]
        tempo_measured = (song.tempo_bpm is not None and "tempo_bpm" not in song.estimated_fields
                          and match.bpm is not None)
        return {
            "verified": VERIFIED_WEIGHT,
            "exact_title": EXACT_TITLE_WEIGHT if normalize(match.title) == normalize(song.title) else 0.0,
            "mbid": MBID_WEIGHT if song.mbid else 0.0,
            "measured_tempo": MEASURED_TEMPO_WEIGHT if tempo_measured else 0.0,
            "genre_tag": GENRE_TAG_WEIGHT if self._genre_backed_by(song.genre, tags) else 0.0,
            "mood_tag": MOOD_TAG_WEIGHT if normalize(song.mood) in tags else 0.0,
        }

    def confidence(self, song: Song, match: Optional[DeezerMatch] = None,
                   tags: Iterable[str] = ()) -> float:
        """0-1 score: how well the song's data is backed by the APIs. 0 if unverified."""
        return round(sum(self.confidence_breakdown(song, match, tags).values()), 4)

    # --- all checks together ---

    def review(self, song: Song, existing: Iterable[Song], match: Optional[DeezerMatch],
               tags: Iterable[str] = ()) -> ValidationResult:
        """
        Runs every guardrail on a song. match is the result of verify() (None =
        not found). The song is accepted only if it is verified, passes the
        schema check and is not a duplicate.
        """
        reasons = []
        if match is None:
            reasons.append("not found on Deezer")
        reasons.extend(self.schema_errors(song))
        if self.is_duplicate(song, existing):
            reasons.append("duplicate of a song already in the database")
        breakdown = self.confidence_breakdown(song, match, tags)
        return ValidationResult(
            accepted=not reasons,
            reasons=reasons,
            confidence=round(sum(breakdown.values()), 4),
            breakdown=breakdown,
            match=match,
        )
