"""
SQLite storage for MusicMatchMaker (a single .db3 file, built-in sqlite3).

Tables:
    songs           approved songs, used for recommendations
    pending_songs   staged songs waiting for manual approve/reject
    external_cache  cached Last.fm / Deezer API responses

New songs always enter pending_songs first. approve() moves a song into
songs in one transaction, so a song is never half-moved.

The database backs up the Python-side checks: a UNIQUE constraint on the
normalized title + artist (and on mbid), and CHECK constraints that keep
energy, valence, acousticness and confidence between 0 and 1.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

try:
    from src.genres import TAXONOMY, normalize
    from src.recommender import Song
except ModuleNotFoundError:
    from genres import TAXONOMY, normalize
    from recommender import Song

DEFAULT_DB_PATH = "data/musicmatchmaker.db3"


class DuplicateSongError(ValueError):
    """Raised when a song (same normalized title + artist, or same mbid) already exists."""


@dataclass
class PendingSong:
    """A staged song plus its review information."""
    pending_id: int
    song: Song
    verified: Optional[bool]  # None = not checked yet
    proposed_at: datetime


# Columns shared by songs and pending_songs
_SONG_COLUMNS = """
    title             TEXT    NOT NULL,
    artist            TEXT    NOT NULL,
    title_key         TEXT    NOT NULL,
    artist_key        TEXT    NOT NULL,
    genre             TEXT    NOT NULL,
    mood              TEXT    NOT NULL,
    energy            REAL    NOT NULL CHECK (energy BETWEEN 0 AND 1),
    tempo_bpm         REAL             CHECK (tempo_bpm IS NULL OR tempo_bpm > 0),
    valence           REAL    NOT NULL CHECK (valence BETWEEN 0 AND 1),
    acousticness      REAL    NOT NULL CHECK (acousticness BETWEEN 0 AND 1),
    source            TEXT    NOT NULL,
    mbid              TEXT    UNIQUE,
    deezer_id         INTEGER,
    preview_url       TEXT,
    confidence        REAL             CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    estimated_fields  TEXT    NOT NULL DEFAULT '[]',
    UNIQUE (title_key, artist_key)
"""

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS songs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    added_at          TEXT    NOT NULL,
    {_SONG_COLUMNS}
);

CREATE TABLE IF NOT EXISTS pending_songs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    proposed_at       TEXT    NOT NULL,
    verified          INTEGER          CHECK (verified IS NULL OR verified IN (0, 1)),
    {_SONG_COLUMNS}
);

CREATE TABLE IF NOT EXISTS external_cache (
    key               TEXT    PRIMARY KEY,
    data              TEXT    NOT NULL,
    fetched_at        TEXT    NOT NULL
);
"""

# Song fields stored in both tables, in insert order
_FIELDS = [
    "title", "artist", "title_key", "artist_key", "genre", "mood", "energy",
    "tempo_bpm", "valence", "acousticness", "source", "mbid", "deezer_id",
    "preview_url", "confidence", "estimated_fields",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _song_values(song: Song) -> Dict:
    """Converts a Song into column values (genre mapped onto the taxonomy)."""
    return {
        "title": song.title.strip(),
        "artist": song.artist.strip(),
        "title_key": normalize(song.title),
        "artist_key": normalize(song.artist),
        "genre": TAXONOMY.canonical(song.genre),
        "mood": normalize(song.mood),
        "energy": float(song.energy),
        "tempo_bpm": None if song.tempo_bpm is None else float(song.tempo_bpm),
        "valence": float(song.valence),
        "acousticness": float(song.acousticness),
        "source": song.source,
        "mbid": song.mbid or None,
        "deezer_id": song.deezer_id,
        "preview_url": song.preview_url,
        "confidence": song.confidence,
        "estimated_fields": json.dumps(list(song.estimated_fields)),
    }


def _row_to_song(row: sqlite3.Row, added_at: Optional[str] = None) -> Song:
    return Song(
        id=row["id"],
        title=row["title"],
        artist=row["artist"],
        genre=row["genre"],
        mood=row["mood"],
        energy=row["energy"],
        tempo_bpm=row["tempo_bpm"],
        valence=row["valence"],
        acousticness=row["acousticness"],
        source=row["source"],
        mbid=row["mbid"],
        deezer_id=row["deezer_id"],
        preview_url=row["preview_url"],
        confidence=row["confidence"],
        estimated_fields=json.loads(row["estimated_fields"]),
        added_at=datetime.fromisoformat(added_at) if added_at else None,
    )


def _translate_integrity_error(error: sqlite3.IntegrityError, song: Song) -> Exception:
    message = str(error)
    if "UNIQUE" in message:
        return DuplicateSongError(f'duplicate song: "{song.title}" by {song.artist} ({message})')
    if "CHECK" in message:
        return ValueError(f'song "{song.title}" has a value out of range ({message})')
    return error


class SongDatabase:
    """Owns the songs, pending_songs and external_cache tables."""

    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()

    # --- setup ---

    def create_tables(self) -> None:
        with self.conn:
            self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SongDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- approved songs ---

    def get_songs(self) -> List[Song]:
        rows = self.conn.execute("SELECT * FROM songs ORDER BY id").fetchall()
        return [_row_to_song(row, row["added_at"]) for row in rows]

    def genre_counts(self) -> Dict[str, int]:
        """Number of approved songs per genre (used by the coverage check)."""
        rows = self.conn.execute("SELECT genre, COUNT(*) AS n FROM songs GROUP BY genre")
        return {row["genre"]: row["n"] for row in rows}

    # --- pending songs ---

    def _exists_in_songs(self, values: Dict) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM songs WHERE (title_key = ? AND artist_key = ?) OR (mbid IS NOT NULL AND mbid = ?)",
            (values["title_key"], values["artist_key"], values["mbid"]),
        ).fetchone()
        return row is not None

    def add_pending(self, song: Song, verified: Optional[bool] = None) -> int:
        """
        Stages a song for review and returns its pending id. song.id is ignored.
        Raises DuplicateSongError if the song is already pending or approved,
        and ValueError if a value is out of range.
        """
        values = _song_values(song)
        if self._exists_in_songs(values):
            raise DuplicateSongError(f'"{song.title}" by {song.artist} is already in songs')

        values["proposed_at"] = _now().isoformat()
        values["verified"] = None if verified is None else int(bool(verified))
        columns = ", ".join(values)
        placeholders = ", ".join(f":{name}" for name in values)
        try:
            with self.conn:
                cursor = self.conn.execute(
                    f"INSERT INTO pending_songs ({columns}) VALUES ({placeholders})", values
                )
        except sqlite3.IntegrityError as error:
            raise _translate_integrity_error(error, song) from error
        return int(cursor.lastrowid)  # type: ignore[arg-type]

    def list_pending(self) -> List[PendingSong]:
        rows = self.conn.execute("SELECT * FROM pending_songs ORDER BY id").fetchall()
        return [
            PendingSong(
                pending_id=row["id"],
                song=_row_to_song(row),
                verified=None if row["verified"] is None else bool(row["verified"]),
                proposed_at=datetime.fromisoformat(row["proposed_at"]),
            )
            for row in rows
        ]

    def set_verification(self, pending_id: int, verified: bool, deezer_id: Optional[int] = None,
                         preview_url: Optional[str] = None, tempo_bpm: Optional[float] = None) -> None:
        """
        Records an existence check on a pending song. Deezer fields are only
        filled in when given; tempo_bpm only fills a missing tempo.
        Raises KeyError if the pending id does not exist.
        """
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE pending_songs SET verified = ?, "
                "deezer_id = COALESCE(?, deezer_id), "
                "preview_url = COALESCE(?, preview_url), "
                "tempo_bpm = COALESCE(tempo_bpm, ?) "
                "WHERE id = ?",
                (int(bool(verified)), deezer_id, preview_url, tempo_bpm, pending_id),
            )
        if cursor.rowcount == 0:
            raise KeyError(f"no pending song with id {pending_id}")

    def approve(self, pending_id: int) -> int:
        """
        Moves a pending song into songs in one transaction and returns its new id.
        Raises KeyError if the pending id does not exist. On any error nothing changes.
        """
        columns = ", ".join(_FIELDS)
        try:
            with self.conn:
                cursor = self.conn.execute(
                    f"INSERT INTO songs (added_at, {columns}) "
                    f"SELECT ?, {columns} FROM pending_songs WHERE id = ?",
                    (_now().isoformat(), pending_id),
                )
                if cursor.rowcount == 0:
                    raise KeyError(f"no pending song with id {pending_id}")
                new_id = int(cursor.lastrowid)  # type: ignore[arg-type]
                self.conn.execute("DELETE FROM pending_songs WHERE id = ?", (pending_id,))
        except sqlite3.IntegrityError as error:
            raise DuplicateSongError(f"pending song {pending_id} is already in songs ({error})") from error
        return new_id

    def reject(self, pending_id: int) -> None:
        """Deletes a pending song. Raises KeyError if the pending id does not exist."""
        with self.conn:
            cursor = self.conn.execute("DELETE FROM pending_songs WHERE id = ?", (pending_id,))
        if cursor.rowcount == 0:
            raise KeyError(f"no pending song with id {pending_id}")

    # --- external API cache ---

    def cache_get(self, key: str, max_age_seconds: Optional[float] = None) -> Optional[dict]:
        """Returns cached data for key, or None if missing or older than max_age_seconds."""
        row = self.conn.execute(
            "SELECT data, fetched_at FROM external_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        if max_age_seconds is not None:
            age = (_now() - datetime.fromisoformat(row["fetched_at"])).total_seconds()
            if age > max_age_seconds:
                return None
        return json.loads(row["data"])

    def cache_put(self, key: str, data: dict) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO external_cache (key, data, fetched_at) VALUES (?, ?, ?)",
                (key, json.dumps(data), _now().isoformat()),
            )
