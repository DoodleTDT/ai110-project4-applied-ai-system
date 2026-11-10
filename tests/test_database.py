import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from scripts.seed_db import seed_from_csv
from src.database import DuplicateSongError, PendingSong, SongDatabase
from src.recommender import Song

SONGS_CSV = Path(__file__).parent.parent / "data" / "songs.csv"


@pytest.fixture
def db(tmp_path):
    database = SongDatabase(str(tmp_path / "test.db3"))
    yield database
    database.close()


def make_song(**overrides) -> Song:
    values = dict(id=0, title="Test Song", artist="Test Artist", genre="rock", mood="sad",
                  energy=0.3, tempo_bpm=90, valence=0.2, acousticness=0.6)
    values.update(overrides)
    return Song(**values)


# --- setup ---

def test_new_database_has_empty_tables(db):
    assert db.get_songs() == []
    assert db.list_pending() == []
    assert db.genre_counts() == {}


def test_create_tables_is_safe_to_run_twice(db):
    db.add_pending(make_song())
    db.create_tables()
    assert len(db.list_pending()) == 1


def test_data_persists_after_reopening(tmp_path):
    path = str(tmp_path / "persist.db3")
    with SongDatabase(path) as first:
        first.approve(first.add_pending(make_song()))
    with SongDatabase(path) as second:
        assert [song.title for song in second.get_songs()] == ["Test Song"]


# --- add_pending / list_pending ---

def test_add_pending_round_trips_all_fields(db):
    song = make_song(source="lastfm", mbid="abc-123", deezer_id=42,
                     preview_url="https://example.com/p.mp3", confidence=0.8,
                     estimated_fields=["energy", "valence"], tempo_bpm=None)
    pending_id = db.add_pending(song, verified=True)

    [pending] = db.list_pending()
    assert isinstance(pending, PendingSong)
    assert pending.pending_id == pending_id
    assert pending.verified is True
    assert isinstance(pending.proposed_at, datetime)
    stored = pending.song
    assert (stored.source, stored.mbid, stored.deezer_id) == ("lastfm", "abc-123", 42)
    assert stored.preview_url == "https://example.com/p.mp3"
    assert stored.confidence == 0.8
    assert stored.estimated_fields == ["energy", "valence"]
    assert stored.tempo_bpm is None


def test_verified_defaults_to_not_checked(db):
    db.add_pending(make_song())
    assert db.list_pending()[0].verified is None


def test_genre_is_stored_as_taxonomy_name(db):
    db.add_pending(make_song(genre="Hip Hop"))
    assert db.list_pending()[0].song.genre == "hip-hop"


def test_duplicate_pending_song_is_rejected(db):
    db.add_pending(make_song(title="Same Song", artist="Same Artist"))
    with pytest.raises(DuplicateSongError):
        db.add_pending(make_song(title="  same   SONG ", artist="same artist"))


def test_duplicate_mbid_is_rejected(db):
    db.add_pending(make_song(title="One", mbid="xyz"))
    with pytest.raises(DuplicateSongError):
        db.add_pending(make_song(title="Two", mbid="xyz"))


def test_songs_without_mbid_do_not_clash(db):
    db.add_pending(make_song(title="One"))
    db.add_pending(make_song(title="Two"))
    assert len(db.list_pending()) == 2


def test_pending_song_already_approved_is_rejected(db):
    db.approve(db.add_pending(make_song()))
    with pytest.raises(DuplicateSongError):
        db.add_pending(make_song(title="TEST SONG"))


@pytest.mark.parametrize("field, value", [
    ("energy", 1.5), ("energy", -0.1), ("valence", 2.0),
    ("acousticness", -1.0), ("confidence", 1.2), ("tempo_bpm", 0),
])
def test_out_of_range_values_are_rejected(db, field, value):
    with pytest.raises(ValueError, match="out of range"):
        db.add_pending(make_song(**{field: value}))
    assert db.list_pending() == []


def test_check_constraint_also_blocks_raw_sql(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "INSERT INTO songs (added_at, title, artist, title_key, artist_key, genre, mood, "
            "energy, valence, acousticness, source) "
            "VALUES ('now', 't', 'a', 't', 'a', 'rock', 'sad', 5.0, 0.5, 0.5, 'test')"
        )


# --- approve / reject ---

def test_approve_moves_song_into_songs(db):
    pending_id = db.add_pending(make_song(confidence=0.7, estimated_fields=["energy"]))
    new_id = db.approve(pending_id)

    assert db.list_pending() == []
    [song] = db.get_songs()
    assert song.id == new_id
    assert song.title == "Test Song"
    assert song.confidence == 0.7
    assert song.estimated_fields == ["energy"]
    assert isinstance(song.added_at, datetime)


def test_approve_unknown_id_raises_and_changes_nothing(db):
    db.add_pending(make_song())
    with pytest.raises(KeyError):
        db.approve(999)
    assert len(db.list_pending()) == 1
    assert db.get_songs() == []


def test_failed_approve_is_rolled_back(db):
    # Force a clash: a row in songs that bypassed add_pending's check
    pending_id = db.add_pending(make_song())
    db.conn.execute(
        "INSERT INTO songs (added_at, title, artist, title_key, artist_key, genre, mood, "
        "energy, valence, acousticness, source) "
        "VALUES ('2026-01-01T00:00:00+00:00', 'Test Song', 'Test Artist', 'test song', 'test artist', 'rock', 'sad', "
        "0.3, 0.2, 0.6, 'manual')"
    )
    db.conn.commit()

    with pytest.raises(DuplicateSongError):
        db.approve(pending_id)
    # The pending row is still there: nothing was half-moved
    assert [p.pending_id for p in db.list_pending()] == [pending_id]
    assert len(db.get_songs()) == 1


def test_reject_deletes_pending_song(db):
    pending_id = db.add_pending(make_song())
    db.reject(pending_id)
    assert db.list_pending() == []
    assert db.get_songs() == []


def test_reject_unknown_id_raises(db):
    with pytest.raises(KeyError):
        db.reject(999)


# --- genre_counts ---

def test_genre_counts_only_count_approved_songs(db):
    db.approve(db.add_pending(make_song(title="A", genre="rock")))
    db.approve(db.add_pending(make_song(title="B", genre="rock")))
    db.approve(db.add_pending(make_song(title="C", genre="Jazz")))
    db.add_pending(make_song(title="D", genre="pop"))
    assert db.genre_counts() == {"rock": 2, "jazz": 1}


# --- cache ---

def test_cache_round_trip(db):
    db.cache_put("lastfm:tag:rock", {"tracks": [{"name": "X"}]})
    assert db.cache_get("lastfm:tag:rock") == {"tracks": [{"name": "X"}]}


def test_cache_miss_returns_none(db):
    assert db.cache_get("missing") is None


def test_cache_put_overwrites(db):
    db.cache_put("key", {"v": 1})
    db.cache_put("key", {"v": 2})
    assert db.cache_get("key") == {"v": 2}


def test_cache_respects_max_age(db):
    db.cache_put("key", {"v": 1})
    db.conn.execute("UPDATE external_cache SET fetched_at = '2000-01-01T00:00:00+00:00'")
    assert db.cache_get("key", max_age_seconds=3600) is None
    assert db.cache_get("key") == {"v": 1}


# --- seed script ---

def test_seed_stages_all_csv_rows_as_pending(db):
    counts = seed_from_csv(str(SONGS_CSV), db)
    assert counts == {"added": 50, "skipped": 0}
    assert db.get_songs() == []

    pending = db.list_pending()
    assert len(pending) == 50
    assert all(p.song.source == "seed_csv" for p in pending)
    assert all(p.verified is None for p in pending)
    assert pending[0].song.title == "Sunrise City"


def test_seed_twice_skips_duplicates(db):
    seed_from_csv(str(SONGS_CSV), db)
    assert seed_from_csv(str(SONGS_CSV), db) == {"added": 0, "skipped": 50}
    assert len(db.list_pending()) == 50
