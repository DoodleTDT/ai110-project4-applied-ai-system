"""Streamlit app tests: review and library tabs against a temporary database (no API calls)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app
from src.database import SongDatabase
from src.recommender import Song, UserProfile
from src.pipeline import PlaylistEntry, PlaylistResult

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def make_song(title, **overrides):
    values = dict(id=0, title=title, artist="Artist", genre="rock", mood="sad", energy=0.3, tempo_bpm=90,
                  valence=0.2, acousticness=0.6, source="lastfm", confidence=0.9,
                  estimated_fields=["energy", "valence"])
    values.update(overrides)
    return Song(**values)


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "app.db3")
    with SongDatabase(path) as db:
        db.add_pending(make_song("Verified Song", deezer_id=11), verified=True)
        db.add_pending(make_song("Missing Song", source="seed_csv"), verified=False)
        db.add_pending(make_song("Unchecked Song"), verified=None)
    monkeypatch.setenv("MMM_DB_PATH", path)
    return path


def run_app():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    return at.run()


def button(at, key):
    return next(b for b in at.button if b.key == key)


def pending_ids(path):
    with SongDatabase(path) as db:
        return {p.song.title: p.pending_id for p in db.list_pending()}


# --- app runs ---

def test_app_starts_without_errors(db_path):
    at = run_app()
    assert not at.exception
    assert at.title[0].value == "🎧 MusicMatchMaker"
    assert [t.label for t in at.tabs] == ["Playlist", "Pending review", "Library"]


def test_build_button_is_disabled_until_text_is_entered(db_path):
    at = run_app()
    build = next(b for b in at.button if b.label == "Build playlist")
    assert build.disabled


def test_request_box_enforces_the_length_limit(db_path):
    at = run_app()
    assert at.text_area[0].max_chars == app.MAX_REQUEST_CHARS


# --- review tab ---

def test_only_verified_songs_can_be_approved(db_path):
    at = run_app()
    ids = pending_ids(db_path)
    assert not button(at, f"approve_{ids['Verified Song']}").disabled
    at.radio[0].set_value("All").run()
    assert button(at, f"approve_{ids['Missing Song']}").disabled
    assert button(at, f"approve_{ids['Unchecked Song']}").disabled


def test_approve_moves_song_into_library(db_path):
    at = run_app()
    ids = pending_ids(db_path)
    button(at, f"approve_{ids['Verified Song']}").click().run()
    assert not at.exception
    with SongDatabase(db_path) as db:
        assert [s.title for s in db.get_songs()] == ["Verified Song"]
        assert "Verified Song" not in {p.song.title for p in db.list_pending()}


def test_reject_deletes_pending_song(db_path):
    at = run_app()
    at.radio[0].set_value("All").run()
    ids = pending_ids(db_path)
    button(at, f"reject_{ids['Unchecked Song']}").click().run()
    assert not at.exception
    assert "Unchecked Song" not in pending_ids(db_path)


def test_bulk_reject_needs_confirmation(db_path):
    at = run_app()
    assert button(at, "bulk_reject").disabled
    next(c for c in at.checkbox if c.key == "confirm_bulk_reject").check().run()
    button(at, "bulk_reject").click().run()
    assert not at.exception
    assert set(pending_ids(db_path)) == {"Verified Song", "Unchecked Song"}


# --- library tab ---

def test_library_shows_approved_songs(db_path):
    with SongDatabase(db_path) as db:
        db.approve(pending_ids(db_path)["Verified Song"])
    at = run_app()
    assert not at.exception
    assert any(m.label == "Approved songs" and m.value == "1" for m in at.metric)


# --- table helpers ---

def test_playlist_rows_show_status_and_reason():
    song = make_song("Creep")
    result = PlaylistResult(
        request="sad rock", profile=UserProfile(favorite_genres=["rock"], favorite_mood="sad", target_energy=0.3),
        entries=[PlaylistEntry(song=song, score=7.5, reason="Fits.", source="pending", pending_review=True)],
        used_external=True)
    [row] = app.playlist_rows(result)
    assert row["Status"] == "pending review"
    assert row["Why"] == "Fits."
    assert row["Score /9"] == 7.5


def test_pending_rows_label_verification():
    with SongDatabase(":memory:") as db:
        db.add_pending(make_song("A"), verified=False)
        [row] = app.pending_rows(db.list_pending())
    assert row["Deezer"] == "not found on Deezer"
    assert row["Estimated"] == "energy, valence"
