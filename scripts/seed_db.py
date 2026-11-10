"""
One-time seed: copies the starter CSV into pending_songs (not songs), then
checks each row on Deezer.

Every row is staged with source='seed_csv' and goes through the normal
approve/reject review. data/songs.csv is never modified, because
tests/test_recommender.py loads it directly.

The Deezer check records verified = True/False plus the Deezer track id and
preview URL on a match. The starter CSV appears to be made up, so most rows
are expected to fail. Rows whose check could not run (network error, rate
limit) keep verified = None and are retried the next time the seed runs.

Usage (from the project root):
    python scripts/seed_db.py [--csv data/songs.csv] [--db data/musicmatchmaker.db3] [--no-verify]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import DEFAULT_DB_PATH, DuplicateSongError, SongDatabase  # noqa: E402
from src.external import DeezerClient, ExternalAPIError  # noqa: E402
from src.recommender import Song, load_songs  # noqa: E402

DEFAULT_CSV_PATH = "data/songs.csv"


def verify_pending(db: SongDatabase, deezer: DeezerClient, only_unchecked: bool = True) -> Dict[str, int]:
    """
    Runs the Deezer title + artist check on pending songs and records the result.
    Returns counts: {"verified": n, "not_found": n, "errors": n}.
    """
    counts = {"verified": 0, "not_found": 0, "errors": 0}
    for pending in db.list_pending():
        if only_unchecked and pending.verified is not None:
            continue
        song = pending.song
        try:
            match = deezer.search(song.title, song.artist)
        except ExternalAPIError as error:
            print(f'  could not check "{song.title}" by {song.artist}: {error}')
            counts["errors"] += 1
            continue
        if match is None:
            db.set_verification(pending.pending_id, False)
            counts["not_found"] += 1
        else:
            db.set_verification(pending.pending_id, True, deezer_id=match.deezer_id,
                                preview_url=match.preview_url, tempo_bpm=match.bpm)
            counts["verified"] += 1
    return counts


def seed_from_csv(csv_path: str, db: SongDatabase, deezer: Optional[DeezerClient] = None) -> Dict[str, int]:
    """
    Stages every CSV row in pending_songs, then (if deezer is given) checks
    every unchecked pending row on Deezer. Rows that are already pending or
    approved are skipped, so running the seed twice is safe.
    Returns counts: {"added", "skipped"} plus {"verified", "not_found", "errors"} when checked.
    """
    counts = {"added": 0, "skipped": 0}
    for row in load_songs(csv_path):
        song = Song(**row, source="seed_csv")
        try:
            db.add_pending(song, verified=None)
            counts["added"] += 1
        except DuplicateSongError:
            counts["skipped"] += 1

    if deezer is not None:
        counts.update(verify_pending(db, deezer))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed pending_songs from the starter CSV.")
    parser.add_argument("--csv", default=str(PROJECT_ROOT / DEFAULT_CSV_PATH))
    parser.add_argument("--db", default=str(PROJECT_ROOT / DEFAULT_DB_PATH))
    parser.add_argument("--no-verify", action="store_true", help="skip the Deezer check")
    args = parser.parse_args()

    with SongDatabase(args.db) as db:
        deezer = None if args.no_verify else DeezerClient(db=db)
        counts = seed_from_csv(args.csv, db, deezer)
        print(f"Seeded {args.db}: {counts['added']} added to pending_songs, "
              f"{counts['skipped']} skipped as duplicates.")
        if deezer is not None:
            print(f"Deezer check: {counts['verified']} verified, {counts['not_found']} not found, "
                  f"{counts['errors']} could not be checked (retried on the next run).")
        print(f"songs: {len(db.get_songs())} rows, pending_songs: {len(db.list_pending())} rows")


if __name__ == "__main__":
    main()
