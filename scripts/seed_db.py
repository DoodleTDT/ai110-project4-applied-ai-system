"""
One-time seed: copies the starter CSV into pending_songs (not songs).

Every row is staged with source='seed_csv' and goes through the normal
approve/reject review. data/songs.csv is never modified, because
tests/test_recommender.py loads it directly.

The Deezer existence check (verified + deezer_id) is added in step 3;
until then seeded rows have verified = None (not checked).

Usage (from the project root):
    python scripts/seed_db.py [--csv data/songs.csv] [--db data/musicmatchmaker.db3]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import DEFAULT_DB_PATH, DuplicateSongError, SongDatabase  # noqa: E402
from src.recommender import Song, load_songs  # noqa: E402

DEFAULT_CSV_PATH = "data/songs.csv"


def seed_from_csv(csv_path: str, db: SongDatabase) -> Dict[str, int]:
    """
    Stages every CSV row in pending_songs. Rows that are already pending or
    approved are skipped, so running the seed twice is safe.
    Returns counts: {"added": n, "skipped": n}.
    """
    counts = {"added": 0, "skipped": 0}
    for row in load_songs(csv_path):
        song = Song(**row, source="seed_csv")
        try:
            db.add_pending(song, verified=None)
            counts["added"] += 1
        except DuplicateSongError:
            counts["skipped"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed pending_songs from the starter CSV.")
    parser.add_argument("--csv", default=str(PROJECT_ROOT / DEFAULT_CSV_PATH))
    parser.add_argument("--db", default=str(PROJECT_ROOT / DEFAULT_DB_PATH))
    args = parser.parse_args()

    with SongDatabase(args.db) as db:
        counts = seed_from_csv(args.csv, db)
        print(f"Seeded {args.db}: {counts['added']} added to pending_songs, "
              f"{counts['skipped']} skipped as duplicates.")
        print(f"songs: {len(db.get_songs())} rows, pending_songs: {len(db.list_pending())} rows")


if __name__ == "__main__":
    main()
