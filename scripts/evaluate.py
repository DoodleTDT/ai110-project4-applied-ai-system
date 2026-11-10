"""
Runs the hand-labeled test prompts through the full agent and writes a report.

By default the agent runs on a copy of the database (eval/eval_run.db3), so
songs staged during the evaluation don't end up in your real pending_songs.

This makes real Claude, Last.fm and Deezer calls. On claude-haiku-5-5 the
eight prompts cost about one cent in total.

Usage (from the project root):
    python scripts/evaluate.py [--k 10] [--db data/musicmatchmaker.db3] [--in-place]

Outputs:
    eval/results.json   every number, per prompt
    eval/report.md      summary tables
"""

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import DEFAULT_DB_PATH, SongDatabase  # noqa: E402
from src.evaluation import (PromptResult, baseline_playlist, cost_usd, genre_diversity,  # noqa: E402
                            label_profile, load_labels, mean, parse_checks, precision_at_k,
                            song_fits, traceability)
from src.llm import MODEL, LLMError  # noqa: E402
from src.pipeline import PlaylistAgent  # noqa: E402
from src.recommender import Song, load_songs, score_song  # noqa: E402


def pool_genres(db: SongDatabase):
    """Genres of every song the agent can recommend: approved + verified pending."""
    return [s.genre for s in db.get_songs()] + [p.song.genre for p in db.list_pending() if p.verified]


def label_prefs(label):
    p = label_profile(label)
    return {"favorite_genres": p.favorite_genres, "favorite_mood": p.favorite_mood,
            "target_energy": p.target_energy, "likes_acoustic": p.likes_acoustic}


def run_prompt(agent: PlaylistAgent, label, csv_songs, k: int) -> PromptResult:
    baseline = baseline_playlist(csv_songs, label, k)
    baseline_songs = [Song(**song) for song, _, _ in baseline]
    try:
        result = agent.run(label.prompt, k=k)
    except (LLMError, ValueError) as error:
        return PromptResult(
            id=label.id, prompt=label.prompt, parse={}, profile={}, used_external=False, staged=0,
            checked_on_deezer=0, not_found_on_deezer=0, precision=None, traceability=None,
            mean_score=None, baseline_precision=precision_at_k(baseline_songs, label),
            baseline_mean_score=mean(s for _, s, _ in baseline), tokens={}, cost_usd=None,
            error=str(error))

    songs = [e.song for e in result.entries]
    prefs = label_prefs(label)
    # Score both playlists against the same hand-labeled profile, so they compare fairly
    rescored = [score_song(prefs, asdict(s))[0] for s in songs]
    reasons = [r["reasons"][0] for r in result.rejected if r["reasons"]]
    could_not_check = sum(r.startswith("could not check") for r in reasons)
    tokens = result.tokens
    return PromptResult(
        id=label.id, prompt=label.prompt,
        parse=parse_checks(result.profile, label), profile=asdict(result.profile),
        used_external=result.used_external, staged=len(result.staged),
        checked_on_deezer=len(result.staged) + len(result.rejected) - could_not_check,
        not_found_on_deezer=sum(r == "not found on Deezer" for r in reasons),
        precision=precision_at_k(songs, label), traceability=traceability(songs),
        mean_score=mean(rescored),
        baseline_precision=precision_at_k(baseline_songs, label),
        baseline_mean_score=mean(score_song(prefs, song)[0] for song, _, _ in baseline),
        tokens=tokens, cost_usd=cost_usd(MODEL, tokens.get("input_tokens", 0), tokens.get("output_tokens", 0)),
        playlist=[{"title": s.title, "artist": s.artist, "genre": s.genre, "energy": s.energy,
                   "fits_label": song_fits(s, label), "status": "pending" if e.pending_review else "approved"}
                  for s, e in zip(songs, result.entries)],
    )


def fmt(value, pct=False, digits=2):
    if value is None:
        return "-"
    return f"{value:.0%}" if pct else f"{value:.{digits}f}"


def write_report(path: Path, summary: dict, results) -> None:
    lines = [
        "# MusicMatchMaker evaluation",
        "",
        f"Run: {summary['date']} · model `{summary['model']}` · k = {summary['k']} · "
        f"{len(results)} hand-labeled prompts ([eval/prompts.json](prompts.json))",
        "",
        "## Summary",
        "",
        "| Metric | RAG system | Original scorer (CSV) |",
        "|---|---|---|",
        f"| Precision@k (genre related and energy in range) | {fmt(summary['precision'], True)} | "
        f"{fmt(summary['baseline_precision'], True)} |",
        f"| Mean score against the labeled profile (out of 9) | {fmt(summary['mean_score'])} | "
        f"{fmt(summary['baseline_mean_score'])} |",
        f"| Playlist songs confirmed on Deezer | {fmt(summary['traceability'], True)} | "
        f"{fmt(summary['seed_traceability'], True)} |",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Parse accuracy: genre | {fmt(summary['parse']['genre'], True)} |",
        f"| Parse accuracy: mood | {fmt(summary['parse']['mood'], True)} |",
        f"| Parse accuracy: energy in labeled range | {fmt(summary['parse']['energy'], True)} |",
        f"| Parse accuracy: acoustic (prompts that state it) | {fmt(summary['parse']['acoustic'], True)} |",
        f"| External candidates Deezer could not find | {summary['not_found']} of {summary['checked']} "
        f"({fmt(summary['unverifiable_rate'], True)}) |",
        f"| Starter CSV rows Deezer could not find | {summary['seed_not_found']} of {summary['seed_rows']} "
        f"({fmt(summary['seed_unverifiable_rate'], True)}) |",
        f"| Songs staged for review | {summary['staged']} |",
        f"| Tokens per prompt (mean) | {fmt(summary['mean_tokens'], digits=0)} |",
        f"| Total cost | ${summary['total_cost']:.4f} |" if summary["total_cost"] is not None else "| Total cost | - |",
        "",
        "## Genre diversity of the song pool (approved + verified pending)",
        "",
        "| | Songs | Genres | Largest genre | Largest share | Entropy (0-1) |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("before", "after"):
        d = summary["diversity"][name]
        lines.append(f"| {name.capitalize()} | {d['songs']} | {d['genres']} | {d['top_genre'] or '-'} | "
                     f"{fmt(d['top_share'], True)} | {fmt(d['entropy'])} |")
    lines += [
        "",
        "## Per prompt",
        "",
        "| Prompt | Parsed as | Parse (g/m/e/a) | Searched outside | Staged | Precision | Baseline precision | Tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    mark = {True: "✓", False: "✗", None: "-"}
    for r in results:
        if r.error:
            lines.append(f"| {r.prompt} | error: {r.error} | | | | | {fmt(r.baseline_precision, True)} | |")
            continue
        p = r.profile
        parsed = f"{', '.join(p['favorite_genres'])} · {p['favorite_mood']} · {p['target_energy']:.2f}"
        checks = "/".join(mark[r.parse[key]] for key in ("genre", "mood", "energy", "acoustic"))
        lines.append(f"| {r.prompt} | {parsed} | {checks} | {'yes' if r.used_external else 'no'} | {r.staged} | "
                     f"{fmt(r.precision, True)} | {fmt(r.baseline_precision, True)} | {r.tokens.get('used', 0):,} |")
    lines += [
        "",
        "## Notes",
        "",
        "- Precision counts a song as fitting when its genre is a labeled genre (or a parent or subgenre "
        "of one) and its energy is within 0.1 of the labeled range. For Last.fm songs, genre and energy are "
        "estimated by Claude, so this measures agreement with those estimates, not a listening test.",
        "- Both playlists are re-scored against the same hand-labeled profile, so the mean scores compare "
        "fairly even when the parser read the request differently.",
        "- The original scorer only has the 50-song CSV, and Deezer could find none of its songs, so its "
        "playlists are 0% traceable to a real song.",
        "- Prompts run one after another on the same database, so songs staged by earlier prompts are "
        "candidates for later ones.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MusicMatchMaker on hand-labeled prompts.")
    parser.add_argument("--prompts", default=str(PROJECT_ROOT / "eval" / "prompts.json"))
    parser.add_argument("--db", default=str(PROJECT_ROOT / DEFAULT_DB_PATH))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "eval"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--in-place", action="store_true", help="run on --db itself instead of a copy")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db)
    if not args.in_place:
        copy = out / "eval_run.db3"
        shutil.copyfile(db_path, copy)
        db_path = copy

    labels = load_labels(args.prompts)
    csv_songs = load_songs(str(PROJECT_ROOT / "data" / "songs.csv"))
    results = []
    with SongDatabase(str(db_path)) as db:
        seed = [p for p in db.list_pending() if p.song.source == "seed_csv"]
        seed_not_found = sum(p.verified is False for p in seed)
        before = genre_diversity(pool_genres(db))
        agent = PlaylistAgent.create(db)
        for label in labels:
            start = time.time()
            result = run_prompt(agent, label, csv_songs, args.k)
            results.append(result)
            status = f"error: {result.error}" if result.error else (
                f"precision {fmt(result.precision, True)} (baseline {fmt(result.baseline_precision, True)}), "
                f"staged {result.staged}, {result.tokens.get('used', 0):,} tokens")
            print(f"[{label.id}] {time.time() - start:.0f}s - {status}")
        after = genre_diversity(pool_genres(db))

    ok = [r for r in results if not r.error]
    checked = sum(r.checked_on_deezer for r in ok)
    not_found = sum(r.not_found_on_deezer for r in ok)
    costs = [r.cost_usd for r in ok if r.cost_usd is not None]
    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": MODEL, "k": args.k,
        "database": str(db_path), "prompts": len(results), "errors": len(results) - len(ok),
        "precision": mean(r.precision for r in ok),
        "baseline_precision": mean(r.baseline_precision for r in results),
        "mean_score": mean(r.mean_score for r in ok),
        "baseline_mean_score": mean(r.baseline_mean_score for r in results),
        "traceability": mean(r.traceability for r in ok),
        "seed_traceability": 0.0,
        "parse": {key: mean(None if r.parse[key] is None else float(r.parse[key]) for r in ok)
                  for key in ("genre", "mood", "energy", "acoustic")},
        "checked": checked, "not_found": not_found,
        "unverifiable_rate": not_found / checked if checked else None,
        "seed_rows": len(seed), "seed_not_found": seed_not_found,
        "seed_unverifiable_rate": seed_not_found / len(seed) if seed else None,
        "staged": sum(r.staged for r in ok),
        "mean_tokens": mean(r.tokens.get("used") for r in ok),
        "total_cost": sum(costs) if costs else None,
        "diversity": {"before": before, "after": after},
    }

    (out / "results.json").write_text(
        json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2, default=str),
        encoding="utf-8")
    write_report(out / "report.md", summary, results)
    print(f"\nWrote {out / 'results.json'} and {out / 'report.md'}")
    cost = f"${summary['total_cost']:.4f}" if summary["total_cost"] is not None else "-"
    print(f"Precision@{args.k}: {fmt(summary['precision'], True)} vs baseline "
          f"{fmt(summary['baseline_precision'], True)} · total cost {cost}")


if __name__ == "__main__":
    main()
