# MusicMatchMaker evaluation

Run: 2026-11-10 16:38 · model `claude-haiku-5-5` · k = 10 · 8 hand-labeled prompts ([eval/prompts.json](prompts.json))

## Summary

| Metric | RAG system | Original scorer (CSV) |
|---|---|---|
| Precision@k (genre related and energy in range) | 71% | 45% |
| Mean score against the labeled profile (out of 9) | 6.09 | 5.09 |
| Playlist songs confirmed on Deezer | 100% | 0% |

| Metric | Value |
|---|---|
| Parse accuracy: genre | 100% |
| Parse accuracy: mood | 100% |
| Parse accuracy: energy in labeled range | 100% |
| Parse accuracy: acoustic (prompts that state it) | 100% |
| External candidates Deezer could not find | 10 of 120 (8%) |
| Starter CSV rows Deezer could not find | 50 of 50 (100%) |
| Songs staged for review | 110 |
| Tokens per prompt (mean) | 6709 |
| Total cost | $0.0094 |

## Genre diversity of the song pool (approved + verified pending)

| | Songs | Genres | Largest genre | Largest share | Entropy (0-1) |
|---|---|---|---|---|---|
| Before | 13 | 5 | pop | 31% | 0.91 |
| After | 123 | 29 | pop | 15% | 0.89 |

## Per prompt

| Prompt | Parsed as | Parse (g/m/e/a) | Searched outside | Staged | Precision | Baseline precision | Tokens |
|---|---|---|---|---|---|---|---|
| sad low-energy rock | rock · sad · 0.25 | ✓/✓/✓/- | yes | 15 | 70% | 0% | 6,891 |
| Rainy Sunday study playlist, calm and a little jazzy | jazz, smooth jazz · focused · 0.30 | ✓/✓/✓/- | yes | 14 | 80% | 30% | 6,466 |
| upbeat pop for a summer road trip | pop, dance pop · energetic · 0.85 | ✓/✓/✓/- | yes | 14 | 100% | 70% | 6,837 |
| angry heavy metal for the gym | heavy metal · angry · 0.95 | ✓/✓/✓/✓ | yes | 15 | 70% | 80% | 6,824 |
| acoustic folk songs for a campfire | folk, indie folk · relaxed · 0.30 | ✓/✓/✓/✓ | yes | 12 | 70% | 50% | 6,816 |
| late night lofi beats to code to | lofi, electronic · focused · 0.35 | ✓/✓/✓/- | yes | 13 | 40% | 80% | 6,303 |
| romantic r&b slow jams | r&b, soul · romantic · 0.30 | ✓/✓/✓/- | yes | 15 | 100% | 20% | 6,839 |
| high energy electronic dance music for a party | dance pop, electronic, house · energetic · 0.90 | ✓/✓/✓/✓ | yes | 12 | 40% | 30% | 6,695 |

## Notes

- Precision counts a song as fitting when its genre is a labeled genre (or a parent or subgenre of one) and its energy is within 0.1 of the labeled range. For Last.fm songs, genre and energy are estimated by Claude, so this measures agreement with those estimates, not a listening test.
- Both playlists are re-scored against the same hand-labeled profile, so the mean scores compare fairly even when the parser read the request differently.
- The original scorer only has the 50-song CSV, and Deezer could find none of its songs, so its playlists are 0% traceable to a real song.
- Prompts run one after another on the same database, so songs staged by earlier prompts are candidates for later ones.
