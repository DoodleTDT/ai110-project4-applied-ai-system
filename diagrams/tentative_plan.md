# Refining the RAG idea for MusicMatchMaker

## Context
Right now [recommender.py](ai110-project4-applied-ai-system/src/recommender.py) scores a fixed 50-song CSV against a rigid `UserProfile` (up to 3 genres, mood, energy, acoustic). The goal: a user describes a playlist in natural language, an AI retrieves the best matches from the local DB **and** outside sources, and new songs get added to the DB so it grows over time. The suggestions below tighten that idea so it stays accurate and explainable, and so it fixes the gaps listed in the model card (too few songs, too much pop, exact-word genre matching).

## Refinements

### 1. Use the LLM to translate the request, and let retrieval pick the songs
- Step 1: the LLM turns "rainy Sunday study playlist, a little jazzy" into structured preferences (genres, mood, energy range, acoustic, tempo/valence).
- Reuse `UserProfile` and `score_song()` as the ranker, with its fixed 9-point weights in v1. That keeps the scoring explainable (`explain_recommendation` still works) and keeps the existing tests useful. 
- The LLM should never get to "recommend" a song from memory alone. Every song it returns has to come from a retrieval step (local DB or an external API result). That is the main defense against made-up songs and artists.

### 2. A pipeline with a clear "search outside" trigger
1. Parse request → profile.
2. Retrieve local candidates (existing scorer, plus optional semantic search over a short text description per song).
3. **Coverage check:** if fewer than `k` songs score above a threshold (e.g. 6/9), or a requested genre has fewer than N songs, go to external search. Otherwise skip it. This saves API calls and gives the system a rule you can explain. The `songs` table will start nearly empty (see section 8), so early requests will almost always search outside. That's expected.
4. External search → the songs table schema → validate.
5. Merge, re-rank with the same scorer, and have the LLM write the final playlist with a reason for each song, citing the retrieved data.

### 3. Pick an external source that returns real metadata
- The scorer needs energy, valence, acousticness and tempo. Spotify's audio-features endpoint is closed to new apps (since late 2024), so plan for that.
- **Decision: Last.fm + Deezer.**
  - **Last.fm** handles external search: `tag.getTopTracks` (e.g. "sad" + "rock"), `track.getSimilar`, and `track.getTopTags`. Its tags supply genre and mood, and the LLM uses them as hints when it estimates the other fields.
  - **Deezer** verifies that each song exists (`/search` on title + artist) and supplies `bpm` (treated as missing when it is 0) plus a preview URL.
  - When Last.fm returns an `mbid` (MusicBrainz ID), store it as an optional ID for deduping. MusicBrainz is not called directly because of its 1 request/second limit.
  - Energy, valence and acousticness are always LLM-estimated and listed in `estimated_fields`.
  - The Last.fm API key goes in `.env`. Deezer's public search needs no key.
- Treat the API as the source of truth for "this song exists." Treat the LLM as the source of the "vibe" labels.

### 4. Guardrails before anything is added to the DB
- **Verify:** only add a song if the title and artist match a Deezer search result. This applies to the 50 starter songs too. The starter CSV appears to be made up, so most of its rows are expected to fail.
- **Dedupe:** reuse `_normalize` and `_check_duplicates` (title + artist) so near-duplicates like "Song (Remastered)" get caught.
- **Schema validation:** numbers must be in 0–1, the genre must be on a controlled list.
- **Provenance columns:** `source`, `added_at`, `confidence`, `estimated_fields`.
- **Staging:** write new songs to a `pending_songs` table in the SQLite database (`.db3`) first, and move them into the main `songs` table after a review. Each promotion runs as a single transaction (insert into `songs`, delete from `pending_songs`), so a song is never half-moved. This keeps one bad LLM call from corrupting the dataset.
  - **v1: manual approval only.** Every verified song waits in pending until it is approved. Nothing is promoted automatically.
  - Each pending song still records its `confidence` score and `estimated_fields`. The review panel shows both to help with approve/reject decisions.
  - **Later (optional):** auto-approve songs above a confidence threshold, once the confidence scores have proven to match real quality.

### 5. Fix the genre problem at the same time
- Add a small genre taxonomy (`metal → rock`, `indie pop → pop`, …) that the LLM maps new genres onto. This replaces the word-subset rule in `_match_genre` and fixes the metal/rock limitation. It also stops new labels like "Alt-Rock" from fragmenting the genre list.

### 6. Make the expansion fix the bias, not make it worse
- Track how many songs each genre has. When external search runs, prefer genres that have few songs, so the DB doesn't fill up with even more pop.

### 7. Evaluation (needed for the "applied AI" part)
- Mock the LLM and the API in pytest so the tests are deterministic (parsing, coverage trigger, dedupe, validation).
- Metrics: hallucination rate (songs suggested that fail verification), precision against hand-labeled test prompts, genre diversity before and after expansion, and a comparison with the current rigid scorer.
- Log every agent step (query, retrieved items, added songs) so you can include it in [ai_interactions.md](ai110-project4-applied-ai-system/ai_interactions.md) and the model card.

### 8. Practical scope
- LLM: Claude API with tool use. Tools: `search_local_db`, `search_external`, `propose_song_addition`. A small, cheap model can handle parsing.
- Cache external results and keep API keys in a `.env` file.
- The UI can use Streamlit, which is already in `requirements.txt`: a text box for the description, a results table with reasons, and a "pending additions" review panel that shows each song's confidence and has Approve / Reject buttons.
- Storage: SQLite from the start (Python's built-in `sqlite3`, no extra dependency), in a single `.db3` file.
  - Tables: `songs`, `pending_songs`, and an optional `external_cache` table for the cached API results.
  - A one-time seed script copies all 50 rows from `data/songs.csv` into `pending_songs` (not `songs`), with `source='seed_csv'`. It then runs the Deezer title + artist check on each row and records the result (`verified` true/false, plus the Deezer track ID when one matches). The rows then go through the normal approve/reject review.
  - `data/songs.csv` stays unchanged, because `tests/test_recommender.py` loads it directly as test data.
  - Put a `UNIQUE` constraint on the normalized title + artist (and on `mbid` when there is one), so the database itself rejects duplicates on top of `_check_duplicates`.
  - Use `CHECK` constraints to keep energy, valence and acousticness between 0 and 1, which backs up the schema validation in section 4.

## Open decisions for the user
- ~~Which external source (Last.fm / MusicBrainz / Deezer)?~~ Decided: Last.fm + Deezer (see section 3).
- ~~Auto-add verified songs, or always require approval?~~ Decided: manual approval for v1. Confidence is still recorded, and auto-add is a possible later addition (see section 4).
- ~~Keep the hand-tuned 9-point scorer as the ranker, or let the user (or the LLM) adjust the weights per request?~~ Decided: v1 keeps the fixed 9-point weights. Per-request weights (set by the user or the LLM) are a possible later addition if the rankings feel too rigid.

## Verification (once implemented)
- `pytest` with mocked LLM and API covering parse → retrieve → coverage trigger → validate → stage.
- After seeding: `songs` is empty and `pending_songs` has 50 rows, each with a recorded verification result.
- Manual run: an underrepresented-genre prompt (e.g. "sad low-energy rock") should trigger external search, add verified songs to the pending list with a confidence score (they reach the main table only after approval), and return a playlist where every song can be traced to a source.
