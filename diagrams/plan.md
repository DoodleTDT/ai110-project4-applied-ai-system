# Refining the RAG idea for MusicMatchMaker

## Context
Right now [recommender.py](ai110-project4-applied-ai-system/src/recommender.py) scores a fixed 50-song CSV against a rigid `UserProfile` (up to 3 genres, mood, energy, acoustic). The goal: a user describes a playlist in natural language, an AI retrieves the best matches from the local DB **and** outside sources, and new songs get added to the DB so it grows over time. The suggestions below tighten that idea so it stays accurate and explainable, and so it fixes the gaps listed in the model card (too few songs, too much pop, exact-word genre matching).

## Refinements

### 1. Use the LLM to translate the request, and let retrieval pick the songs
- Step 1: the LLM turns "rainy Sunday study playlist, a little jazzy" into structured preferences (genres, mood, energy range, acoustic, tempo/valence).
- Reuse `UserProfile` and `score_song()` as the ranker. That keeps the scoring explainable (`explain_recommendation` still works) and keeps the existing tests useful.
- The LLM should never get to "recommend" a song from memory alone. Every song it returns has to come from a retrieval step (local DB or an external API result). That is the main defense against made-up songs and artists.

### 2. A pipeline with a clear "search outside" trigger
1. Parse request → profile.
2. Retrieve local candidates (existing scorer, plus optional semantic search over a short text description per song).
3. **Coverage check:** if fewer than `k` songs score above a threshold (e.g. 6/9), or a requested genre has fewer than N songs, go to external search. Otherwise skip it. This saves API calls and gives the system a rule you can explain.
4. External search → normalize to the CSV schema → validate.
5. Merge, re-rank with the same scorer, and have the LLM write the final playlist with a reason for each song, citing the retrieved data.

### 3. Pick an external source that returns real metadata
- The scorer needs energy, valence, acousticness and tempo. Spotify's audio-features endpoint is closed to new apps (since late 2024), so plan for that.
- Options: Last.fm (tags work as genre/mood), MusicBrainz (verifies the song exists, gives IDs), Deezer (has BPM). Any field you can't get could be estimated by the LLM, but only if it's flagged `estimated=true`.
- Treat the API as the source of truth for "this song exists." Treat the LLM as the source of the "vibe" labels.

### 4. Guardrails before anything is added to the DB
- **Verify:** only add a song if the title and artist match an external API record.
- **Dedupe:** reuse `_normalize` and `_check_duplicates` (title + artist) so near-duplicates like "Song (Remastered)" get caught.
- **Schema validation:** numbers must be in 0–1, the genre must be on a controlled list.
- **Provenance columns:** `source`, `added_at`, `confidence`, `estimated_fields`.
- **Staging:** write new songs to `songs_pending.csv` first and promote them to `songs.csv` after a review. The review can be by a human, or automatic for high-confidence songs. This keeps one bad LLM call from corrupting the dataset.

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
- The UI can use Streamlit, which is already in `requirements.txt`: a text box for the description, a results table with reasons, and a "pending additions" review panel.
- CSV is fine at this size. Move to SQLite only if concurrent writes or >~5k songs become a problem.

## Open decisions for the user
- Which external source (Last.fm / MusicBrainz / Deezer)?
- Auto-add verified songs, or always require approval?
- Keep the hand-tuned 9-point scorer as the ranker, or let the user (or the LLM) adjust the weights per request?

## Verification (once implemented)
- `pytest` with mocked LLM and API covering parse → retrieve → coverage trigger → validate → stage.
- Manual run: an underrepresented-genre prompt (e.g. "sad low-energy rock") should trigger external search, add verified songs to the pending list, and return a playlist where every song can be traced to a source.