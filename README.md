## Original Project
This is an updated version of Project 3: `Music Recommender Simulation`. This was a small music recommendation system that prioritized User preferences against genres/associated genres (4.0), mood (2.0), energy (2.0), and acoustics (1.0) through a weighted scale (out of 9.0 points), ranking the songs within a static database of 50. A playlist, with the size the User asks for, would be generated with the top songs in the database.

## Title and Summary
**MusicMatchMaker**

What your project does and why it matters.

## Architecture Overview
The system diagram ([diagrams/RAG_system_diagram_draft.mmd](diagrams/RAG_system_diagram_draft.mmd)) shows how the parts connect. The user types a playlist request into the **Streamlit app**, and the **PlaylistAgent** ([src/pipeline.py](src/pipeline.py)) turns it into a ranked playlist through a fixed sequence of steps:

1. **Parse.** `RequestParser` (Claude) turns the free-text request into a structured user profile (genres, mood, target energy, acoustic preference). Genre labels are mapped onto a controlled list by `GenreTaxonomy`.
2. **Retrieve locally.** The agent loads candidates from the SQLite database: approved songs plus pending songs that Deezer has confirmed exist.
3. **Rank.** The original Project 3 scorer (`recommend_songs` / `score_song`, out of 9 points) ranks the candidates against the profile.
4. **Check coverage.** `CoverageChecker` decides whether the local songs are good enough. If fewer than *k* songs score 6+ out of 9, or a requested genre has fewer than 5 songs, the agent searches outside.
5. **Expand (only when needed).** `LastFmClient` pulls top tracks for the weakest genres and the mood, skipping songs whose tags clash with the request. Each candidate is then checked on **Deezer** to confirm it's a real recording. Only songs that pass go to `VibeEstimator` (Claude), which estimates energy, valence and acousticness from the Last.fm tags. `SongValidator` then validates the data, removes duplicates and gives each song a confidence score. Songs that pass are saved to `pending_songs`, never straight into the library.
6. **Re-rank and explain.** All candidates are scored again with the same scorer, and `PlaylistWriter` (Claude) writes one reason per song. If Claude is unavailable or over the token budget, the scorer's own reasons are used instead.

Two things in the design keep the AI grounded:
- **Claude never picks songs.** It only parses the request, estimates missing audio features, and writes explanations. Every song in a playlist comes from the database or the music APIs, and the step order is fixed in code, so every guardrail always runs.
- **A human approves new songs.** Externally found songs appear in playlists marked *pending review*, and they join the `songs` table only when someone approves them in the **Pending review** tab.

Supporting pieces: the **SQLite database** holds `songs`, `pending_songs` and an `external_cache` for Last.fm and Deezer responses. The **one-time seed script** loads the original 50-song CSV into `pending_songs`. `AgentLogger` records each step of every run for debugging and for [ai_interactions.md](ai_interactions.md) and the model card.


### Setup Instructions
**Requirements:** Python 3.10+ (developed on 3.13), an internet connection, and two API keys:
- **Anthropic API key**: https://console.anthropic.com/
- **Last.fm API key** (free): https://www.last.fm/api/account/create

Deezer is used for song verification and needs no key.

1. **Clone the repo and move into it**
   ```bash
   git clone <https://github.com/DoodleTDT/ai110-project4-applied-ai-system.git>
   cd ai110-project4-applied-ai-system
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your API keys.** Copy `.env.example` to `.env` (or `key.env`) in the project root and fill it in:
   ```
   LASTFM_API_KEY=your-lastfm-key
   ANTHROPIC_API_KEY=your-anthropic-key
   ```
   Both files are git-ignored, so your keys are never committed.

5. **Seed the database.** This stages the 50 starter songs from `data/songs.csv` into `data/musicmatchmaker.db3` and checks each one on Deezer. Running it again is safe (duplicates are skipped, and failed checks are retried).
   ```bash
   python scripts/seed_db.py
   ```
   Add `--no-verify` to skip the Deezer check (for example, when offline).

6. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Streamlit opens the app in your browser (usually http://localhost:8501). It has three tabs:
   - **Playlist**: describe the playlist you want in plain English and get ranked songs with reasons.
   - **Pending review**: approve or reject songs the agent found (nothing is approved automatically).
   - **Library**: approved songs and genre counts.

   To use a different database file, set `MMM_DB_PATH` before launching (e.g. `$env:MMM_DB_PATH="data/other.db3"` in PowerShell).

**Optional**
- Run the tests: `pytest`
- Run the evaluation on the hand-labeled prompts (makes real Claude, Last.fm and Deezer calls; about one cent total). It runs on a copy of the database and writes `eval/results.json` and `eval/report.md`:
  ```bash
  python scripts/evaluate.py
  ```


## Sample Outputs
Need to provide inputs that can give consistent/similar outputs that others can test

## Design Decisions
Why you built it this way, and what trade-offs you made.

## Testing Summary
What worked, what didn't, and what you learned.

## Reflection
A brief note on what this project taught you about AI and problem-solving.

Read `model_card.md` for more notes



