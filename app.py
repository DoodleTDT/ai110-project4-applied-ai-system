"""
MusicMatchMaker - Streamlit app.

    Playlist        describe a playlist; the agent returns ranked songs with reasons
    Pending review  approve or reject staged songs (nothing is approved automatically)
    Library         approved songs and genre counts

Run from the project root:
    streamlit run app.py

The database path can be changed with the MMM_DB_PATH environment variable
(default: data/musicmatchmaker.db3).
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.config import MissingAPIKeyError
from src.database import DEFAULT_DB_PATH, DuplicateSongError, PendingSong, SongDatabase
from src.external import DeezerClient, ExternalAPIError
from src.llm import MAX_REQUEST_CHARS, MODEL, LLMError, RequestTooLongError
from src.pipeline import PlaylistAgent, PlaylistResult
from src.recommender import Song

DB_PATH = os.environ.get("MMM_DB_PATH", DEFAULT_DB_PATH)
EXAMPLE_REQUEST = "Rainy Sunday study playlist, calm and a little jazzy"


# ---------------------------------------------------------------------------
# Table helpers (plain functions, so they can be tested without Streamlit)
# ---------------------------------------------------------------------------

def playlist_rows(result: PlaylistResult) -> List[Dict[str, Any]]:
    return [{
        "#": i,
        "Title": e.song.title,
        "Artist": e.song.artist,
        "Genre": e.song.genre,
        "Mood": e.song.mood,
        "Energy": round(e.song.energy, 2),
        "Score /9": round(e.score, 2),
        "Status": "pending review" if e.pending_review else "approved",
        "Why": e.reason,
    } for i, e in enumerate(result.entries, 1)]


def verification_label(verified) -> str:
    return {True: "verified", False: "not found on Deezer", None: "not checked"}[verified]


def pending_rows(pending: List[PendingSong]) -> List[Dict[str, Any]]:
    return [{
        "ID": p.pending_id,
        "Title": p.song.title,
        "Artist": p.song.artist,
        "Genre": p.song.genre,
        "Mood": p.song.mood,
        "Energy": p.song.energy,
        "Valence": p.song.valence,
        "Acoustic": p.song.acousticness,
        "BPM": p.song.tempo_bpm,
        "Source": p.song.source,
        "Deezer": verification_label(p.verified),
        "Confidence": p.song.confidence,
        "Estimated": ", ".join(p.song.estimated_fields) or "-",
    } for p in pending]


def library_rows(songs: List[Song]) -> List[Dict[str, Any]]:
    return [{
        "ID": s.id, "Title": s.title, "Artist": s.artist, "Genre": s.genre, "Mood": s.mood,
        "Energy": s.energy, "Valence": s.valence, "Acoustic": s.acousticness, "BPM": s.tempo_bpm,
        "Source": s.source, "Confidence": s.confidence,
        "Added": s.added_at.strftime("%Y-%m-%d") if s.added_at else "",
    } for s in songs]


def can_approve(pending: PendingSong) -> bool:
    """Only songs Deezer confirmed exist can be approved (guardrail: verify before adding)."""
    return pending.verified is True


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def playlist_tab(k: int) -> None:
    st.subheader("Describe a playlist")
    request = st.text_area(
        "What do you want to listen to?", placeholder=EXAMPLE_REQUEST, max_chars=MAX_REQUEST_CHARS,
        help=f"Up to {MAX_REQUEST_CHARS} characters.",
    )
    if st.button("Build playlist", type="primary", disabled=not request.strip()):
        with st.spinner("Searching the library and, if needed, Last.fm and Deezer..."):
            try:
                with SongDatabase(DB_PATH) as db:
                    st.session_state["result"] = PlaylistAgent.create(db).run(request, k=k)
            except (RequestTooLongError, ValueError) as error:
                st.error(f"Request not accepted: {error}")
            except MissingAPIKeyError as error:
                st.error(str(error))
            except LLMError as error:
                st.error(f"Claude could not process the request: {error}")
            except ExternalAPIError as error:
                st.error(f"A music API failed: {error}")

    result: Optional[PlaylistResult] = st.session_state.get("result")
    if result is None:
        return

    profile = result.profile
    acoustic = {True: "acoustic", False: "non-acoustic", None: "no preference"}[profile.likes_acoustic]
    st.caption(f'Request: "{result.request}"')
    st.write(f"**Understood as:** {', '.join(profile.favorite_genres)} · {profile.favorite_mood} · "
             f"energy {profile.target_energy:.2f} · {acoustic}")

    if not result.entries:
        st.warning("No songs found. Try a different description, or approve songs in Pending review.")
        return
    st.dataframe(pd.DataFrame(playlist_rows(result)), hide_index=True, width="stretch",
                 column_config={"Why": st.column_config.TextColumn(width="large")})

    pending_count = sum(e.pending_review for e in result.entries)
    if pending_count:
        st.info(f"{pending_count} song(s) are still waiting for review. Approve them in the "
                "Pending review tab to add them to the library.")

    tokens = result.tokens
    cols = st.columns(4)
    cols[0].metric("Searched outside", "yes" if result.used_external else "no")
    cols[1].metric("New songs staged", len(result.staged))
    cols[2].metric("Candidates rejected", len(result.rejected))
    cols[3].metric("Tokens used", f"{tokens.get('used', 0):,} / {tokens.get('limit', 0):,}")

    with st.expander("Details: rejected candidates and token use"):
        if result.rejected:
            st.dataframe(pd.DataFrame([{"Song": r["song"], "Reasons": "; ".join(r["reasons"])}
                                       for r in result.rejected]), hide_index=True, width="stretch")
        else:
            st.write("No candidates were rejected.")
        if tokens.get("calls"):
            st.dataframe(pd.DataFrame(tokens["calls"]), hide_index=True)
        st.caption(f"Model: {MODEL} · run id {result.run_id} (see logs/agent_steps.jsonl)")


def review_tab() -> None:
    st.subheader("Pending review")
    st.write("Songs are staged here first. Approving one moves it into the library in a single "
             "transaction. Only songs that Deezer confirmed exist can be approved.")

    with SongDatabase(DB_PATH) as db:
        pending = db.list_pending()
        show = st.radio("Show", ["Verified", "Not found / not checked", "All"], horizontal=True)
        if show == "Verified":
            shown = [p for p in pending if p.verified is True]
        elif show == "Not found / not checked":
            shown = [p for p in pending if p.verified is not True]
        else:
            shown = pending
        st.caption(f"{len(shown)} of {len(pending)} pending song(s)")

        unverified = [p for p in pending if p.verified is False]
        if unverified:
            with st.expander(f"Reject all {len(unverified)} song(s) Deezer could not find"):
                confirm = st.checkbox("Yes, delete these pending songs", key="confirm_bulk_reject")
                if st.button("Reject all not found", disabled=not confirm, key="bulk_reject"):
                    for p in unverified:
                        db.reject(p.pending_id)
                    st.toast(f"Rejected {len(unverified)} song(s)")
                    st.rerun()

        if not shown:
            st.write("Nothing to review here.")
            return

        st.dataframe(pd.DataFrame(pending_rows(shown)), hide_index=True, width="stretch",
                     column_config={"Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0,
                                                                                  format="%.2f")})

        st.markdown("**Review one at a time**")
        for p in shown:
            song = p.song
            cols = st.columns([5, 1, 1, 1])
            confidence = f"{song.confidence:.2f}" if song.confidence is not None else "-"
            cols[0].write(f"**{song.title}** - {song.artist}  \n"
                          f"{song.genre} · {song.mood} · {verification_label(p.verified)} · "
                          f"confidence {confidence}")
            if cols[1].button("Approve", key=f"approve_{p.pending_id}", disabled=not can_approve(p),
                              help=None if can_approve(p) else "Only verified songs can be approved"):
                try:
                    db.approve(p.pending_id)
                    st.toast(f"Approved {song.title}")
                except (DuplicateSongError, KeyError) as error:
                    st.error(str(error))
                st.rerun()
            if cols[2].button("Reject", key=f"reject_{p.pending_id}"):
                try:
                    db.reject(p.pending_id)
                    st.toast(f"Rejected {song.title}")
                except KeyError as error:
                    st.error(str(error))
                st.rerun()
            if song.deezer_id and cols[3].button("Preview", key=f"preview_{p.pending_id}"):
                st.session_state["preview"] = (p.pending_id, song.deezer_id)

            preview = st.session_state.get("preview")
            if preview and preview[0] == p.pending_id:
                try:
                    url = DeezerClient().fresh_preview_url(preview[1])
                    if url:
                        st.audio(url)
                    else:
                        st.caption("Deezer has no preview for this track.")
                except ExternalAPIError as error:
                    st.caption(f"Preview unavailable: {error}")


def library_tab() -> None:
    st.subheader("Library")
    with SongDatabase(DB_PATH) as db:
        songs = db.get_songs()
        counts = db.genre_counts()
    if not songs:
        st.write("No approved songs yet. Build a playlist, then approve songs in Pending review.")
        return
    cols = st.columns(3)
    cols[0].metric("Approved songs", len(songs))
    cols[1].metric("Genres", len(counts))
    top_genre, top_count = max(counts.items(), key=lambda item: item[1])
    cols[2].metric("Largest genre", f"{top_genre} ({top_count / len(songs):.0%})")
    st.bar_chart(pd.Series(counts, name="songs").sort_values(ascending=False))
    st.dataframe(pd.DataFrame(library_rows(songs)), hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="MusicMatchMaker", page_icon="🎧", layout="wide")
    st.title("🎧 MusicMatchMaker")
    with st.sidebar:
        st.header("Settings")
        k = st.slider("Songs in the playlist", min_value=3, max_value=20, value=10)
        st.caption(f"Database: `{DB_PATH}`")
        st.caption(f"Model: `{MODEL}`")
        st.caption("Every song comes from the library or a Last.fm search confirmed on Deezer. "
                   "Claude only labels and describes songs; it never picks them.")

    playlist, review, library = st.tabs(["Playlist", "Pending review", "Library"])
    with playlist:
        playlist_tab(k)
    with review:
        review_tab()
    with library:
        library_tab()


if __name__ == "__main__":
    main()
