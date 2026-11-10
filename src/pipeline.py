"""
Playlist agent: request text -> ranked playlist with a reason per song.

    1. parse        RequestParser turns the request into a UserProfile
    2. local        rank approved songs + verified pending songs with recommend_songs
    3. coverage     CoverageChecker decides whether to search outside
    4. external     Last.fm candidates -> dedupe -> Deezer verify -> tags ->
                    VibeEstimator -> SongValidator.review -> pending_songs
    5. re-rank      everything again with the same scorer
    6. write        PlaylistWriter gives one reason per song (falls back to the
                    score_song reasons if Claude is unavailable or over budget)

The order is fixed in code rather than chosen by Claude, so every guardrail
always runs: Claude only labels and describes songs that came from the
database or the APIs, and never picks songs itself.

Token guardrails in this module:
    * MAX_EXTERNAL_CANDIDATES caps how many external songs one request handles.
    * Duplicate and Deezer checks run before VibeEstimator, so unknown or
      already-known songs never cost Claude tokens.
    * One TokenBudget is shared by all Claude calls in a request.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.database import DuplicateSongError, SongDatabase
    from src.external import DeezerClient, ExternalAPIError, LastFmClient
    from src.genres import TAXONOMY
    from src.llm import (DEFAULT_TOKEN_LIMIT, VIBE_FIELDS, LLMError, PlaylistWriter, RequestParser,
                         TokenBudget, TokenBudgetExceeded, VibeEstimator)
    from src.logging_utils import AgentLogger
    from src.recommender import Song, UserProfile, recommend_songs
    from src.validation import SongValidator
except ModuleNotFoundError:
    from database import DuplicateSongError, SongDatabase
    from external import DeezerClient, ExternalAPIError, LastFmClient
    from genres import TAXONOMY
    from llm import (DEFAULT_TOKEN_LIMIT, VIBE_FIELDS, LLMError, PlaylistWriter, RequestParser,
                     TokenBudget, TokenBudgetExceeded, VibeEstimator)
    from logging_utils import AgentLogger
    from recommender import Song, UserProfile, recommend_songs
    from validation import SongValidator

MAX_EXTERNAL_CANDIDATES = 15
LASTFM_TRACKS_PER_TAG = 30


@dataclass
class PlaylistEntry:
    song: Song
    score: float
    reason: str
    source: str             # "songs" (approved) or "pending" (waiting for review)
    pending_review: bool
    score_reasons: str = ""


@dataclass
class PlaylistResult:
    request: str
    profile: UserProfile
    entries: List[PlaylistEntry]
    used_external: bool
    staged: List[Song] = field(default_factory=list)       # new songs added to pending_songs
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    tokens: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


@dataclass
class _Candidate:
    """A song the agent can rank, plus where it came from."""
    song: Song
    pending: bool


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

class CoverageChecker:
    """
    Decides when local songs are not enough:
      * fewer than k songs score at least `threshold` (out of 9), or
      * a requested genre has fewer than `min_per_genre` songs
        (a genre's subgenres count, e.g. metal counts toward rock).
    """

    def __init__(self, threshold: float = 6.0, min_per_genre: int = 5, taxonomy=TAXONOMY):
        self.threshold = threshold
        self.min_per_genre = min_per_genre
        self.taxonomy = taxonomy

    def genre_coverage(self, profile: UserProfile, genre_counts: Dict[str, int]) -> Dict[str, int]:
        """Songs available per requested genre, counting subgenres."""
        coverage = {}
        for genre in profile.favorite_genres:
            coverage[genre] = sum(n for song_genre, n in genre_counts.items()
                                  if self.taxonomy.matches([genre], song_genre) == genre)
        return coverage

    def needs_external(self, ranked: List[Tuple[Dict, float, str]], profile: UserProfile, k: int,
                       genre_counts: Dict[str, int]) -> bool:
        good = sum(1 for _, score, _ in ranked if score >= self.threshold)
        if good < k:
            return True
        return any(n < self.min_per_genre for n in self.genre_coverage(profile, genre_counts).values())

    def target_genres(self, profile: UserProfile, genre_counts: Dict[str, int]) -> List[str]:
        """Requested genres, fewest songs first, so expansion fills gaps instead of adding more pop."""
        coverage = self.genre_coverage(profile, genre_counts)
        order = {genre: i for i, genre in enumerate(profile.favorite_genres)}
        return sorted(profile.favorite_genres, key=lambda g: (coverage[g], order[g]))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def _stub_song(title: str, artist: str, mbid: Optional[str] = None, deezer_id: Optional[int] = None) -> Song:
    """A placeholder Song used only for duplicate checks before any data is estimated."""
    return Song(id=0, title=title, artist=artist, genre="", mood="", energy=0.0, tempo_bpm=None,
                valence=0.0, acousticness=0.0, mbid=mbid, deezer_id=deezer_id)


class PlaylistAgent:
    def __init__(self, db: SongDatabase, parser: RequestParser, lastfm, deezer,
                 estimator: VibeEstimator, validator: SongValidator, writer: PlaylistWriter,
                 logger: Optional[AgentLogger] = None, coverage: Optional[CoverageChecker] = None,
                 max_external_candidates: int = MAX_EXTERNAL_CANDIDATES,
                 token_limit: int = DEFAULT_TOKEN_LIMIT):
        self.db = db
        self.parser = parser
        self.lastfm = lastfm
        self.deezer = deezer
        self.estimator = estimator
        self.validator = validator
        self.writer = writer
        self.logger = logger or AgentLogger(log_path=None)
        self.coverage = coverage or CoverageChecker()
        self.max_external_candidates = max_external_candidates
        self.token_limit = token_limit

    @classmethod
    def create(cls, db: SongDatabase, **kwargs) -> "PlaylistAgent":
        """Builds an agent with the real Claude, Last.fm and Deezer clients."""
        deezer = DeezerClient(db=db)
        return cls(db=db, parser=RequestParser(), lastfm=LastFmClient(db=db), deezer=deezer,
                   estimator=VibeEstimator(db=db), validator=SongValidator(deezer=deezer),
                   writer=PlaylistWriter(), logger=AgentLogger(), **kwargs)

    # --- tools ---

    def search_local_db(self, profile: UserProfile) -> List[_Candidate]:
        """Approved songs plus pending songs that passed the Deezer check."""
        candidates = [_Candidate(song, pending=False) for song in self.db.get_songs()]
        candidates += [_Candidate(p.song, pending=True) for p in self.db.list_pending() if p.verified]
        self.logger.log_step("search_local_db", {
            "approved": sum(not c.pending for c in candidates),
            "verified_pending": sum(c.pending for c in candidates),
        })
        return candidates

    def search_external(self, profile: UserProfile, known: List[Song],
                        genre_counts: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Collects up to max_external_candidates new songs from Last.fm, querying the
        requested genres with the fewest songs first, then the mood. Songs that are
        already known are skipped here, before any Deezer or Claude call.
        """
        tags = self.coverage.target_genres(profile, genre_counts) + [profile.favorite_mood]
        seen = list(known)
        candidates: List[Dict[str, Any]] = []
        for tag in tags:
            if len(candidates) >= self.max_external_candidates:
                break
            try:
                tracks = self.lastfm.tag_top_tracks(tag, limit=LASTFM_TRACKS_PER_TAG)
            except ExternalAPIError as error:
                self.logger.log_step("search_external_error", {"tag": tag, "error": str(error)})
                continue
            for track in tracks:
                if len(candidates) >= self.max_external_candidates:
                    break
                if not track.get("title") or not track.get("artist"):
                    continue
                stub = _stub_song(track["title"], track["artist"], mbid=track.get("mbid"))
                if self.validator.is_duplicate(stub, seen):
                    continue
                seen.append(stub)
                candidates.append({**track, "query_tag": tag})
        self.logger.log_step("search_external", {
            "tags": tags, "candidates": [f'{c["title"]} - {c["artist"]}' for c in candidates],
        })
        return candidates

    def propose_song_addition(self, candidate: Dict[str, Any], match, tags: List[str],
                              vibe: Dict[str, Any], existing: List[Song]) -> Tuple[Optional[Song], List[str]]:
        """
        Builds the full Song, runs every guardrail and stages it in pending_songs.
        Returns (song, []) when staged, or (None, reasons) when rejected.
        """
        song = Song(
            id=0, title=match.title, artist=match.artist, genre=vibe["genre"], mood=vibe["mood"],
            energy=vibe["energy"], tempo_bpm=match.bpm, valence=vibe["valence"],
            acousticness=vibe["acousticness"], source="lastfm", mbid=candidate.get("mbid"),
            deezer_id=match.deezer_id, preview_url=match.preview_url,
            estimated_fields=list(VIBE_FIELDS),
        )
        result = self.validator.review(song, existing, match, tags)
        if not result.accepted:
            return None, result.reasons
        song.confidence = result.confidence
        try:
            song.id = self.db.add_pending(song, verified=True)
        except (DuplicateSongError, ValueError) as error:
            return None, [str(error)]
        self.logger.log_step("propose_song_addition", {
            "title": song.title, "artist": song.artist, "genre": song.genre,
            "confidence": song.confidence, "breakdown": result.breakdown,
        })
        return song, []

    # --- helpers ---

    @staticmethod
    def _rank(profile: UserProfile, candidates: List[_Candidate], k: int) -> List[Tuple[_Candidate, float, str]]:
        """Ranks candidates with recommend_songs (ids are re-numbered: two tables share the list)."""
        dicts = []
        for i, candidate in enumerate(candidates):
            song_dict = asdict(candidate.song)
            song_dict["id"] = i
            dicts.append(song_dict)
        ranked = recommend_songs(asdict(profile), dicts, k=k)
        return [(candidates[song_dict["id"]], score, reasons) for song_dict, score, reasons in ranked]

    @staticmethod
    def _genre_counts(candidates: List[_Candidate]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for candidate in candidates:
            genre = TAXONOMY.canonical(candidate.song.genre)
            counts[genre] = counts.get(genre, 0) + 1
        return counts

    def _expand(self, profile: UserProfile, candidates: List[_Candidate],
                rejected: List[Dict[str, Any]]) -> List[Song]:
        """Step 4: external search through every guardrail. Returns the newly staged songs."""
        known = [c.song for c in candidates]
        external = self.search_external(profile, known, self._genre_counts(candidates))

        # Deezer check (and tags) before Claude: songs that fail never reach VibeEstimator
        verified = []
        for candidate in external:
            label = f'{candidate["title"]} - {candidate["artist"]}'
            try:
                match = self.validator.verify(candidate)
            except ExternalAPIError as error:
                rejected.append({"song": label, "reasons": [f"could not check on Deezer: {error}"]})
                continue
            if match is None:
                rejected.append({"song": label, "reasons": ["not found on Deezer"]})
                continue
            stub = _stub_song(match.title, match.artist, mbid=candidate.get("mbid"), deezer_id=match.deezer_id)
            if self.validator.is_duplicate(stub, known + [v[0]["stub"] for v in verified]):
                rejected.append({"song": label, "reasons": ["duplicate of a song already in the database"]})
                continue
            try:
                tags = self.lastfm.top_tags(candidate["title"], candidate["artist"])
            except ExternalAPIError:
                tags = []
            verified.append(({**candidate, "stub": stub}, match, tags))

        if not verified:
            return []
        try:
            vibes = self.estimator.estimate_many(
                [{"title": m.title, "artist": m.artist, "tags": tags} for _, m, tags in verified])
        except LLMError as error:
            # Includes TokenBudgetExceeded: verified songs are skipped, nothing is guessed
            self.logger.log_step("estimate_vibe_error", {"error": str(error)})
            rejected.extend({"song": f"{m.title} - {m.artist}", "reasons": [f"no vibe estimate: {error}"]}
                            for _, m, _ in verified)
            return []

        staged: List[Song] = []
        for (candidate, match, tags), vibe in zip(verified, vibes):
            label = f"{match.title} - {match.artist}"
            if vibe is None:
                rejected.append({"song": label, "reasons": ["no vibe estimate returned"]})
                continue
            song, reasons = self.propose_song_addition(candidate, match, tags, vibe, known + staged)
            if song is None:
                rejected.append({"song": label, "reasons": reasons})
            else:
                staged.append(song)
        return staged

    def _reasons(self, request: str, ranked: List[Tuple[_Candidate, float, str]]) -> List[str]:
        """Step 6: Claude-written reasons, or the score_song reasons if Claude is unavailable."""
        fallback = [f"Score {score:.2f}/9.00: {reasons}" for _, score, reasons in ranked]
        items = [{
            "title": c.song.title, "artist": c.song.artist, "genre": c.song.genre, "mood": c.song.mood,
            "energy": c.song.energy, "source": "pending review" if c.pending else "approved",
            "score": round(score, 2), "score_reasons": reasons,
        } for c, score, reasons in ranked]
        try:
            written = self.writer.write(request, items)
        except LLMError as error:
            self.logger.log_step("write_playlist_fallback", {"error": str(error)})
            return fallback
        return [text or fallback[i] for i, text in enumerate(written)]

    # --- main entry point ---

    def run(self, request: str, k: int = 10) -> PlaylistResult:
        run_id = self.logger.start_run()
        budget = TokenBudget(limit=self.token_limit)
        for module in (self.parser, self.estimator, self.writer):
            module.budget = budget

        self.logger.log_step("request", {"text": request, "k": k})
        profile = self.parser.parse(request)
        self.logger.log_step("parse_request", {"profile": asdict(profile)})

        candidates = self.search_local_db(profile)
        ranked = self._rank(profile, candidates, k=len(candidates))
        genre_counts = self._genre_counts(candidates)
        used_external = self.coverage.needs_external(ranked, profile, k, genre_counts)
        self.logger.log_step("coverage_check", {
            "needs_external": used_external,
            "good_matches": sum(1 for _, s, _ in ranked if s >= self.coverage.threshold),
            "genre_coverage": self.coverage.genre_coverage(profile, genre_counts),
        })

        staged: List[Song] = []
        rejected: List[Dict[str, Any]] = []
        if used_external:
            staged = self._expand(profile, candidates, rejected)
            candidates = candidates + [_Candidate(song, pending=True) for song in staged]
            self.logger.log_step("external_results", {
                "staged": [f"{s.title} - {s.artist}" for s in staged], "rejected": rejected,
            })

        top = self._rank(profile, candidates, k=k)
        reasons = self._reasons(request, top)
        entries = [
            PlaylistEntry(song=c.song, score=score, reason=reason,
                          source="pending" if c.pending else "songs", pending_review=c.pending,
                          score_reasons=score_reasons)
            for (c, score, score_reasons), reason in zip(top, reasons)
        ]
        self.logger.log_step("playlist", {
            "songs": [{"title": e.song.title, "artist": e.song.artist, "score": round(e.score, 2),
                       "source": e.source} for e in entries],
            "tokens": budget.summary(),
        })
        return PlaylistResult(request=request, profile=profile, entries=entries,
                              used_external=used_external, staged=staged, rejected=rejected,
                              tokens=budget.summary(), run_id=run_id)
