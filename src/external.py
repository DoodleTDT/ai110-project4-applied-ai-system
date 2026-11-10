"""
External music API clients.

LastFmClient  finds candidate songs and their tags (genre / mood hints).
DeezerClient  confirms a song exists and supplies bpm and a preview URL.

Both clients cache successful responses in SongDatabase.external_cache
(when a database is given) and wait a short time between requests to stay
under the APIs' rate limits. Only public song metadata is sent to the
APIs; requests use a generic User-Agent.
"""

import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

import requests

try:
    from src.config import require_key
    from src.genres import normalize
except ModuleNotFoundError:
    from config import require_key
    from genres import normalize

USER_AGENT = "MusicMatchMaker/0.1 (student project)"
DEFAULT_TIMEOUT = 10.0
DEFAULT_CACHE_SECONDS = 7 * 24 * 3600  # one week

LASTFM_URL = "https://ws.audioscrobbler.com/2.0/"
DEEZER_URL = "https://api.deezer.com"

# Last.fm error codes that mean "nothing found" rather than a failure
_LASTFM_NOT_FOUND = {6}


class ExternalAPIError(RuntimeError):
    """Raised when an external API request fails (network, bad key, rate limit, bad data)."""


@dataclass
class DeezerMatch:
    """A Deezer track that matched a title + artist search."""
    deezer_id: int
    title: str
    artist: str
    bpm: Optional[float]        # Deezer reports 0 when it has no tempo; stored as None
    preview_url: Optional[str]


def comparable(text: str) -> str:
    """
    Loose form of a title or artist for matching: lowercased, without
    bracketed parts ("(Remastered 2011)", "[Live]"), without a " - ..." suffix,
    without a leading "the", and with punctuation removed.
    """
    text = normalize(text)
    text = re.sub(r"[\(\[].*?[\)\]]", " ", text)
    text = text.split(" - ")[0]
    text = re.sub(r"[^\w\s&]", " ", text)
    text = " ".join(text.split())
    if text.startswith("the "):
        text = text[4:]
    return text


class _HttpClient:
    """Shared request, throttle and cache logic."""

    def __init__(self, db=None, session=None, min_interval: float = 0.0,
                 timeout: float = DEFAULT_TIMEOUT, cache_seconds: Optional[float] = DEFAULT_CACHE_SECONDS,
                 sleep: Callable[[float], None] = time.sleep):
        self.db = db
        self.session = session or requests.Session()
        self.min_interval = min_interval
        self.timeout = timeout
        self.cache_seconds = cache_seconds
        self._sleep = sleep
        self._last_request = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._throttle()
        try:
            response = self.session.get(url, params=params, timeout=self.timeout,
                                        headers={"User-Agent": USER_AGENT})
        except requests.RequestException as error:
            raise ExternalAPIError(f"request to {url} failed: {error}") from error
        try:
            data = response.json()
        except ValueError as error:
            raise ExternalAPIError(f"{url} returned non-JSON (HTTP {response.status_code})") from error
        return data, response.status_code

    def _cached(self, key: str, fetch: Callable[[], Any]) -> Any:
        """Returns the cached value for key, or calls fetch() and caches its result."""
        if self.db is not None:
            hit = self.db.cache_get(key, max_age_seconds=self.cache_seconds)
            if hit is not None:
                return hit["value"]
        value = fetch()
        if self.db is not None:
            self.db.cache_put(key, {"value": value})
        return value


# ---------------------------------------------------------------------------
# Last.fm
# ---------------------------------------------------------------------------

def _as_list(value: Any) -> List[Any]:
    """Last.fm returns a single object instead of a one-item list in some responses."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _track_dict(item: Dict[str, Any]) -> Dict[str, Any]:
    artist = item.get("artist") or {}
    artist_name = artist.get("name") if isinstance(artist, dict) else str(artist)
    return {
        "title": str(item.get("name", "")).strip(),
        "artist": str(artist_name or "").strip(),
        "mbid": (item.get("mbid") or "").strip() or None,
    }


class LastFmClient(_HttpClient):
    """Last.fm search: tag.getTopTracks, track.getSimilar, track.getTopTags."""

    def __init__(self, api_key: Optional[str] = None, db=None, session=None,
                 min_interval: float = 0.25, **kwargs):
        super().__init__(db=db, session=session, min_interval=min_interval, **kwargs)
        self.api_key = api_key or require_key("LASTFM_API_KEY")

    def _call(self, method: str, **params) -> Optional[Dict[str, Any]]:
        """Calls a Last.fm method. Returns None for "not found", raises on other errors."""
        query = {"method": method, "api_key": self.api_key, "format": "json", "autocorrect": 1, **params}
        data, status = self._get_json(LASTFM_URL, query)
        if isinstance(data, dict) and "error" in data:
            if data["error"] in _LASTFM_NOT_FOUND:
                return None
            raise ExternalAPIError(f"Last.fm {method} error {data['error']}: {data.get('message', '')}")
        if status >= 400 or not isinstance(data, dict):
            raise ExternalAPIError(f"Last.fm {method} returned HTTP {status}")
        return data

    def tag_top_tracks(self, tag: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Top tracks for a tag (e.g. "rock", "sad"): [{title, artist, mbid}]."""
        tag = normalize(tag)

        def fetch():
            data = self._call("tag.getTopTracks", tag=tag, limit=limit)
            items = _as_list(((data or {}).get("tracks") or {}).get("track"))
            return [_track_dict(item) for item in items]

        return self._cached(f"lastfm:tag.getTopTracks:{tag}:{limit}", fetch)

    def similar_tracks(self, title: str, artist: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Tracks similar to a song: [{title, artist, mbid, match}] (match is 0-1)."""

        def fetch():
            data = self._call("track.getSimilar", track=title, artist=artist, limit=limit)
            items = _as_list(((data or {}).get("similartracks") or {}).get("track"))
            results = []
            for item in items:
                track = _track_dict(item)
                try:
                    track["match"] = float(item.get("match", 0))
                except (TypeError, ValueError):
                    track["match"] = 0.0
                results.append(track)
            return results

        key = f"lastfm:track.getSimilar:{normalize(artist)}|{normalize(title)}:{limit}"
        return self._cached(key, fetch)

    def top_tags(self, title: str, artist: str, limit: int = 10) -> List[str]:
        """The song's most-used tags, lowercased, most popular first."""

        def fetch():
            data = self._call("track.getTopTags", track=title, artist=artist)
            items = _as_list(((data or {}).get("toptags") or {}).get("tag"))
            tags = []
            for item in items:
                name = normalize(item.get("name", ""))
                if name and name not in tags:
                    tags.append(name)
            return tags[:limit]

        key = f"lastfm:track.getTopTags:{normalize(artist)}|{normalize(title)}:{limit}"
        return self._cached(key, fetch)


# ---------------------------------------------------------------------------
# Deezer
# ---------------------------------------------------------------------------

class DeezerClient(_HttpClient):
    """Deezer public API (no key): confirms a song exists and fetches its bpm."""

    def __init__(self, db=None, session=None, min_interval: float = 0.12, **kwargs):
        # Deezer allows 50 requests per 5 seconds
        super().__init__(db=db, session=session, min_interval=min_interval, **kwargs)

    def _call(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data, status = self._get_json(f"{DEEZER_URL}{path}", params)
        if isinstance(data, dict) and "error" in data:
            error = data["error"] or {}
            raise ExternalAPIError(
                f"Deezer {path} error {error.get('code', '?')}: {error.get('message', '')}"
            )
        if status >= 400 or not isinstance(data, dict):
            raise ExternalAPIError(f"Deezer {path} returned HTTP {status}")
        return data

    def _find(self, title: str, artist: str) -> Optional[Dict[str, Any]]:
        """Returns the first search result whose title and artist match, or None."""
        want_title, want_artist = comparable(title), comparable(artist)
        clean = lambda s: s.replace('"', " ").strip()
        queries = [f'artist:"{clean(artist)}" track:"{clean(title)}"', f"{clean(artist)} {clean(title)}"]
        for query in queries:
            results = self._call("/search", {"q": query, "limit": 10}).get("data") or []
            for item in results:
                item_artist = (item.get("artist") or {}).get("name", "")
                if comparable(item.get("title", "")) == want_title and comparable(item_artist) == want_artist:
                    return item
        return None

    def search(self, title: str, artist: str) -> Optional[DeezerMatch]:
        """
        Looks up a song by title + artist. Returns a DeezerMatch only when both
        match (ignoring case, punctuation and suffixes like "(Remastered)"),
        otherwise None. "Not found" results are cached too.
        """

        def fetch():
            item = self._find(title, artist)
            if item is None:
                return None
            track = self._call(f"/track/{int(item['id'])}")
            bpm = track.get("bpm")
            try:
                bpm = float(bpm) if bpm else None
            except (TypeError, ValueError):
                bpm = None
            match = DeezerMatch(
                deezer_id=int(item["id"]),
                title=str(item.get("title", "")),
                artist=str((item.get("artist") or {}).get("name", "")),
                bpm=bpm if bpm and bpm > 0 else None,
                preview_url=item.get("preview") or None,
            )
            return asdict(match)

        value = self._cached(f"deezer:search:{normalize(artist)}|{normalize(title)}", fetch)
        return DeezerMatch(**value) if value else None
