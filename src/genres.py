"""
Controlled genre list and parent links (the genre taxonomy).

Every genre label in the system (user requests, the CSV, Last.fm tags, LLM
output) is mapped onto this list with GenreTaxonomy.canonical(). Genre
matching then follows parent links, so a "rock" fan matches "metal" and
"punk", and a "pop" fan matches "indie pop". A specific favorite never
matches a broader genre: an "indie pop" fan does not match plain "pop".
"""

from typing import Dict, Iterable, List, Optional


def normalize(label: str) -> str:
    """Lowercases a label and collapses surrounding/inner whitespace."""
    return " ".join(str(label).strip().lower().split())


# genre -> parent genre (None for a top-level genre)
GENRE_PARENTS: Dict[str, Optional[str]] = {
    # Top-level genres
    "pop": None,
    "rock": None,
    "hip-hop": None,
    "r&b": None,
    "soul": None,
    "funk": None,
    "jazz": None,
    "blues": None,
    "classical": None,
    "country": None,
    "folk": None,
    "electronic": None,
    "latin": None,
    "reggae": None,
    "gospel": None,
    "lofi": None,
    "soundtrack": None,
    "world": None,

    # Pop
    "indie pop": "pop",
    "synthpop": "pop",
    "k-pop": "pop",
    "dance pop": "pop",

    # Rock
    "metal": "rock",
    "punk": "rock",
    "indie rock": "rock",
    "alternative rock": "rock",
    "hard rock": "rock",
    "grunge": "alternative rock",
    "heavy metal": "metal",

    # Hip-hop
    "trap": "hip-hop",
    "rap": "hip-hop",

    # Jazz / blues
    "smooth jazz": "jazz",
    "bebop": "jazz",

    # Country / folk
    "bluegrass": "country",
    "indie folk": "folk",

    # Electronic
    "synthwave": "electronic",
    "ambient": "electronic",
    "house": "electronic",
    "techno": "electronic",
    "drum and bass": "electronic",
    "dubstep": "electronic",

    # Latin
    "salsa": "latin",
    "reggaeton": "latin",
    "bossa nova": "latin",

    # Reggae
    "dancehall": "reggae",
}

# Other spellings -> canonical genre. Hyphen/space and word-order variants
# ("hip hop", "pop indie") are handled automatically and don't need entries.
GENRE_ALIASES: Dict[str, str] = {
    "hiphop": "hip-hop",
    "rnb": "r&b",
    "r and b": "r&b",
    "rhythm and blues": "r&b",
    "alt rock": "alternative rock",
    "alternative": "alternative rock",
    "lo-fi": "lofi",
    "lo fi": "lofi",
    "lofi hip hop": "lofi",
    "electronica": "electronic",
    "edm": "electronic",
    "dnb": "drum and bass",
    "synth pop": "synthpop",
    "punk rock": "punk",
    "kpop": "k-pop",
    "classic rock": "rock",
}


def _loose_key(label: str) -> str:
    """Spelling-insensitive key: hyphens become spaces and word order is ignored."""
    return " ".join(sorted(normalize(label).replace("-", " ").split()))


class GenreTaxonomy:
    """Maps genre labels onto a controlled list and matches them by parent links."""

    def __init__(
        self,
        parents: Optional[Dict[str, Optional[str]]] = None,
        aliases: Optional[Dict[str, str]] = None,
    ):
        self.parents: Dict[str, Optional[str]] = dict(GENRE_PARENTS if parents is None else parents)
        self.aliases: Dict[str, str] = dict(GENRE_ALIASES if aliases is None else aliases)

        for genre, parent in self.parents.items():
            if parent is not None and parent not in self.parents:
                raise ValueError(f'genre "{genre}" has unknown parent "{parent}"')
        for alias, target in self.aliases.items():
            if target not in self.parents:
                raise ValueError(f'alias "{alias}" points to unknown genre "{target}"')

        self._loose: Dict[str, str] = {}
        for genre in self.parents:
            self._loose[_loose_key(genre)] = genre
        for alias, target in self.aliases.items():
            self._loose.setdefault(_loose_key(alias), target)

    def canonical(self, label: str) -> str:
        """
        Returns the controlled-list name for a label. Unknown labels are
        returned normalized but otherwise unchanged (see is_allowed).
        """
        genre = normalize(label)
        if genre in self.parents:
            return genre
        if genre in self.aliases:
            return self.aliases[genre]
        return self._loose.get(_loose_key(genre), genre)

    def parent(self, genre: str) -> Optional[str]:
        """Returns the parent of a genre, or None for top-level or unknown genres."""
        return self.parents.get(self.canonical(genre))

    def ancestors(self, genre: str) -> List[str]:
        """Returns the parents of a genre, closest first (e.g. grunge -> alternative rock, rock)."""
        chain = []
        current = self.parent(genre)
        while current is not None and current not in chain:
            chain.append(current)
            current = self.parents.get(current)
        return chain

    def is_allowed(self, genre: str) -> bool:
        """True if the label maps onto the controlled list."""
        return self.canonical(genre) in self.parents

    def matches(self, favorites: Iterable[str], song_genre: str) -> Optional[str]:
        """
        Returns the favorite (canonical name) that matches the song's genre, or None.
        An exact match wins; otherwise the closest parent the user listed wins.
        """
        wanted = {self.canonical(favorite) for favorite in favorites}
        song = self.canonical(song_genre)
        if song in wanted:
            return song
        for ancestor in self.ancestors(song):
            if ancestor in wanted:
                return ancestor
        return None


# Shared default instance
TAXONOMY = GenreTaxonomy()
