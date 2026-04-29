"""Extract named-entity PERSON mentions from Dutch text using spaCy.

On first use, the ``nl_core_news_md`` model is loaded (downloaded
automatically if missing). It returns PERSON entities from the text,
optionally filtered against a set of known Premier League players/teams.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

log = logging.getLogger(__name__)


@lru_cache(maxsize=256)
def _get_player_pattern(name: str) -> re.Pattern[str]:
    """Return a compiled word-boundary regex for a player name (cached)."""
    return re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)

_NLP = None  # lazy-loaded spaCy pipeline


def _load_spacy():  # noqa: ANN202
    """Lazy-load the spaCy Dutch model, downloading it if needed."""
    global _NLP  # noqa: PLW0603
    if _NLP is not None:
        return _NLP
    try:
        import spacy
    except ImportError:
        raise RuntimeError(
            "spaCy is required for player extraction. "
            "Install it with:  pip install spacy"
        )
    model_name = "nl_core_news_md"
    try:
        _NLP = spacy.load(model_name)
    except OSError:
        log.info("Downloading spaCy model '%s' (one-time, ~15 MB)…", model_name)
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", model_name],
            stdout=subprocess.DEVNULL,
        )
        _NLP = spacy.load(model_name)
    log.info("spaCy model '%s' loaded", model_name)
    return _NLP


# ---------------------------------------------------------------------------
# Premier League squads: current 2025-26 clubs and a representative sample
# of players. This is not exhaustive — spaCy NER finds names first, and
# this set is used as a secondary confirmation / filter.
# ---------------------------------------------------------------------------

PREMIER_LEAGUE_CLUBS: frozenset[str] = frozenset({
    "Arsenal", "Aston Villa", "AFC Bournemouth", "Bournemouth",
    "Brentford", "Brighton", "Brighton & Hove Albion",
    "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Ipswich Town", "Ipswich", "Leicester City", "Leicester",
    "Liverpool", "Manchester City", "Man City",
    "Manchester United", "Man United",
    "Newcastle United", "Newcastle",
    "Nottingham Forest", "Nott Forest",
    "Southampton", "Tottenham Hotspur", "Tottenham", "Spurs",
    "West Ham United", "West Ham",
    "Wolverhampton Wanderers", "Wolves",
})

# A representative (not exhaustive) set of high-profile PL players.
# The extractor uses NER as the primary signal; this list supplements
# recall for names spaCy might miss in Dutch sports text.
PREMIER_LEAGUE_PLAYERS: frozenset[str] = frozenset({
    # Arsenal
    "Saka", "Ødegaard", "Odegaard", "Saliba", "Rice", "Havertz", "Raya",
    "Timber", "Trossard", "Martinelli", "Gabriel",
    # Aston Villa
    "Watkins", "Onana", "McGinn", "Konsa",
    # Chelsea
    "Palmer", "Caicedo", "Enzo Fernández", "Enzo Fernandez", "Jackson",
    "Mudryk", "Nkunku", "Sanchez",
    # Liverpool
    "Salah", "Van Dijk", "Szoboszlai", "Mac Allister", "Díaz", "Diaz",
    "Nunez", "Núñez", "Alexander-Arnold", "Jota", "Gravenberch",
    # Manchester City
    "Haaland", "De Bruyne", "Foden", "Rodri", "Bernardo Silva",
    "Ederson", "Grealish", "Doku", "Walker", "Stones",
    # Manchester United
    "Fernandes", "Rashford", "Hojlund", "Højlund", "Mount",
    "Garnacho", "Mainoo", "Onana",
    # Newcastle
    "Isak", "Gordon", "Tonali", "Guimarães", "Guimaraes",
    "Trippier",
    # Tottenham
    "Son", "Maddison", "Romero", "Van de Ven", "Bissouma",
    # West Ham
    "Bowen", "Paquetá", "Paqueta", "Kudus",
    # Others
    "Eze", "Olise", "Mbeumo", "Wissa", "Calvert-Lewin",
    "Neto", "Cunha", "Matheus Cunha", "Szmodics",
    # Belgian / Flemish connection (extra coverage for Sporza)
    "De Bruyne", "Trossard", "Onana", "Doku", "Tielemans",
    "Praet", "Castagne", "Faes",
})

# Player → Club mapping for attribution in predictions.
PLAYER_TO_CLUB: dict[str, str] = {
    # Arsenal
    "Saka": "Arsenal", "Odegaard": "Arsenal", "Ødegaard": "Arsenal",
    "Saliba": "Arsenal", "Rice": "Arsenal", "Havertz": "Arsenal",
    "Raya": "Arsenal", "Timber": "Arsenal", "Trossard": "Arsenal",
    "Martinelli": "Arsenal", "Gabriel": "Arsenal",
    # Aston Villa
    "Watkins": "Aston Villa", "McGinn": "Aston Villa", "Konsa": "Aston Villa",
    # Chelsea
    "Palmer": "Chelsea", "Caicedo": "Chelsea", "Jackson": "Chelsea",
    "Mudryk": "Chelsea", "Nkunku": "Chelsea", "Enzo Fernandez": "Chelsea",
    "Enzo Fernández": "Chelsea", "Sanchez": "Chelsea",
    # Liverpool
    "Salah": "Liverpool", "Van Dijk": "Liverpool", "Szoboszlai": "Liverpool",
    "Mac Allister": "Liverpool", "Diaz": "Liverpool", "Díaz": "Liverpool",
    "Nunez": "Liverpool", "Núñez": "Liverpool", "Jota": "Liverpool",
    "Gravenberch": "Liverpool", "Alexander-Arnold": "Liverpool",
    # Manchester City
    "Haaland": "Manchester City", "De Bruyne": "Manchester City",
    "Foden": "Manchester City", "Rodri": "Manchester City",
    "Bernardo Silva": "Manchester City", "Ederson": "Manchester City",
    "Grealish": "Manchester City", "Doku": "Manchester City",
    "Walker": "Manchester City", "Stones": "Manchester City",
    # Manchester United
    "Fernandes": "Manchester United", "Rashford": "Manchester United",
    "Hojlund": "Manchester United", "Højlund": "Manchester United",
    "Mount": "Manchester United", "Garnacho": "Manchester United",
    "Mainoo": "Manchester United",
    # Newcastle
    "Isak": "Newcastle United", "Gordon": "Newcastle United",
    "Tonali": "Newcastle United", "Guimaraes": "Newcastle United",
    "Guimarães": "Newcastle United", "Trippier": "Newcastle United",
    # Tottenham
    "Son": "Tottenham Hotspur", "Maddison": "Tottenham Hotspur",
    "Romero": "Tottenham Hotspur", "Van de Ven": "Tottenham Hotspur",
    "Bissouma": "Tottenham Hotspur",
    # West Ham
    "Bowen": "West Ham United", "Paqueta": "West Ham United",
    "Paquetá": "West Ham United", "Kudus": "West Ham United",
    # Others
    "Eze": "Crystal Palace", "Olise": "Crystal Palace",
    "Mbeumo": "Brentford", "Wissa": "Brentford",
    "Calvert-Lewin": "Everton", "Neto": "Wolverhampton Wanderers",
    "Cunha": "Wolverhampton Wanderers", "Matheus Cunha": "Wolverhampton Wanderers",
    "Szmodics": "Ipswich Town",
    # Belgian extras
    "Tielemans": "Aston Villa", "Praet": "Leicester City",
    "Castagne": "Fulham", "Faes": "Leicester City",
    "Onana": "Manchester United",
}


@dataclass(slots=True)
class PlayerMention:
    """A single detected player name occurrence."""
    name: str
    start_char: int
    end_char: int
    context: str  # surrounding sentence / short snippet


@dataclass(slots=True)
class ExtractionResult:
    """All player mentions found in a body of text."""
    mentions: list[PlayerMention] = field(default_factory=list)
    unique_names: list[str] = field(default_factory=list)


def extract_player_mentions(
    paragraphs: Sequence[str],
    *,
    premier_league_only: bool = True,
) -> ExtractionResult:
    """Detect player names in Dutch paragraphs.

    Uses two strategies in combination:

    1. **spaCy NER** — finds PERSON entities, filtered against the
       known player set (when ``premier_league_only=True``).
    2. **Dictionary matching** — scans text for known player names
       from ``PREMIER_LEAGUE_PLAYERS`` directly. This catches names
       that spaCy's Dutch model misses (common in sports text).

    Parameters
    ----------
    paragraphs:
        The article's body text, one string per paragraph.
    premier_league_only:
        If True (default), only keep names that appear in the
        ``PREMIER_LEAGUE_PLAYERS`` set or seem to reference a
        Premier League context. Set False to return all persons.
    """
    full_text = "\n".join(paragraphs)

    mentions: list[PlayerMention] = []
    seen_names: dict[str, None] = {}  # ordered-set emulation

    # --- Strategy 1: spaCy NER ---
    try:
        nlp = _load_spacy()
        doc = nlp(full_text)
        for ent in doc.ents:
            if ent.label_ != "PER":
                continue
            name = ent.text.strip()
            if not name or len(name) < 2:
                continue
            if premier_league_only and not _is_premier_league_name(name, full_text):
                log.debug("NER found '%s' but filtered out (not PL-related)", name)
                continue
            # Build a context snippet: the sentence containing the mention.
            sent = ent.sent.text.strip() if ent.sent else ""
            context = sent[:200] if sent else full_text[max(0, ent.start_char - 60):ent.end_char + 60]
            mentions.append(
                PlayerMention(
                    name=name,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    context=context,
                )
            )
            if name not in seen_names:
                seen_names[name] = None
    except Exception as exc:
        log.warning("spaCy NER failed (%s); falling back to dictionary only", exc)

    # --- Strategy 2: Dictionary-based matching ---
    # Catches player names spaCy's Dutch model misses.
    already_found = {n.lower() for n in seen_names}
    for player_name in PREMIER_LEAGUE_PLAYERS:
        if player_name.lower() in already_found:
            continue
        # Use pre-compiled patterns from cache (avoids recompiling per call).
        pattern = _get_player_pattern(player_name)
        for match in pattern.finditer(full_text):
            # Extract a context snippet around the match.
            start = max(0, match.start() - 80)
            end = min(len(full_text), match.end() + 80)
            context = full_text[start:end].strip()
            canonical = _canonicalise_name(player_name)
            mentions.append(
                PlayerMention(
                    name=canonical,
                    start_char=match.start(),
                    end_char=match.end(),
                    context=context,
                )
            )
            if canonical not in seen_names:
                seen_names[canonical] = None
            # Only record the first occurrence per player name for uniqueness,
            # but keep adding mentions for count tracking.

    ner_count = sum(1 for m in mentions if m.name in {n for n in seen_names})
    log.debug(
        "Extraction: %d total mentions, %d unique players (%d from NER, %d from dict)",
        len(mentions), len(seen_names),
        len([n for n in seen_names if n in already_found or True]),
        len(seen_names) - len(already_found),
    )

    return ExtractionResult(
        mentions=mentions,
        unique_names=list(seen_names.keys()),
    )


def _canonicalise_name(name: str) -> str:
    """Return the preferred display form of a player name."""
    # Use PLAYER_TO_CLUB keys as canonical forms.
    if name in PLAYER_TO_CLUB:
        return name
    # Try case-insensitive match.
    lower = name.lower()
    for canonical in PLAYER_TO_CLUB:
        if canonical.lower() == lower:
            return canonical
    return name


def _is_premier_league_name(name: str, surrounding_text: str) -> bool:
    """Heuristic: does this PERSON entity relate to the Premier League?"""
    # Direct match against known player surnames.
    parts = name.split()
    for part in parts:
        if part in PREMIER_LEAGUE_PLAYERS:
            return True
    if name in PREMIER_LEAGUE_PLAYERS:
        return True
    # Check if the surrounding text mentions a PL club.
    lowered = surrounding_text.lower()
    for club in PREMIER_LEAGUE_CLUBS:
        if club.lower() in lowered:
            return True
    # Common English PL keywords in Dutch sports writing.
    if any(kw in lowered for kw in ("premier league", "engelse competitie", "engeland")):
        return True
    return False
