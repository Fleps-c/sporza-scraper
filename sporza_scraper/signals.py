"""Performance signal detection from Dutch sports text.

Classifies text surrounding player mentions into actionable performance
signals (form, goal threat, injury, etc.) using keyword matching with
proximity scoring.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

log = logging.getLogger(__name__)

SignalType = Literal[
    "positive_form",
    "goal_threat",
    "assist_threat",
    "training_positive",
    "injury_doubt",
    "injury_confirmed",
    "poor_form",
    "transfer_rumour",
    "tactical_role",
    "team_momentum_positive",
    "team_momentum_negative",
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class PerformanceSignal:
    signal_type: SignalType
    player: str
    evidence: str
    confidence: float  # 0.0–1.0
    score: float       # -1.0–+1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Signal definitions: (keywords, base_score)
# Each keyword list is checked case-insensitively against the sentence.
# ---------------------------------------------------------------------------

_SIGNAL_DEFS: dict[SignalType, tuple[list[str], float]] = {
    "positive_form": (
        [
            "in vorm", "uitstekend", "schitterend", "sterk", "dominant",
            "fenomenaal", "briljant", "geweldig", "indrukwekkend",
            "subliem", "prachtig", "prima", "meesterlijk", "beslissend",
            "onhoudbaar", "onstopbaar", "topvorm", "goede vorm",
            "beste speler", "man van de match", "uitblinker",
        ],
        0.75,
    ),
    "goal_threat": (
        [
            "scoorde", "gescoord", "doelpunt", "doelpunten", "topscorer",
            "trefzeker", "goal", "goals", "treffer", "raak schieten",
            "afwerking", "schot op doel", "hattrick", "hat-trick",
            "dubbel", "tweemaal", "drie keer", "koppen raak",
            "van dichtbij", "in de kruising",
        ],
        0.80,
    ),
    "assist_threat": (
        [
            "assist", "assists", "aangever", "creatief", "laatste pass",
            "kansen creëren", "kans creëren", "voorzet", "voorzetten",
            "sleutelpass", "key pass", "beslissende pass", "stiftje",
            "steekpass", "steekbal",
        ],
        0.60,
    ),
    "training_positive": (
        [
            "goed getraind", "klaar", "fit", "hersteld", "terug in selectie",
            "mee getraind", "mee traint", "groepstraining",
            "beschikbaar", "inzetbaar", "wedstrijdfit", "speelklaar",
            "terug van blessure", "volledig fit",
        ],
        0.45,
    ),
    "injury_doubt": (
        [
            "blessure", "twijfelachtig", "geblesseerd", "onzeker",
            "mist mogelijk", "vraagteken", "last van", "klachten",
            "pijn", "hamstring", "knieklachten", "spierblessure",
            "een tik", "aangeslagen", "niet zeker",
        ],
        -0.60,
    ),
    "injury_confirmed": (
        [
            "niet beschikbaar", "out", "afwezig", "geschorst",
            "mist de wedstrijd", "niet inzetbaar", "langdurig",
            "kruisband", "operatie", "geopereerd", "maanden uit",
            "weken uit", "seizoen voorbij", "definitief out",
            "schorsing", "rode kaart",
        ],
        -0.90,
    ),
    "poor_form": (
        [
            "slecht", "wisselvallig", "niet in vorm", "teleurstellend",
            "bank", "op de bank", "gewisseld", "naar de kant",
            "onzichtbaar", "zwak", "matig", "onder de maat",
            "geen indruk", "niet overtuigend", "tegenvallend",
        ],
        -0.50,
    ),
    "transfer_rumour": (
        [
            "transfer", "interesse", "vertrek", "gelinkt aan",
            "wil weg", "bijna rond", "onderhandeling", "bod",
            "aanbod", "overstap", "transitie", "clausule",
            "transfersom", "miljoen", "aanbieding",
        ],
        -0.20,
    ),
    "tactical_role": (
        [
            "basisspeler", "captain", "aanvoerder", "spits",
            "centrale positie", "nummer 10", "vleugelspeler",
            "in de basis", "eerste keuze", "titularis",
            "verdediger", "keeper", "middenvelder",
        ],
        0.30,
    ),
    "team_momentum_positive": (
        [
            "reeks", "ongeslagen", "winnende lijn", "opmars",
            "zegereeks", "puntenrecord", "clean sheet", "clean sheets",
            "indrukwekkende reeks", "vormcurve",
        ],
        0.45,
    ),
    "team_momentum_negative": (
        [
            "crisis", "verliezen op rij", "verliesreeks", "neerwaartse spiraal",
            "degradatie", "degradatiestrijd", "slechtste reeks",
            "al weken niet gewonnen", "dieptepunt",
        ],
        -0.50,
    ),
}

# Negation words that flip a positive signal to negative (or weaken it).
_NEGATION_WORDS = frozenset({
    "niet", "geen", "nooit", "nauwelijks", "amper", "zonder",
})

# Maximum character distance between a player name and a keyword for the
# signal to be attributed to that player with high confidence.
_PROXIMITY_CHARS = 200


def detect_signals(
    paragraphs: list[str],
    player_names: list[str],
) -> list[PerformanceSignal]:
    """Detect performance signals for the given players in the text.

    For each player, scan every sentence in the text for keyword matches.
    Signals found in sentences that also contain the player's name get high
    confidence; signals in adjacent sentences get lower confidence.
    """
    if not paragraphs or not player_names:
        return []

    full_text = "\n".join(paragraphs)
    sentences = _split_sentences(full_text)
    signals: list[PerformanceSignal] = []

    for player in player_names:
        player_lower = player.lower()
        # Find sentence indices where the player is mentioned.
        mention_indices: set[int] = set()
        for i, sent in enumerate(sentences):
            if player_lower in sent.lower():
                mention_indices.add(i)
        if not mention_indices:
            continue

        # Scan sentences near player mentions for signals.
        for i, sent in enumerate(sentences):
            # Calculate proximity: 0 = same sentence, 1 = adjacent, etc.
            min_distance = min(abs(i - mi) for mi in mention_indices)
            if min_distance > 2:
                continue  # too far from any mention

            sent_lower = sent.lower()
            is_negated = any(neg in sent_lower for neg in _NEGATION_WORDS)

            for signal_type, (keywords, base_score) in _SIGNAL_DEFS.items():
                for keyword in keywords:
                    if keyword.lower() in sent_lower:
                        # Calculate confidence based on proximity.
                        if min_distance == 0:
                            confidence = 0.95
                        elif min_distance == 1:
                            confidence = 0.65
                        else:
                            confidence = 0.35

                        score = base_score
                        # Negation flips the sign for form/positive signals.
                        if is_negated and score > 0:
                            score = -score * 0.5
                        elif is_negated and score < 0:
                            score = score * 0.6  # weaken negative signals

                        signals.append(PerformanceSignal(
                            signal_type=signal_type,
                            player=player,
                            evidence=sent.strip()[:200],
                            confidence=round(confidence, 2),
                            score=round(score, 4),
                        ))
                        break  # one match per signal type per sentence

    return signals


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    raw = _SENTENCE_SPLIT_RE.split(text)
    # Also split on newlines.
    sentences: list[str] = []
    for chunk in raw:
        for line in chunk.split("\n"):
            stripped = line.strip()
            if stripped:
                sentences.append(stripped)
    return sentences


def classify_article_type(title: str | None, body: list[str]) -> str:
    """Classify an article into a type based on title and body text."""
    combined = ((title or "") + " " + " ".join(body[:5])).lower()
    if any(kw in combined for kw in ("voorbeschouwing", "preview", "aftrap", "vanavond")):
        return "pre-match preview"
    if any(kw in combined for kw in ("nabeschouwing", "terugblik", "analyse", "samenvatting")):
        return "post-match analysis"
    if any(kw in combined for kw in ("blessure", "geblesseerd", "onzeker", "twijfelachtig")):
        return "injury update"
    if any(kw in combined for kw in ("transfer", "overstap", "interesse", "bod")):
        return "transfer news"
    if any(kw in combined for kw in ("training", "getraind", "selectie")):
        return "training report"
    if any(kw in combined for kw in ("interview", "spreekt", "vertelt", "zegt")):
        return "interview"
    if any(kw in combined for kw in ("stand", "klassement", "ranglijst")):
        return "standings update"
    return "general coverage"
