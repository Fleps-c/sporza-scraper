"""Dutch-language sentiment analyser.

Uses a built-in lexicon of ~350 Dutch words scored from -1.0 (very negative)
to +1.0 (very positive). Each text is tokenised, matched against the lexicon,
and scored as the mean of all matching tokens. Negation words ("niet", "geen",
"nooit", …) flip the sign of the next scored word.

This is lightweight and requires no external model downloads. For higher
accuracy, swap in the ``transformers`` pipeline with the
``DTAI-KULeuven/robbert-v2-dutch-sentiment`` model (see README).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")

# ---------------------------------------------------------------------------
# Dutch sentiment lexicon (manually curated core subset).
# Scores: -1.0 … +1.0
# ---------------------------------------------------------------------------
_LEXICON: dict[str, float] = {
    # --- strong positive ---
    "uitstekend": 1.0, "fantastisch": 1.0, "briljant": 1.0, "geweldig": 0.9,
    "fenomenaal": 1.0, "magistraal": 1.0, "subliem": 1.0, "perfect": 1.0,
    "onstopbaar": 0.9, "heerlijk": 0.9, "schitterend": 0.9, "prachtig": 0.9,
    "indrukwekkend": 0.9, "overweldigend": 0.8, "meesterlijk": 0.9,
    # --- positive ---
    "goed": 0.6, "sterk": 0.6, "winnaar": 0.7, "winst": 0.7, "zege": 0.7,
    "overwinning": 0.7, "heldhaftig": 0.7, "held": 0.7, "succesvol": 0.7,
    "succes": 0.6, "prima": 0.6, "mooi": 0.6, "knap": 0.6, "snel": 0.5,
    "alert": 0.5, "betrouwbaar": 0.6, "dominant": 0.6, "aanvallend": 0.4,
    "scoren": 0.6, "gescoord": 0.6, "goal": 0.5, "doelpunt": 0.5,
    "winnend": 0.6, "assist": 0.5, "talent": 0.6, "getalenteerd": 0.7,
    "topscorer": 0.7, "kampioen": 0.8, "titel": 0.5, "beker": 0.5,
    "feest": 0.6, "blij": 0.6, "tevreden": 0.5, "trots": 0.6,
    "hoopvol": 0.5, "verdiend": 0.5, "lof": 0.6, "applaus": 0.6,
    "juichen": 0.6, "vieren": 0.6, "progressie": 0.5, "verbetering": 0.5,
    "terugkeer": 0.4, "comeback": 0.5, "debuut": 0.4, "leider": 0.5,
    "stabiel": 0.4, "solide": 0.5, "efficiënt": 0.5, "creatief": 0.5,
    "beslissend": 0.6, "cruciaal": 0.4, "inspirerend": 0.6, "kans": 0.3,
    # --- mild positive ---
    "redelijk": 0.3, "oké": 0.2, "aanvaardbaar": 0.2, "gemiddeld": 0.1,
    "normaal": 0.1,
    # --- mild negative ---
    "matig": -0.3, "wisselvallig": -0.3, "onzeker": -0.3, "twijfel": -0.3,
    "moeizaam": -0.3, "ongelukkig": -0.4, "pech": -0.4, "missen": -0.3,
    "gemist": -0.4, "naast": -0.2, "fout": -0.4, "fouten": -0.4,
    "zwak": -0.5, "tegenvaller": -0.5, "tegenvallend": -0.5,
    "probleem": -0.4, "problemen": -0.4, "moeite": -0.3,
    # --- negative ---
    "verlies": -0.7, "verloren": -0.7, "verliezen": -0.6,
    "nederlaag": -0.7, "degradatie": -0.8, "uitschakeling": -0.6,
    "teleurstelling": -0.7, "teleurstellend": -0.7, "slecht": -0.6,
    "blessure": -0.6, "geblesseerd": -0.6, "geschorst": -0.5,
    "schorsing": -0.5, "rode kaart": -0.6, "penalty gemist": -0.6,
    "strafschop gemist": -0.6, "eigen doelpunt": -0.7, "owngoal": -0.7,
    "boete": -0.5, "sanctie": -0.5, "frustratie": -0.5, "gefrustreerd": -0.5,
    "woede": -0.6, "woedend": -0.7, "kritiek": -0.5, "bekritiseerd": -0.5,
    "controversieel": -0.4, "schandaal": -0.7, "crisis": -0.6,
    "onrust": -0.5, "opstootje": -0.5, "incident": -0.4, "provocatie": -0.5,
    "zwakte": -0.5, "machteloos": -0.6, "kansloos": -0.7, "hopeloos": -0.8,
    # --- strong negative ---
    "rampzalig": -1.0, "desastreus": -1.0, "verschrikkelijk": -0.9,
    "catastrofaal": -1.0, "afgrijselijk": -0.9, "vernederend": -0.9,
    "vernederd": -0.9, "afgang": -0.8, "fiasco": -0.9, "debacle": -0.9,
    "drama": -0.7, "nachtmerrie": -0.8,
}

# Negation words flip the sign of the next scored token.
_NEGATION_WORDS = frozenset(
    {"niet", "geen", "nooit", "noch", "nergens", "niemand", "niets", "nauwelijks", "amper"}
)

# Intensifiers multiply the score of the next scored token.
_INTENSIFIERS: dict[str, float] = {
    "zeer": 1.4, "heel": 1.3, "erg": 1.3, "enorm": 1.5, "bijzonder": 1.3,
    "uiterst": 1.5, "ongelooflijk": 1.5, "verschrikkelijk": 1.4, "echt": 1.2,
    "absoluut": 1.4, "totaal": 1.3, "extreem": 1.5,
}


@dataclass(slots=True)
class SentimentResult:
    """Sentiment score for a piece of text."""
    score: float       # -1.0 … +1.0
    label: str         # "positive" | "negative" | "neutral"
    word_hits: int     # number of lexicon matches
    token_count: int   # total tokens scanned


def analyse_text(text: str) -> SentimentResult:
    """Score a Dutch text using the built-in lexicon."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    scores: list[float] = []
    negate_next = False
    intensify_next = 1.0
    for token in tokens:
        if token in _NEGATION_WORDS:
            negate_next = True
            continue
        if token in _INTENSIFIERS:
            intensify_next = _INTENSIFIERS[token]
            continue
        base = _LEXICON.get(token)
        if base is not None:
            adjusted = base * intensify_next
            if negate_next:
                adjusted = -adjusted
            scores.append(max(-1.0, min(1.0, adjusted)))
        negate_next = False
        intensify_next = 1.0

    if not scores:
        return SentimentResult(score=0.0, label="neutral", word_hits=0, token_count=len(tokens))

    mean = sum(scores) / len(scores)
    label = "positive" if mean > 0.05 else "negative" if mean < -0.05 else "neutral"
    return SentimentResult(
        score=round(mean, 4),
        label=label,
        word_hits=len(scores),
        token_count=len(tokens),
    )


def analyse_paragraphs(paragraphs: list[str]) -> SentimentResult:
    """Score an entire article by averaging paragraph scores."""
    results = [analyse_text(p) for p in paragraphs if p.strip()]
    if not results:
        return SentimentResult(score=0.0, label="neutral", word_hits=0, token_count=0)
    total_hits = sum(r.word_hits for r in results)
    total_tokens = sum(r.token_count for r in results)
    # Weighted average by word hits (paragraphs with more sentiment words count more).
    if total_hits == 0:
        mean = 0.0
    else:
        mean = sum(r.score * r.word_hits for r in results) / total_hits
    label = "positive" if mean > 0.05 else "negative" if mean < -0.05 else "neutral"
    return SentimentResult(score=round(mean, 4), label=label, word_hits=total_hits, token_count=total_tokens)
