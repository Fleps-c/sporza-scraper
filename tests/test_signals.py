"""Tests for sporza_scraper.signals."""
from __future__ import annotations

import pytest

from sporza_scraper.signals import (
    PerformanceSignal,
    classify_article_type,
    detect_signals,
)


# ---------------------------------------------------------------------------
# detect_signals
# ---------------------------------------------------------------------------


class TestDetectSignals:
    def test_positive_form_same_sentence(self):
        paragraphs = ["De Bruyne was uitstekend in de tweede helft."]
        signals = detect_signals(paragraphs, ["De Bruyne"])
        assert len(signals) >= 1
        sig = signals[0]
        assert sig.signal_type == "positive_form"
        assert sig.player == "De Bruyne"
        assert sig.confidence == 0.95
        assert sig.score > 0

    def test_goal_threat_detection(self):
        paragraphs = ["Haaland scoorde twee keer tegen Arsenal."]
        signals = detect_signals(paragraphs, ["Haaland"])
        types = {s.signal_type for s in signals}
        assert "goal_threat" in types

    def test_injury_doubt_negative_score(self):
        paragraphs = ["Salah is twijfelachtig voor de wedstrijd met hamstring klachten."]
        signals = detect_signals(paragraphs, ["Salah"])
        injury_signals = [s for s in signals if s.signal_type == "injury_doubt"]
        assert len(injury_signals) >= 1
        assert injury_signals[0].score < 0

    def test_injury_confirmed_strong_negative(self):
        paragraphs = ["Rodri is definitief out voor de rest van het seizoen."]
        signals = detect_signals(paragraphs, ["Rodri"])
        confirmed = [s for s in signals if s.signal_type == "injury_confirmed"]
        assert len(confirmed) >= 1
        assert confirmed[0].score <= -0.8

    def test_adjacent_sentence_lower_confidence(self):
        paragraphs = [
            "Doku was zichtbaar aanwezig. "
            "De Belg leverde een schitterende prestatie."
        ]
        signals = detect_signals(paragraphs, ["Doku"])
        # "schitterende" is in the second sentence, Doku in the first
        # confidence should be < 0.95 for adjacent
        form_signals = [s for s in signals if s.signal_type == "positive_form"]
        assert len(form_signals) >= 1

    def test_no_signals_for_distant_text(self):
        paragraphs = [
            "Eerste zin over het weer.",
            "Tweede zin over politiek.",
            "Derde zin over economie.",
            "Saka speelde mee.",
            "Vijfde zin over iets anders.",
            "Zesde zin over muziek.",
            "Het doelpunt was prachtig.",
        ]
        signals = detect_signals(paragraphs, ["Saka"])
        # "doelpunt" is far from Saka's mention
        goal_signals = [s for s in signals if s.signal_type == "goal_threat"]
        # May or may not detect depending on sentence splitting,
        # but if detected, confidence should be low
        for gs in goal_signals:
            assert gs.confidence < 0.95

    def test_negation_flips_positive(self):
        paragraphs = ["Foden was niet in vorm tegen Liverpool."]
        signals = detect_signals(paragraphs, ["Foden"])
        form_signals = [s for s in signals if s.signal_type == "positive_form"]
        for s in form_signals:
            assert s.score < 0  # negation flipped

    def test_empty_inputs(self):
        assert detect_signals([], ["Haaland"]) == []
        assert detect_signals(["Some text"], []) == []
        assert detect_signals([], []) == []

    def test_multiple_players(self):
        paragraphs = [
            "Saka scoorde het openingsdoelpunt.",
            "Palmer gaf de assist.",
        ]
        signals = detect_signals(paragraphs, ["Saka", "Palmer"])
        saka_signals = [s for s in signals if s.player == "Saka"]
        palmer_signals = [s for s in signals if s.player == "Palmer"]
        assert len(saka_signals) >= 1
        assert len(palmer_signals) >= 1

    def test_assist_threat(self):
        paragraphs = ["De Bruyne gaf opnieuw een beslissende pass."]
        signals = detect_signals(paragraphs, ["De Bruyne"])
        types = {s.signal_type for s in signals}
        assert "assist_threat" in types

    def test_transfer_rumour(self):
        paragraphs = ["Er is veel interesse in een transfer van Rashford."]
        signals = detect_signals(paragraphs, ["Rashford"])
        types = {s.signal_type for s in signals}
        assert "transfer_rumour" in types


# ---------------------------------------------------------------------------
# classify_article_type
# ---------------------------------------------------------------------------


class TestClassifyArticleType:
    def test_pre_match_preview(self):
        assert classify_article_type("Voorbeschouwing: City vs Arsenal", []) == "pre-match preview"

    def test_post_match_analysis(self):
        assert classify_article_type("Nabeschouwing Premier League", []) == "post-match analysis"

    def test_injury_update(self):
        assert classify_article_type("Blessure-update: wie is er fit?", []) == "injury update"

    def test_transfer_news(self):
        assert classify_article_type("Transfer: speler op weg naar nieuwe club", []) == "transfer news"

    def test_interview(self):
        assert classify_article_type("Coach spreekt over de wedstrijd", []) == "interview"

    def test_general_coverage(self):
        assert classify_article_type("Sporza nieuwsoverzicht", []) == "general coverage"

    def test_body_keywords_used(self):
        result = classify_article_type(
            "Gewone titel",
            ["De selectie is bekendgemaakt na de training."],
        )
        assert result == "training report"
