"""
Tests for the harness. These test the *harness code*, not the model's
intelligence. They should pass regardless of which LLM you use.
"""

import json
from dotenv import load_dotenv
import pytest
from pathlib import Path

from src.harness import (
    EvalExample,
    load_corpus,
    parse_label,
    score_example,
    summarize,
)


load_dotenv()
# ---- Corpus loading --------------------------------------------------------


def test_load_corpus(tmp_path: Path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        json.dumps({"id": "a", "request": "hi", "label": "junk", "category": "x"})
        + "\n"
        + json.dumps({"id": "b", "request": "create x", "label": "relevant", "category": "y"})
        + "\n"
    )
    examples = load_corpus(p)
    assert len(examples) == 2
    assert examples[0].id == "a"
    assert examples[1].label == "relevant"


def test_load_corpus_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        ' {"id":"a","request":"x","label":"junk","category":"c"}\n'
        "\n"
        "   \n"
        ' {"id":"b","request":"y","label":"relevant","category":"c"}\n'
    )
    examples = load_corpus(p)
    assert len(examples) == 2


# ---- Output parsing --------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("relevant", "relevant"),
        ("junk", "junk"),
        ("RELEVANT", "relevant"),
        ("Junk.", "junk"),
        ("  relevant  ", "relevant"),
        ("I think this is: relevant", "relevant"),
        ("The answer: junk.", "junk"),
        ("hello world", None),
        ("", None),
        ("yes", None),  # not one of our labels
    ],
)
def test_parse_label(raw: str, expected: str | None):
    assert parse_label(raw) == expected


# ---- Scoring ---------------------------------------------------------------


def _example(label: str = "relevant", category: str = "clear_relevant") -> EvalExample:
    return EvalExample(id="t", request="r", label=label, category=category)


def test_score_example_correct():
    r = score_example(_example("relevant"), "relevant", latency_ms=10.0)
    assert r.correct is True
    assert r.predicted == "relevant"


def test_score_example_wrong_label():
    r = score_example(_example("relevant"), "junk", latency_ms=10.0)
    assert r.correct is False
    assert r.predicted == "junk"


def test_score_example_unparseable():
    r = score_example(_example("relevant"), "garbled output", latency_ms=10.0)
    assert r.correct is False
    assert r.predicted == "<unparseable>"


# ---- Summary metrics -------------------------------------------------------


def test_summary_perfect():
    results = [
        score_example(_example("relevant"), "relevant", 10.0),
        score_example(_example("junk", "c"), "junk", 10.0),
    ]
    s = summarize(results)
    assert s["accuracy"] == 1.0
    assert s["precision_relevant"] == 1.0
    assert s["recall_relevant"] == 1.0
    assert s["precision_junk"] == 1.0
    assert s["recall_junk"] == 1.0
    assert s["n_unparseable"] == 0


def test_summary_all_mislabeled():
    results = [
        score_example(_example("relevant"), "junk", 10.0),
        score_example(_example("junk", "c"), "relevant", 10.0),
    ]
    s = summarize(results)
    assert s["accuracy"] == 0.0
    assert s["precision_relevant"] == 0.0
    assert s["recall_relevant"] == 0.0


def test_summary_unparseable_does_not_crash():
    results = [
        score_example(_example("relevant"), "garbage", 10.0),
        score_example(_example("junk", "c"), "junk", 10.0),
    ]
    s = summarize(results)
    assert s["accuracy"] == 0.5
    assert s["n_unparseable"] == 1


def test_summary_asymmetric_cost_signal():
    """Precision_relevant = 'of things I called relevant, how many were right'.
    If the model over-predicts 'junk' (false negatives for 'relevant'),
    recall_relevant drops. This is the metric you want to watch."""
    # All 4 'relevant' get labeled 'junk'; 0 'junk' get mislabeled.
    results = [
        score_example(_example("relevant", "a"), "junk", 10.0),
        score_example(_example("relevant", "a"), "junk", 10.0),
        score_example(_example("relevant", "a"), "junk", 10.0),
        score_example(_example("relevant", "a"), "junk", 10.0),
    ]
    s = summarize(results)
    assert s["recall_relevant"] == 0.0
    assert s["precision_junk"] == 1.0
    # Direction: model is over-cautious, rejecting real requests.