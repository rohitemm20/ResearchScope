"""Run from backend/:  python -m pytest tests -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.research_gaps import analyze_research_gaps  # noqa: E402


def _blk(i, text):
    return {"id": i, "text": text, "bbox": [50, 100 + i * 20, 500, 120 + i * 20], "type": "text"}


PARSED = {"pages": [
    {"page_number": 1, "blocks": [
        _blk(1, "I. INTRODUCTION"),
        _blk(2, "Existing approaches suffer from significant limitations in handling obfuscated binaries, as shown by Zhang et al. [3]."),
    ]},
    {"page_number": 2, "blocks": [
        _blk(3, "V. DISCUSSION"),
        _blk(4, "Our approach was evaluated only on one dataset, which limits the generality of the findings."),
        _blk(5, "The proposed method has no limitations regarding memory usage on commodity hardware."),
        _blk(6, "VI. CONCLUSION AND FUTURE WORK"),
        _blk(7, "In future work, we plan to extend the framework to Android applications."),
        _blk(8, "REFERENCES"),
        _blk(9, "[1] A. Author, Future work on limitations of everything, 2020."),
    ]},
]}


def test_detects_author_statements_with_provenance():
    result = analyze_research_gaps(PARSED, "abc123abc123")
    cats = {g["category"] for g in result["research_gaps"]}
    assert {"weak_evaluation", "future_opportunities"} <= cats
    top = result["research_gaps"][0]
    assert top["page"] == 2 and "DISCUSSION" in top["section"].upper() or "CONCLUSION" in top["section"].upper()
    assert top["matched_phrases"]


def test_prior_work_criticism_is_flagged_and_ranked_lower():
    gaps = analyze_research_gaps(PARSED, "abc123abc123")["research_gaps"]
    prior = [g for g in gaps if g["scope"] == "prior_work"]
    own = [g for g in gaps if g["scope"] == "this_work"]
    assert prior and own
    assert max(g["score"] for g in prior) < min(g["score"] for g in own)


def test_negated_boilerplate_and_references_are_ignored():
    text = " ".join(g["evidence"] for g in analyze_research_gaps(PARSED, "abc123abc123")["research_gaps"])
    assert "no limitations" not in text
    assert "Author" not in text


def test_plain_text_fallback_and_empty_input():
    ok = analyze_research_gaps("Our approach was evaluated only on one dataset, which limits generality.", "x")
    assert ok["input_mode"] == "plain_text" and ok["total_candidates"] >= 1
    try:
        analyze_research_gaps("   ", "x")
    except ValueError:
        return
    raise AssertionError("empty text should raise ValueError")
