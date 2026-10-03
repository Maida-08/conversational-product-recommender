"""Tests for recommendation selection: primary, secondary, cautions, modes."""
import pytest

from app.config import get_settings
from app.data.loader import load_catalog
from app.services.attribute_extractor import rules_extract
from app.services.query_understanding import rules_parse
from app.services.ranker import Ranker
from app.services.recommender import recommend, review_flags
from app.services.retriever import Retriever
from app.services.review_analyzer import ensure_review_evidence

pytestmark = pytest.mark.skipif(not get_settings().catalog_csv.exists(), reason="synthetic catalog missing")


@pytest.fixture(scope="module")
def products():
    return ensure_review_evidence([rules_extract(r) for r in load_catalog()])


@pytest.fixture(scope="module")
def ranked_for(products):
    retriever, ranker = Retriever(products), Ranker(products)

    def _go(message):
        q = rules_parse(message)
        return ranker.rank(q, retriever.retrieve(q).candidates)
    return _go


def pid(r):
    return r.product.product_id


# ------------------------------------------------------------------ scenario A
def test_scenario_a_primary_is_the_over_budget_requirement_match(ranked_for):
    rec = recommend(ranked_for("I need a lightweight waterproof running shoe, ideally under $80"))
    assert pid(rec.primary) == "P001" and rec.mode == "tradeoff"
    assert any("over your $80 budget" in t for t in rec.primary.tradeoffs)     # stated, not hidden


def test_scenario_a_secondary_fixes_the_budget_tradeoff(ranked_for):
    rec = recommend(ranked_for("I need a lightweight waterproof running shoe, ideally under $80"))
    assert pid(rec.secondary[0]) == "P002" and not rec.secondary[0].over_budget_by


def test_scenario_a_contradicted_product_is_a_caution_not_a_recommendation(ranked_for):
    rec = recommend(ranked_for("I need a lightweight waterproof running shoe, ideally under $80"))
    shown = {pid(rec.primary), *(pid(r) for r in rec.secondary)}
    assert "P006" not in shown and "P006" in {pid(r) for r in rec.cautions}
    assert rec.cautions[0].label == "conflict"


# ------------------------------------------------------------------ scenario C
def test_scenario_c_review_flagged_sneaker_is_a_caution_with_its_warning(ranked_for):
    rec = recommend(ranked_for("I want a casual sneaker that won't fall apart in light rain"))
    caution = next(r for r in rec.cautions if pid(r) == "P003")
    assert any("reviewers" in t for t in caution.tradeoffs)
    assert "P003" not in {pid(rec.primary), *(pid(r) for r in rec.secondary)}


def test_near_tie_prefers_the_option_without_review_warnings(ranked_for):
    ranked = ranked_for("I want a casual sneaker that won't fall apart in light rain")
    rec = recommend(ranked)
    eligible = [r for r in ranked if r.label in ("strong", "partial")]
    assert pid(rec.primary) != pid(eligible[0]) or review_flags(eligible[0]) == 0
    assert review_flags(rec.primary) == 0
    assert any("fewer review warnings" in n for n in rec.notes)


def test_secondary_is_not_worse_than_primary_on_hard_requirements(ranked_for):
    rec = recommend(ranked_for("I want a casual sneaker that won't fall apart in light rain"))
    unmet = lambda r: sum(1 for q in r.requirements if q.kind == "hard" and q.state != "met")
    assert all(unmet(r) <= unmet(rec.primary) for r in rec.secondary)
    assert "P011" not in {pid(r) for r in rec.secondary}      # road shoe: water resistance not stated


# ------------------------------------------------------------------ scenario D
def test_scenario_d_basic_earbuds_are_a_confident_primary(ranked_for):
    rec = recommend(ranked_for("Looking for cheap wireless earbuds, just need something basic"))
    assert pid(rec.primary) == "P014" and rec.mode == "confident"


def test_scenario_d_bad_options_are_never_recommended(ranked_for):
    rec = recommend(ranked_for("Looking for cheap wireless earbuds, just need something basic"))
    shown = {pid(rec.primary), *(pid(r) for r in rec.secondary)}
    assert "P026" not in shown                 # wired earphones
    assert "P017" not in shown                 # cheapest, but 3.4 stars


def test_excluded_ids_are_not_shown_twice(ranked_for):
    ranked = ranked_for("Looking for cheap wireless earbuds, just need something basic")
    rec = recommend(ranked, exclude_ids={"P015"})
    assert "P015" not in {pid(rec.primary), *(pid(r) for r in rec.secondary)}


# ---------------------------------------------------------------- modes and limits
def test_closest_available_when_nothing_is_eligible(ranked_for):
    ranked = [r.model_copy(update={"label": "poor"}) for r in ranked_for("running shoe")]
    rec = recommend(ranked)
    assert rec.mode == "closest_available" and rec.secondary == [] and rec.notes


def test_conflict_product_is_the_fallback_primary_when_nothing_else_is_eligible(ranked_for):
    ranked = ranked_for("running shoe")
    for r in ranked:
        r.label = "poor"
    ranked[1].label = "conflict"
    assert recommend(ranked).primary is ranked[1]


def test_secondary_is_capped_and_unique(ranked_for):
    rec = recommend(ranked_for("running shoe"))
    ids = [pid(rec.primary)] + [pid(r) for r in rec.secondary] + [pid(r) for r in rec.cautions]
    assert len(rec.secondary) <= 2 and len(rec.cautions) <= 2 and len(ids) == len(set(ids))


def test_empty_ranking_returns_none():
    assert recommend([]) is None


def test_primary_is_never_poor_when_eligible_products_exist(ranked_for):
    for msg in ["running shoe", "wireless earbuds", "waterproof rain jacket", "air purifier"]:
        ranked = ranked_for(msg)
        if any(r.label in ("strong", "partial") for r in ranked):
            assert recommend(ranked).primary.label in ("strong", "partial")