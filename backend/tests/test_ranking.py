"""Tests for the ranker. Uses rules-only extraction (no LLM, no processed JSON)."""
import pytest

from app.config import get_settings
from app.data.loader import load_catalog
from app.models.schemas import Budget
from app.services.attribute_extractor import rules_extract
from app.services.query_understanding import rules_parse
from app.services.ranker import (CatalogStats, Ranker, budget_fit, evaluate_requirement, shrunk_rating)
from app.services.retriever import Retriever

pytestmark = pytest.mark.skipif(not get_settings().catalog_csv.exists(), reason="synthetic catalog missing")


@pytest.fixture(scope="module")
def products():
    return [rules_extract(r) for r in load_catalog()]


@pytest.fixture(scope="module")
def by_id(products):
    return {p.product_id: p for p in products}


@pytest.fixture(scope="module")
def stats(products):
    return CatalogStats.from_products(products)


@pytest.fixture(scope="module")
def run(products):
    retriever, ranker = Retriever(products), Ranker(products)

    def _run(message):
        q = rules_parse(message)
        return {r.product.product_id: r for r in ranker.rank(q, retriever.retrieve(q).candidates)}
    return _run


SCENARIO_A = "I need a lightweight waterproof running shoe, ideally under $80"


# ------------------------------------------------- hard requirements beat budget
def test_meets_hard_requirement_but_over_budget_beats_in_budget_that_does_not(run):
    r = run(SCENARIO_A)
    assert r["P001"].rank < r["P002"].rank        # $89.50, waterproof  vs  $64.99, waterproof unknown
    assert r["P001"].rank < r["P011"].rank        # vs $39.99, nothing known


def test_over_budget_tradeoff_is_stated_with_the_amount(run):
    r = run(SCENARIO_A)["P001"]
    assert r.over_budget_by == pytest.approx(9.5)
    assert any("$9.50 over your $80 budget" in t for t in r.tradeoffs)


def test_unknown_requirement_is_flagged_not_hidden(run):
    r = run(SCENARIO_A)["P002"]
    assert any("waterproof" in t and "not stated" in t for t in r.tradeoffs)


def test_hard_requirement_changes_the_ranking(run):
    with_req = run("waterproof lightweight running shoe under $80")
    without = run("lightweight running shoe under $80")
    assert with_req["P001"].rank < without["P001"].rank


# -------------------------------------------------------------- unknown is not false
def test_unknown_scores_between_unmet_and_met(run):
    r = run(SCENARIO_A)
    unknown = next(x for x in r["P002"].requirements if x.requirement == "waterproof")
    unmet = next(x for x in r["P005"].requirements if x.requirement == "waterproof")
    met = next(x for x in r["P001"].requirements if x.requirement == "waterproof")
    assert (unmet.state, unknown.state, met.state) == ("unmet", "unknown", "met")
    assert unmet.satisfaction < unknown.satisfaction < met.satisfaction


def test_explicitly_not_waterproof_is_poor_but_unknown_is_partial(run):
    r = run(SCENARIO_A)
    assert r["P005"].label == "poor" and r["P002"].label == "partial"
    assert r["P002"].score > r["P005"].score


# --------------------------------------------------------------------- budget maths
def test_budget_fit_within_budget_is_full():
    assert budget_fit(79.0, Budget(max=80)) == (1.0, None)


def test_budget_fit_no_budget_is_none():
    assert budget_fit(50.0, Budget()) == (None, None)


def test_firmness_controls_how_hard_overshoot_is_punished():
    soft, _ = budget_fit(89.5, Budget(max=80, firmness="soft"))
    std, over = budget_fit(89.5, Budget(max=80, firmness="standard"))
    firm, _ = budget_fit(89.5, Budget(max=80, firmness="firm"))
    assert soft > std > firm > 0 and over == pytest.approx(9.5)


def test_far_over_budget_scores_zero_and_minimum_is_respected():
    assert budget_fit(200.0, Budget(max=80))[0] == 0.0
    assert budget_fit(50.0, Budget(min=100, firmness="standard"))[0] < 1.0
    assert budget_fit(150.0, Budget(min=100))[0] == 1.0


# ------------------------------------------------------- scenario D: cheap + basic
def test_scenario_d_basic_earbuds_first_and_wired_earphones_poor(run):
    r = run("Looking for cheap wireless earbuds, just need something basic")
    assert r["P014"].rank == 1 and r["P014"].label == "strong"
    assert r["P026"].label == "poor"
    assert next(x for x in r["P026"].requirements if x.requirement == "wireless").state == "unmet"


def test_cheapest_but_poorly_rated_is_penalised_and_not_strong(run):
    r = run("Looking for cheap wireless earbuds, just need something basic")
    assert r["P017"].rank > r["P014"].rank
    assert any("low rating" in p for p in r["P017"].penalties) and r["P017"].label == "partial"


def test_cheap_tone_prefers_lower_price_when_no_budget_number(run):
    r = run("cheap wireless earbuds")
    budget = lambda x: next(c.value for c in x.components if c.name == "budget")
    assert budget(r["P014"]) > budget(r["P015"])


def test_pro_earbuds_stay_available_as_an_option(run):
    r = run("Looking for cheap wireless earbuds, just need something basic")
    assert "P015" in r and r["P015"].rank > r["P014"].rank


# ------------------------------------------------------- scenario C and audience
def test_kids_product_is_penalised_unless_a_child_is_mentioned(run):
    r = run("I want a casual sneaker that won't fall apart in light rain")
    assert any("kids" in p for p in r["P013"].penalties) and r["P013"].rank > r["P003"].rank
    kid = run("a sneaker for my kid")
    assert not any("kids" in p for p in kid["P013"].penalties)


def test_description_silence_on_rain_is_unknown_before_reviews(run):
    r = run("I want a casual sneaker that won't fall apart in light rain")
    state = next(x for x in r["P003"].requirements if x.requirement == "water_resistant").state
    assert state == "unknown"      # Phase 7 will use the reviews to flag this


# ---------------------------------------------------------- upsell-relevant ordering
def test_large_room_uses_the_stated_coverage(run):
    r = run("I want the best air purifier for a large open-plan living room")
    assert r["P034"].rank == 1 and r["P034"].label == "strong"
    assert r["P033"].label == "poor"


# ----------------------------------------------------------- requirement evaluators
def test_lightweight_uses_per_shoe_weight_and_halves_pairs(by_id, stats):
    assert evaluate_requirement("lightweight", by_id["P002"], stats).state == "met"       # 221 g
    assert evaluate_requirement("lightweight", by_id["P001"], stats).state == "partial"   # 289 g
    heavy = evaluate_requirement("lightweight", by_id["P004"], stats)                      # 850 g / pair
    assert heavy.state == "unmet" and "425" in heavy.note


def test_claimed_lightweight_without_a_weight_is_flagged(by_id, stats):
    r = evaluate_requirement("lightweight", by_id["P006"], stats)
    assert r.state == "met" and r.satisfaction == 0.8 and "no weight is given" in r.note


def test_text_evidence_found_is_met_and_silence_is_unknown(by_id, stats):
    assert evaluate_requirement("quiet", by_id["P033"], stats).state == "met"
    assert evaluate_requirement("quiet", by_id["P014"], stats).state == "unknown"
    assert evaluate_requirement("non_slip", by_id["P052"], stats).state == "met"
    assert evaluate_requirement("non_slip", by_id["P051"], stats).state == "unknown"


def test_evidence_quote_is_verbatim_from_description(by_id, stats):
    r = evaluate_requirement("quiet", by_id["P033"], stats)
    assert r.evidence and r.evidence in by_id["P033"].description


def test_large_room_numeric_check(by_id, stats):
    assert evaluate_requirement("large_room", by_id["P034"], stats).state == "met"
    assert evaluate_requirement("large_room", by_id["P033"], stats).state == "unmet"


# ------------------------------------------------------------ quality and weights
def test_rating_is_shrunk_by_review_count(by_id):
    few = by_id["P014"].model_copy(update={"average_rating": 4.9, "num_reviews": 3})
    many = by_id["P014"].model_copy(update={"average_rating": 4.6, "num_reviews": 2000})
    assert shrunk_rating(few) < shrunk_rating(many)


def test_best_raises_the_quality_weight(run):
    q = lambda r: next(c.weight for c in r.components if c.name == "quality")
    assert q(run("best running shoe")["P002"]) > q(run("running shoe")["P002"])


def test_components_are_normalised_and_only_applicable_ones_appear(run):
    no_budget = run("waterproof running shoe")["P001"]
    names = {c.name for c in no_budget.components}
    assert "budget" not in names and {"type", "hard", "use_text", "quality"} <= names
    assert sum(c.weight for c in no_budget.components) == pytest.approx(1.0, abs=0.01)
    with_budget = run("waterproof running shoe under $100")["P001"]
    assert "budget" in {c.name for c in with_budget.components}


def test_score_equals_component_sum_when_no_penalties(run):
    r = run(SCENARIO_A)["P001"]
    assert not r.penalties
    assert r.score == pytest.approx(sum(c.contribution for c in r.components), abs=0.01)


# ------------------------------------------------------------------------ general
def test_ranks_are_sequential_sorted_bounded_and_deterministic(products):
    retriever, ranker = Retriever(products), Ranker(products)
    q = rules_parse(SCENARIO_A)
    a = ranker.rank(q, retriever.retrieve(q).candidates)
    b = ranker.rank(q, retriever.retrieve(q).candidates)
    assert [x.product.product_id for x in a] == [x.product.product_id for x in b]
    assert [x.rank for x in a] == list(range(1, len(a) + 1))
    assert [x.score for x in a] == sorted([x.score for x in a], reverse=True)
    assert all(0.0 <= x.score <= 1.0 for x in a)


def test_empty_candidates_return_empty_list(products):
    assert Ranker(products).rank(rules_parse("running shoe"), []) == []