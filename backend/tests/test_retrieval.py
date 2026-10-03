"""Tests for candidate retrieval. Uses rules-only extraction so no LLM or processed JSON is needed."""
import pytest

from app.config import get_settings
from app.data.loader import load_catalog
from app.models.schemas import StructuredQuery
from app.services.attribute_extractor import rules_extract
from app.services.query_understanding import rules_parse
from app.services.retriever import Retriever, type_relevance

pytestmark = pytest.mark.skipif(not get_settings().catalog_csv.exists(), reason="synthetic catalog missing")


@pytest.fixture(scope="module")
def products():
    return [rules_extract(r) for r in load_catalog()]


@pytest.fixture(scope="module")
def retriever(products):
    return Retriever(products)


@pytest.fixture(scope="module")
def brands(products):
    return sorted({p.brand for p in products})


def ids(res):
    return [c.product.product_id for c in res.candidates]


def rank(res, pid):
    return ids(res).index(pid)


# ------------------------------------------------------------ scenario A
def test_scenario_a_candidates_include_over_budget_best_match(retriever):
    q = rules_parse("I need a lightweight waterproof running shoe, ideally under $80")
    res = retriever.retrieve(q)
    got = ids(res)
    # P001 costs $89.50 (> $80): budget must NOT filter it out
    assert {"P001", "P002", "P006", "P011"} <= set(got)
    assert all(c.product.category == "footwear" for c in res.candidates[:8])


def test_budget_never_changes_the_candidate_set(retriever):
    with_budget = ids(retriever.retrieve(rules_parse("running shoe under $45")))
    without = ids(retriever.retrieve(rules_parse("running shoe")))
    assert set(with_budget) == set(without)


def test_runner_in_title_counts_as_running_shoe(products):
    p006 = next(p for p in products if p.product_id == "P006")
    score, _ = type_relevance(rules_parse("running shoe"), p006)
    assert score == 1.0


# ------------------------------------------------------------ scenario C
def test_scenario_c_sneaker_first_and_alternatives_present(retriever):
    res = retriever.retrieve(rules_parse("I want a casual sneaker that won't fall apart in light rain"))
    assert ids(res)[0] == "P003"
    assert "P004" in ids(res)          # hiking shoe: the natural alternative must stay visible


# ------------------------------------------------------------ scenario D
def test_scenario_d_earbud_candidates(retriever):
    res = retriever.retrieve(rules_parse("Looking for cheap wireless earbuds, just need something basic"))
    got = ids(res)
    assert {"P014", "P015", "P017", "P024"} <= set(got[:4])
    assert "P016" in got and rank(res, "P016") > rank(res, "P014")      # over-ear: related, lower
    p026 = next(c for c in res.candidates if c.product.product_id == "P026")
    assert p026.type_relevance == pytest.approx(0.6)                    # wired earphones: related only


def test_same_category_noise_is_dropped_when_real_matches_exist(retriever):
    got = ids(retriever.retrieve(rules_parse("wireless earbuds")))
    assert "P021" not in got and "P025" not in got      # tablet, speaker: category-only padding


# ------------------------------------------------------- upsell families stay together
@pytest.mark.parametrize("msg,pair", [
    ("power bank", {"P018", "P019"}),
    ("yoga mat", {"P051", "P052"}),
    ("air purifier", {"P033", "P034"}),
    ("kettle", {"P063", "P064"}),
    ("air fryer", {"P071", "P072"}),
    ("hair dryer", {"P045", "P046"}),
])
def test_product_families_are_both_retrieved(retriever, msg, pair):
    assert pair <= set(ids(retriever.retrieve(rules_parse(msg))))


def test_backpack_query_does_not_return_a_tent_or_baskets(retriever):
    got = ids(retriever.retrieve(rules_parse("hiking backpack under $40")))
    assert {"P076", "P077"} <= set(got) and "P075" not in got and "P035" not in got


# ------------------------------------------------------------ ambiguity / no match
def test_ambiguous_query_returns_no_real_matches(retriever):
    res = retriever.retrieve(rules_parse("Something nice for my mom"))
    d = res.diagnostics
    assert d.has_type_match is False and d.n_candidates == 0 and d.top_score == 0.0


def test_unknown_product_has_no_type_match_even_with_a_loose_candidate(retriever):
    q = StructuredQuery(raw_message="I need a standing desk for my home office",
                        product_type="standing desk", category="office")
    res = retriever.retrieve(q)
    assert res.diagnostics.has_type_match is False
    assert "P100" in ids(res)                         # laptop stand: loosely related, category only
    assert res.candidates[0].type_relevance == pytest.approx(0.3)
    assert res.diagnostics.top_score < 0.5            # clearly weak: gate can say "no good match"


def test_rules_only_unknown_product_is_weak(retriever):
    res = retriever.retrieve(rules_parse("I need a standing desk for my home office"))
    assert res.diagnostics.has_type_match is False and res.diagnostics.top_score < 0.35


def test_category_only_query_returns_flat_category_results(retriever):
    res = retriever.retrieve(rules_parse("skincare for my mom"))
    d = res.diagnostics
    assert d.n_candidates >= 5 and d.n_categories_top5 == 1 and d.top5_categories[0] == "beauty"
    assert d.margin < 0.05                            # nothing stands out: flat, the gate may ask


# ------------------------------------------------------------ brand, ordering
def test_named_brand_is_boosted(retriever, brands):
    res = retriever.retrieve(rules_parse("sonicpulse earbuds", known_brands=brands))
    top3 = ids(res)[:3]
    assert set(top3) == {"P014", "P015", "P016"} or {"P014", "P015"} <= set(top3)
    assert all(c.brand_match for c in res.candidates[:2])


def test_results_are_sorted_bounded_and_deterministic(retriever):
    q = rules_parse("waterproof jacket")
    a, b = retriever.retrieve(q, k=5), retriever.retrieve(q, k=5)
    assert ids(a) == ids(b) and len(a.candidates) <= 5
    scores = [c.retrieval_score for c in a.candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 1 for s in scores)


# ------------------------------------------------------------ type_relevance unit tests
def _p(products, pid):
    return next(p for p in products if p.product_id == pid)


def test_type_relevance_levels(products):
    q = rules_parse("running shoe")
    assert type_relevance(q, _p(products, "P002"))[0] == 1.0     # same type
    assert type_relevance(q, _p(products, "P004"))[0] == 0.7     # shares 'shoe'
    assert type_relevance(rules_parse("boots"), _p(products, "P004"))[0] == 0.6   # family
    assert type_relevance(rules_parse("earbuds"), _p(products, "P016"))[0] == 0.6
    assert type_relevance(rules_parse("running shoe"), _p(products, "P027"))[0] == 0.0
    assert type_relevance(rules_parse("shoes"), _p(products, "P003"))[0] == 0.9   # generic shoe


def test_diagnostics_shape(retriever):
    d = retriever.retrieve(rules_parse("running shoe")).diagnostics
    assert d.n_candidates > 0 and 0 < d.top_score <= 1 and d.has_type_match
    assert len(d.top5_categories) == min(5, d.n_candidates)


def test_empty_catalog_is_safe():
    res = Retriever([]).retrieve(rules_parse("running shoe"))
    assert res.candidates == [] and res.diagnostics.n_candidates == 0