"""Tests for the upsell decision: every guard, the benefits, and the orchestration."""
import pytest

from app.config import get_settings
from app.data.loader import load_catalog
from app.services.attribute_extractor import rules_extract
from app.services.query_understanding import rules_parse
from app.services.ranker import Ranker
from app.services.recommender import Recommendation, recommend
from app.services.retriever import Retriever
from app.services.review_analyzer import ensure_review_evidence
from app.services.upsell import benefits, plan_recommendation, same_family, suggest_upsell

pytestmark = pytest.mark.skipif(not get_settings().catalog_csv.exists(), reason="synthetic catalog missing")

CHEAP_EARBUDS = "Looking for cheap wireless earbuds, just need something basic"


@pytest.fixture(scope="module")
def products():
    return ensure_review_evidence([rules_extract(r) for r in load_catalog()])


def make_runner(catalog):
    retriever, ranker = Retriever(catalog), Ranker(catalog)
    brands = sorted({p.brand for p in catalog})

    def run(message):
        q = rules_parse(message, known_brands=brands)
        ranked = ranker.rank(q, retriever.retrieve(q).candidates)
        rec, decision = plan_recommendation(q, ranked)
        return q, rec, decision, ranked
    return run


@pytest.fixture(scope="module")
def run(products):
    return make_runner(products)


def edited(products, pid, **changes):
    """A deep copy of the catalog with one product modified."""
    out = [p.model_copy(deep=True) for p in products]
    p = next(x for x in out if x.product_id == pid)
    for k, v in changes.items():
        setattr(p, k, v)
    return out


def manual_rec(ranked, pid):
    return Recommendation(mode="confident", primary=next(r for r in ranked if r.product.product_id == pid))


def pid(r):
    return r.product.product_id


# ------------------------------------------------------------------ scenario D
def test_scenario_d_pro_earbuds_are_a_brief_optional_mention(run):
    q, rec, dec, _ = run(CHEAP_EARBUDS)
    s = dec.suggestion
    assert pid(rec.primary) == "P014" and rec.mode == "confident"
    assert pid(s.product) == "P015" and s.style == "mention" and s.price_delta == pytest.approx(35.0)
    assert "adds active noise cancellation" in s.benefits
    assert any("30 hours" in b for b in s.benefits)               # a concrete benefit, not "a better option"


def test_the_upsell_never_replaces_the_primary_or_repeats_as_a_secondary(run):
    _, rec, dec, _ = run(CHEAP_EARBUDS)
    shown = {pid(rec.primary), *(pid(r) for r in rec.secondary)}
    assert pid(dec.suggestion.product) not in shown


def test_decoys_are_rejected_with_a_reason(run):
    _, _, dec, _ = run(CHEAP_EARBUDS)
    for decoy in ("P016", "P017", "P024", "P026"):       # over-ear, 3.4 stars, other brand, wired
        assert "family" in dec.rejected[decoy]


# ------------------------------------------------------------------ when NOT to upsell
def test_no_upsell_when_the_primary_is_already_over_budget(run):
    _, _, dec, _ = run("I need a lightweight waterproof running shoe, ideally under $80")
    assert dec.suggestion is None and "above the stated budget" in dec.skipped_reason


def test_no_upsell_when_the_primary_is_already_top_of_its_family(run):
    _, rec, dec, _ = run("I want the best air purifier for a large open-plan living room")
    assert pid(rec.primary) == "P034" and dec.suggestion is None
    assert "not a step up" in dec.rejected["P033"]


def test_no_upsell_when_there_is_no_confident_match(products, run):
    q, _, _, ranked = run("running shoe")
    rec = recommend([r.model_copy(update={"label": "poor"}) for r in ranked])
    dec = suggest_upsell(q, rec, ranked)
    assert dec.suggestion is None and "no confident match" in dec.skipped_reason


# ------------------------------------------------------------------ style
def test_a_non_cheap_ask_gets_a_suggestion_not_a_mention(run):
    q, _, _, ranked = run("wireless earbuds")
    dec = suggest_upsell(q, manual_rec(ranked, "P014"), ranked)
    assert dec.suggestion and dec.suggestion.style == "suggest" and pid(dec.suggestion.product) == "P015"


# ------------------------------------------------------------------ other families
def test_power_bank_upsell_cites_capacity_and_power(run):
    _, rec, dec, _ = run("cheap power bank")
    assert pid(rec.primary) == "P018" and pid(dec.suggestion.product) == "P019"
    assert "battery capacity: 20,000 mAh vs 10,000 mAh" in dec.suggestion.benefits


def test_yoga_mat_upsell_cites_thickness_and_grip(run):
    _, rec, dec, _ = run("cheap basic yoga mat")
    assert pid(rec.primary) == "P051" and pid(dec.suggestion.product) == "P052"
    assert "adds a non-slip surface" in dec.suggestion.benefits


# ------------------------------------------------------------------ guards
def test_price_jump_above_the_cap_is_rejected(products):
    q, rec, dec, _ = make_runner(edited(products, "P015", price_usd=400.0))(CHEAP_EARBUDS)
    assert dec.suggestion is None and "too big" in dec.rejected["P015"]


def test_a_cheap_ask_tolerates_a_smaller_gap_than_a_normal_ask(products):
    catalog = edited(products, "P019", price_usd=90.0)               # +$62 over P018
    _, _, dec_cheap, _ = make_runner(catalog)("cheap power bank")
    assert dec_cheap.suggestion is None and "cheap/basic" in dec_cheap.rejected["P019"]
    run = make_runner(catalog)
    q, _, _, ranked = run("power bank")                              # same gap, no cheap tone
    assert suggest_upsell(q, manual_rec(ranked, "P018"), ranked).suggestion is not None


def test_stated_budget_limits_how_far_an_upsell_may_stretch(run):
    _, rec, dec, _ = run("wireless earbuds under $30")
    assert pid(rec.primary) == "P014" and dec.suggestion is None
    assert "above your stated $30 budget" in dec.rejected["P015"]


@pytest.mark.parametrize("message,offered", [
    ("wireless earbuds, ideally under $40", True),     # soft: up to 1.6 x 40 = $64 >= $59.99
    ("wireless earbuds under $40", False),             # standard: up to 1.25 x 40 = $50 < $59.99
    ("wireless earbuds, no more than $40", False),     # firm: no stretch at all
])
def test_budget_firmness_sets_how_far_an_upsell_may_stretch(run, message, offered):
    q, _, _, ranked = run(message)
    dec = suggest_upsell(q, manual_rec(ranked, "P014"), ranked)
    assert (dec.suggestion is not None) is offered
    if not offered:
        assert "above your stated $40 budget" in dec.rejected["P015"]


def test_a_lower_rated_step_up_is_not_offered(products):
    _, _, dec, _ = make_runner(edited(products, "P015", average_rating=3.5))(CHEAP_EARBUDS)
    assert dec.suggestion is None and "rating" in dec.rejected["P015"]


def test_a_step_up_that_breaks_a_requirement_is_not_offered(products):
    catalog = [p.model_copy(deep=True) for p in products]
    p015 = next(p for p in catalog if p.product_id == "P015")
    p015.attributes.features.remove("wireless")
    p015.attributes.lacks.append("wireless")
    _, _, dec, _ = make_runner(catalog)(CHEAP_EARBUDS)
    assert dec.suggestion is None and "requirements" in dec.rejected["P015"]


def test_a_pricier_twin_with_no_concrete_benefit_is_not_offered(products):
    catalog = [p.model_copy(deep=True) for p in products]
    p014 = next(p for p in catalog if p.product_id == "P014")
    p015 = next(p for p in catalog if p.product_id == "P015")
    p015.description = p014.description
    p015.attributes = p014.attributes.model_copy(deep=True)
    _, _, dec, _ = make_runner(catalog)(CHEAP_EARBUDS)
    assert dec.suggestion is None and "no concrete" in dec.rejected["P015"]


def test_the_smallest_price_jump_wins(products):
    catalog = [p.model_copy(deep=True) for p in products]
    twin = next(p for p in catalog if p.product_id == "P015").model_copy(deep=True)
    twin.product_id, twin.title, twin.price_usd = "X1", "SonicPulse Wireless Earbuds Plus", 39.99
    _, _, dec, _ = make_runner(catalog + [twin])(CHEAP_EARBUDS)
    assert pid(dec.suggestion.product) == "X1" and dec.suggestion.price_delta == pytest.approx(15.0)


# ------------------------------------------------------------------ family and benefits
def test_same_family_means_same_brand_and_product_type(products):
    by = {p.product_id: p for p in products}
    assert same_family(by["P014"], by["P015"]) and same_family(by["P076"], by["P077"])
    assert same_family(by["P018"], by["P019"]) and same_family(by["P093"], by["P094"])
    assert not same_family(by["P014"], by["P016"])        # earbuds vs over-ear headphones
    assert not same_family(by["P014"], by["P017"])        # other brand


@pytest.mark.parametrize("low,high,expected", [
    ("P014", "P015", ["adds active noise cancellation", "up to 30 hours vs 18 hours"]),
    ("P018", "P019", ["20,000 mAh vs 10,000 mAh", "45 W vs 12 W"]),
    ("P022", "P023", ["adds built-in GPS"]),
    ("P051", "P052", ["8 mm vs 6 mm", "adds a non-slip surface"]),
    ("P063", "P064", ["adds adjustable temperature control", "adds a keep-warm function"]),
    ("P065", "P066", ["1200 W vs 600 W"]),
    ("P071", "P072", ["8 qt vs 4 qt", "adds two independent baskets"]),
    ("P076", "P077", ["adds an adjustable frame for heavier loads"]),
    ("P093", "P094", ["40 minutes vs 20 minutes", "adds two batteries"]),
    ("P033", "P034", ["700 sq ft vs 250 sq ft"]),
])
def test_benefits_are_concrete_and_come_from_the_data(products, low, high, expected):
    by = {p.product_id: p for p in products}
    text = " | ".join(benefits(by[low], by[high]))
    # benefits() keeps the 3 most relevant, so check that at least the key facts are present
    assert any(e in text for e in expected), text


def test_a_decline_in_a_metric_is_not_called_a_benefit(products):
    by = {p.product_id: p for p in products}
    text = " | ".join(benefits(by["P022"], by["P023"]))
    assert "14 days" not in text and "battery life" not in text      # Plus has 7 days vs 14: shorter


def test_benefits_are_capped_and_unique(products):
    by = {p.product_id: p for p in products}
    b = benefits(by["P014"], by["P015"])
    assert len(b) <= 3 and len(b) == len(set(b))


def test_no_benefit_between_a_product_and_itself(products):
    p = next(x for x in products if x.product_id == "P014")
    assert benefits(p, p) == []


# ------------------------------------------------------------------ orchestration
def test_plan_recommendation_handles_an_empty_ranking():
    rec, dec = plan_recommendation(rules_parse("running shoe"), [])
    assert rec is None and dec.suggestion is None and dec.skipped_reason


def test_the_primary_is_unchanged_by_the_upsell_step(run):
    q, rec, dec, ranked = run(CHEAP_EARBUDS)
    assert pid(rec.primary) == pid(recommend(ranked).primary)