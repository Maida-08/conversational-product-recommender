"""Tests for review evidence: snippet classification, aggregation, LLM guardrails, ranking integration."""
import pytest

from app.config import get_settings
from app.data.loader import load_catalog
from app.models.schemas import RawProduct
from app.services.attribute_extractor import rules_extract
from app.services.llm_client import LLMError
from app.services.query_understanding import rules_parse
from app.services.ranker import Ranker, RequirementResult, _apply_reviews
from app.services.retriever import Retriever
from app.services.review_analyzer import (ASPECTS, AspectSummary, classify_snippet, ensure_review_evidence,
                                          product_warnings, refine_with_llm, summarize_aspect)

needs_catalog = pytest.mark.skipif(not get_settings().catalog_csv.exists(), reason="synthetic catalog missing")


def pairs(snippet):
    return {(r["aspect"], r["polarity"]) for r in classify_snippet(snippet)}


# ------------------------------------------------------------- snippet classification
@pytest.mark.parametrize("snippet,expected", [
    ("Feet stayed bone dry through a two-hour downpour", {("water", "supports")}),
    ("Handled steady rain on a day hike without leaking", {("water", "supports")}),
    ("Mesh soaked through in the first rain, wet socks the whole run", {("water", "negative")}),
    ("The fabric got soaked in light rain and took a day to dry", {("water", "negative")}),
    ("Not great in the rain, the knit gets damp fast", {("water", "negative")}),
    ("Leaked through the seams on the first stormy run", {("water", "negative")}),
    ("Fine in a light drizzle and a comfortable fit", {("water", "supports")}),
    ("Survived sweaty runs and a rainy 10K without any problem", {("water", "supports")}),
    ("Comfortable seal with no leaking", {("water", "supports")}),
    ("Great daily trainer for dry weather", set()),                      # 'trainer' contains 'rain'
    ("Good grip on wet rocks", {("grip", "supports")}),                   # 'wet' here is not a water comment
    ("Gets slippery with sweaty hands", {("grip", "negative")}),
    ("The coating started peeling after four months", {("durability", "negative")}),
    ("Sturdy but a little heavy for long days", {("durability", "supports"), ("weight", "negative")}),
    ("Soaked through in a heavy downpour after about an hour", {("water", "negative")}),   # 'heavy' = rain
    ("Light as a slipper", {("weight", "supports")}),
    ("Base is a little light", set()),                                    # a complaint about a lamp base
    ("No scent at all and it works for my sensitive skin", {("fragrance", "supports")}),
    ("Scent is calming, not overpowering", set()),
    ("Battery only lasted about 18 minutes on max", {("battery", "negative")}),
    ("Battery easily lasts a work week", {("battery", "supports")}),
    ("Comfortable and looks great with jeans", set()),
])
def test_snippet_classification(snippet, expected):
    assert pairs(snippet) == expected


def test_water_scope_light_vs_heavy():
    assert classify_snippet("Feet stayed bone dry through a two-hour downpour")[0]["scope"] == "heavy"
    assert classify_snippet("Kept me dry in light rain and drizzle")[0]["scope"] == "light"
    assert classify_snippet("Mesh soaked through in the first rain")[0]["scope"] is None


def test_hedged_wording_is_flagged():
    weight = next(r for r in classify_snippet("Sturdy but a little heavy for long days") if r["aspect"] == "weight")
    assert weight["hedged"] is True
    dur = next(r for r in classify_snippet("Sturdy but a little heavy for long days") if r["aspect"] == "durability")
    assert dur["hedged"] is False


def test_records_keep_the_verbatim_snippet_and_valid_fields():
    snippet = "Mesh soaked through in the first rain, wet socks the whole run"
    rec = classify_snippet(snippet)[0]
    assert rec["quote"] == snippet and rec["aspect"] in ASPECTS and rec["source"] == "rules"


# ----------------------------------------------------------------------- aggregation
def _ev(aspect, polarity, scope=None, hedged=False):
    return {"aspect": aspect, "polarity": polarity, "scope": scope, "hedged": hedged, "quote": f"{aspect} {polarity}", "source": "rules"}


def test_hedged_evidence_counts_half():
    s = summarize_aspect([_ev("weight", "negative", hedged=True), _ev("weight", "negative")], "weight")
    assert s.negative == 1.5 and s.n_negative == 2


def test_heavy_rain_complaint_ignored_for_light_rain_requirement_but_not_for_waterproof():
    ev = [_ev("water", "negative", scope="heavy"), _ev("water", "supports", scope="light")]
    assert summarize_aspect(ev, "water", "waterproof").n_negative == 1
    assert summarize_aspect(ev, "water", "resistant").n_negative == 0


def test_light_rain_positive_is_weaker_proof_of_waterproofing():
    ev = [_ev("water", "supports", scope="light")]
    assert summarize_aspect(ev, "water", "waterproof").support == 0.5
    assert summarize_aspect(ev, "water", "resistant").support == 1.0


# ------------------------------------------------- combining with the description
def _req(state, sat=1.0, note="described as waterproof"):
    return RequirementResult(requirement="waterproof", kind="hard", state=state, satisfaction=sat, note=note)


def _sum(n_pos, n_neg, m=4):
    return AspectSummary(aspect="water", support=float(n_pos), negative=float(n_neg), n_support=n_pos,
                         n_negative=n_neg, n_reviews=m,
                         quotes_support=["pos"] * n_pos, quotes_negative=["neg"] * n_neg)


def test_two_agreeing_negatives_against_a_claim_is_a_conflict():
    r = _apply_reviews(_req("met"), _sum(0, 2))
    assert r.state == "conflict" and r.satisfaction == 0.3 and "2 of 4 reviewers" in r.note


def test_a_lone_negative_never_overturns_a_description_claim():
    r = _apply_reviews(_req("met"), _sum(1, 1))
    assert r.state == "met" and r.satisfaction == 1.0 and r.review_note and "1 of 4" in r.review_note


def test_negatives_that_do_not_outnumber_supporters_are_not_a_conflict():
    assert _apply_reviews(_req("met"), _sum(2, 2)).state == "met"


def test_silent_description_with_two_negatives_is_unmet():
    r = _apply_reviews(_req("unknown", 0.4, "not stated"), _sum(0, 2))
    assert r.state == "unmet" and r.satisfaction == 0.1 and r.source == "reviews"


def test_silent_description_with_one_negative_is_partial_and_unverified():
    r = _apply_reviews(_req("unknown", 0.4, "not stated"), _sum(0, 1))
    assert r.state == "partial" and r.satisfaction == 0.25 and "unverified" in r.note


def test_silent_description_review_support_scales_with_agreement():
    one = _apply_reviews(_req("unknown", 0.4, "not stated"), _sum(1, 0))
    two = _apply_reviews(_req("unknown", 0.4, "not stated"), _sum(2, 0))
    assert (one.state, one.satisfaction) == ("partial", 0.6) and (two.state, two.satisfaction) == ("met", 0.8)
    assert "not stated in the description" in two.note


def test_mixed_reviews_leave_unknown_unknown():
    r = _apply_reviews(_req("unknown", 0.4, "not stated"), _sum(1, 1))
    assert r.state == "unknown" and "mixed" in r.note


def test_explicit_description_negative_is_not_overridden_by_reviews():
    assert _apply_reviews(_req("unmet", 0.0, "says not waterproof"), _sum(3, 0)).state == "unmet"


def test_no_evidence_changes_nothing():
    r = _req("met")
    assert _apply_reviews(r, _sum(0, 0)) is r


# ------------------------------------------------------------- LLM refinement guards
class FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload, self.error = payload, error

    def complete_json(self, system, user, max_tokens=3500):
        if self.error:
            raise self.error
        return self.payload, {"model": "fake", "total_tokens": 10}


def _product(snippets):
    raw = RawProduct(product_id="T1", category="footwear", brand="Acme", title="Acme Shoe",
                     description="A shoe.", price_usd=10.0, average_rating=4.0, num_reviews=5,
                     review_snippets=snippets)
    p = rules_extract(raw)
    p.review_evidence = []
    return p


def test_llm_can_add_nuance_the_rules_miss():
    p = _product(["It did not keep my feet dry at all"])
    assert classify_snippet(p.review_snippets[0]) == []
    payload = {"reviews": [{"i": 1, "aspects": [{"aspect": "water", "polarity": "negative", "hedged": False, "scope": None}]}]}
    assert refine_with_llm([p], FakeClient(payload), log=lambda *_: None) == 1
    assert [(r["aspect"], r["polarity"], r["source"]) for r in p.review_evidence] == [("water", "negative", "llm")]


def test_llm_cannot_invent_evidence_without_a_keyword_in_the_review():
    p = _product(["Nice color"])
    payload = {"reviews": [{"i": 1, "aspects": [{"aspect": "water", "polarity": "negative"}]}]}
    refine_with_llm([p], FakeClient(payload), log=lambda *_: None)
    assert p.review_evidence == []


def test_llm_invalid_aspect_or_polarity_is_dropped():
    p = _product(["Soaked in the rain"])
    payload = {"reviews": [{"i": 1, "aspects": [{"aspect": "style", "polarity": "negative"},
                                                {"aspect": "water", "polarity": "great"}]}]}
    refine_with_llm([p], FakeClient(payload), log=lambda *_: None)
    assert p.review_evidence == []


def test_llm_failure_keeps_rule_evidence():
    p = _product(["Mesh soaked through in the first rain"])
    ensure_review_evidence([p])
    before = list(p.review_evidence)
    assert refine_with_llm([p], FakeClient(error=LLMError("rate limited")), log=lambda *_: None) == 0
    assert p.review_evidence == before and before[0]["source"] == "rules"


def test_ensure_review_evidence_fills_only_empty_and_is_idempotent():
    p = _product(["Mesh soaked through in the first rain"])
    ensure_review_evidence([p])
    first = list(p.review_evidence)
    p.review_evidence[0]["source"] = "llm"          # pretend it was refined
    ensure_review_evidence([p])
    assert p.review_evidence[0]["source"] == "llm" and len(p.review_evidence) == len(first)


# ---------------------------------------------------------------- on the catalog
@pytest.fixture(scope="module")
def products():
    return ensure_review_evidence([rules_extract(r) for r in load_catalog()])


@pytest.fixture(scope="module")
def by_id(products):
    return {p.product_id: p for p in products}


@needs_catalog
def test_planted_products_have_the_expected_evidence(by_id):
    def n(pid, aspect, pol):
        return sum(1 for r in by_id[pid].review_evidence if r["aspect"] == aspect and r["polarity"] == pol)
    assert n("P003", "water", "negative") == 2 and n("P003", "water", "supports") == 0
    assert n("P006", "water", "negative") == 2 and n("P006", "water", "supports") == 1
    assert n("P007", "water", "negative") == 1 and n("P007", "water", "supports") == 1
    assert n("P024", "water", "supports") == 1 and n("P049", "fragrance", "supports") == 1
    assert n("P068", "durability", "negative") == 2 and n("P037", "battery", "negative") == 2
    assert n("P002", "water", "supports") == 0          # 'dry weather' must not count


@needs_catalog
def test_every_quote_is_a_verbatim_review_of_that_product(products):
    for p in products:
        for r in p.review_evidence:
            assert r["quote"] in p.review_snippets and r["aspect"] in ASPECTS


@needs_catalog
def test_product_level_warnings_are_readable(by_id):
    assert "2 of 3 reviewers report it let water in" in by_id["P003"].review_warnings
    assert by_id["P001"].review_warnings == []


# ------------------------------------------------------- ranking with review evidence
@pytest.fixture(scope="module")
def run(products):
    retriever, ranker = Retriever(products), Ranker(products)

    def _run(message):
        q = rules_parse(message)
        return {r.product.product_id: r for r in ranker.rank(q, retriever.retrieve(q).candidates)}
    return _run


def req(r, name):
    return next(x for x in r.requirements if x.requirement == name)


def by_snippets(ranked):
    return ranked.product.review_snippets


@needs_catalog
def test_scenario_a_contradicted_waterproof_claim_is_a_flagged_conflict(run):
    r = run("I need a lightweight waterproof running shoe, ideally under $80")
    assert r["P001"].rank < r["P002"].rank < r["P006"].rank
    p006 = r["P006"]
    assert p006.label == "conflict" and req(p006, "waterproof").state == "conflict"
    assert "reviewers" in req(p006, "waterproof").note
    assert all(q in by_snippets(p006) for q in req(p006, "waterproof").review_quotes)


@needs_catalog
def test_scenario_a_secondary_carries_an_unverified_review_warning(run):
    p002 = req(run("I need a lightweight waterproof running shoe, ideally under $80")["P002"], "waterproof")
    assert p002.state == "partial" and p002.source == "reviews" and "unverified" in p002.note


@needs_catalog
def test_scenario_c_review_only_signal_downranks_the_sneaker(run):
    r = run("I want a casual sneaker that won't fall apart in light rain")
    p003 = r["P003"]
    assert req(p003, "water_resistant").state == "unmet" and req(p003, "water_resistant").source == "reviews"
    assert p003.label == "poor" and any("discounted" in p for p in p003.penalties)
    assert r["P007"].rank < p003.rank and r["P004"].rank < p003.rank
    assert req(r["P004"], "water_resistant").state == "met"


@needs_catalog
def test_lone_leak_review_does_not_overturn_the_waterproof_claim(run):
    p007 = run("waterproof hiking boot")["P007"]
    assert req(p007, "waterproof").state == "met" and req(p007, "waterproof").review_note
    assert p007.label not in ("conflict", "poor")
    assert any("1 of 4 reviewers" in t for t in p007.tradeoffs)


@needs_catalog
def test_review_only_positive_is_partial_and_communicated_as_uncertain(run):
    p024 = req(run("wireless earbuds that can handle sweat and rain")["P024"], "water_resistant")
    assert p024.state == "partial" and p024.source == "reviews" and "not stated in the description" in p024.note


@needs_catalog
def test_heavy_downpour_complaint_only_matters_for_a_waterproof_ask(run):
    light = req(run("rain jacket for light rain")["P080"], "water_resistant")
    full = req(run("waterproof rain jacket")["P080"], "waterproof")
    assert light.review_note is None
    assert full.review_note and "1 of 3 reviewers" in full.review_note


@needs_catalog
def test_unrelated_complaints_do_not_affect_ranking(run):
    # P051's grip complaint must not matter when the shopper never asked about grip
    plain = run("yoga mat")["P051"]
    assert all(r.requirement != "non_slip" for r in plain.requirements)
    asked = req(run("non-slip yoga mat")["P051"], "non_slip")
    assert asked.state == "partial" and "unverified" in asked.note


@needs_catalog
def test_fragrance_free_supported_only_by_a_review(run):
    p049 = req(run("a fragrance-free cleanser")["P049"], "fragrance_free")
    assert p049.state == "partial" and p049.source == "reviews"