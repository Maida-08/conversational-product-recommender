"""Tests for response generation: facts, template, grounding guard, and the LLM fallback path."""
import re

import pytest

from app.config import get_settings
from app.data.loader import load_catalog
from app.services.ambiguity import gate, match_check
from app.services.attribute_extractor import rules_extract
from app.services.llm_client import LLMError
from app.services.query_understanding import rules_parse
from app.services.ranker import Ranker
from app.services.recommender import Recommendation
from app.services.response_generator import (MAX_WORDS, build_clarification_facts, build_no_match_facts,
                                             build_recommendation_facts, check_grounding, generate, llm_view,
                                             money, render, requirement_clause)
from app.services.retriever import Retriever
from app.services.review_analyzer import ensure_review_evidence
from app.services.upsell import UpsellDecision, plan_recommendation

pytestmark = pytest.mark.skipif(not get_settings().catalog_csv.exists(), reason="synthetic catalog missing")

SCEN_A = "I need a lightweight waterproof running shoe, ideally under $80"
SCEN_C = "I want a casual sneaker that won't fall apart in light rain"
SCEN_D = "Looking for cheap wireless earbuds, just need something basic"


@pytest.fixture(scope="module")
def products():
    return ensure_review_evidence([rules_extract(r) for r in load_catalog()])


@pytest.fixture(scope="module")
def facts_for(products):
    retriever, ranker = Retriever(products), Ranker(products)
    brands = sorted({p.brand for p in products})

    def _go(message):
        q = rules_parse(message, known_brands=brands)
        g = gate(q, products)
        if g.decision == "clarify":
            return build_clarification_facts(q, g)
        q = g.query
        retrieval = retriever.retrieve(q)
        if match_check(q, retrieval).decision == "no_match":
            return build_no_match_facts(q, [c.product.title for c in retrieval.candidates[:2]])
        rec, up = plan_recommendation(q, ranker.rank(q, retrieval.candidates))
        if rec is None:
            return build_no_match_facts(
                q,
                [c.product.title for c in retrieval.candidates[:2]],
                reason="constraints",
            )
        return build_recommendation_facts(q, rec, up)
    return _go


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload, self.error = payload, error

    def complete_json(self, system, user, max_tokens=3500):
        if self.error:
            raise self.error
        return self.payload, {"model": "fake", "total_tokens": 10}


# --------------------------------------------------------------------------- money
@pytest.mark.parametrize("value,text", [(24.99, "$24.99"), (35.0, "$35"), (89.5, "$89.50"), (1299.0, "$1,299")])
def test_money_formatting(value, text):
    assert money(value) == text


# ------------------------------------------------------------ scenario A: tradeoffs are stated
def test_scenario_a_reply_states_the_price_tradeoff_and_the_review_caution(facts_for):
    text = render(facts_for(SCEN_A))
    assert "Aquaridge StormShield Waterproof Trail Running Shoe ($89.50)" in text
    assert "$9.50 over your $80 budget" in text                      # stated, not hidden
    assert "TrailFox Featherlite Running Shoe ($64.99)" in text       # the in-budget alternative
    assert "Tempo AquaRun Waterproof Lightweight Runner" in text and "2 of 3 reviewers" in text
    assert text.index("StormShield") < text.index("TrailFox") < text.index("AquaRun")


# ------------------------------------------------------------ scenario C: the literal match first
def test_scenario_c_leads_with_the_literal_match_and_its_problem_then_the_alternative(facts_for):
    text = render(facts_for(SCEN_C))
    assert text.index("UrbanStep Canvas Casual Sneaker") < text.index("Northpath Leather Hiking Shoe")
    assert "closest match for a sneaker" in text and "dealbreaker" in text and "reviewers" in text
    assert "My top pick" not in text                                  # not presented as a confident pick


# ------------------------------------------------------------ scenario D: the upsell is a mention
def test_scenario_d_upsell_is_one_brief_optional_sentence_after_the_pick(facts_for):
    text = render(facts_for(SCEN_D))
    assert text.startswith("I'd go with the SonicPulse Wireless Earbuds Lite ($24.99)")
    sentence = next(s for s in re.split(r"(?<=\.)\s", text) if "Earbuds Pro" in s)
    assert sentence.startswith("If you ever want") and "noise cancellation" in sentence and "$35 more" in sentence
    assert text.index("Earbuds Lite") < text.index("Earbuds Pro")
    assert text.count("Earbuds Pro") == 1                             # mentioned once, not pushed


def test_a_secondary_that_adds_nothing_is_left_out_of_the_text(facts_for):
    text = render(facts_for(SCEN_D))
    assert "AeroPods" not in text                                     # only "is wireless", same as the primary


def test_a_cheaper_secondary_is_offered_as_spending_less(facts_for):
    text = render(facts_for("a gift for my wife who loves yoga"))
    assert "If you'd rather spend less" in text and "FlexCore Yoga Mat 6mm ($24.99)" in text


# ------------------------------------------------------------ scenario B and no match
def test_scenario_b_asks_exactly_one_question_and_shows_no_products(facts_for):
    text = render(facts_for("Something nice for my mom"))
    assert text.count("?") == 1 and "$" not in text and "she" in text
    assert "a blanket" in text or "a gift set" in text


def test_no_match_is_honest_and_does_not_recommend(products):
    from app.models.schemas import StructuredQuery
    q = StructuredQuery(raw_message="standing desk", product_type="standing desk", category="office")
    text = render(build_no_match_facts(q, ["DeskMate Adjustable Laptop Stand"]))
    assert "standing desk" in text and "not really the same thing" in text and "$" not in text
    assert text.count("?") == 1


def test_closest_available_says_so_plainly(facts_for, products):
    q = rules_parse(SCEN_D)
    ranked = Ranker(products).rank(q, Retriever(products).retrieve(q).candidates)
    rec = Recommendation(mode="closest_available", primary=ranked[0])
    text = render(build_recommendation_facts(q, rec, UpsellDecision()))
    assert text.startswith("I couldn't find a great match")


# ------------------------------------------------------------ the template is always safe
BANK = [SCEN_A, SCEN_C, SCEN_D, "Something nice for my mom", "skincare for my mom", "something for hiking",
        "a gift for my wife who loves yoga", "cheap power bank", "cheap basic yoga mat", "waterproof hiking boot",
        "wireless earbuds that can handle sweat and rain", "I want the best air purifier for a large open-plan living room",
        "a fragrance-free cleanser", "waterproof rain jacket", "wireless earbuds under $30", "kettle", "air fryer",
        "I need a standing desk for my home office", "hiking backpack under $40", "pet supplies", "running shoe"]


@pytest.mark.parametrize("message", BANK)
def test_every_template_reply_passes_its_own_grounding_guard(facts_for, products, message):
    facts = facts_for(message)
    text = render(facts)
    assert check_grounding(text, facts, products) == [], text
    assert len(text.split()) <= MAX_WORDS


# ---------------------------------------------------------------- the guard catches bad LLM text
@pytest.fixture(scope="module")
def a_facts(facts_for):
    return facts_for(SCEN_A)


@pytest.fixture(scope="module")
def d_facts(facts_for):
    return facts_for(SCEN_D)


def violations(text, facts, products):
    return " | ".join(check_grounding(text, facts, products))


GOOD_A = ("I'd go with the Aquaridge StormShield Waterproof Trail Running Shoe ($89.50). It is fully waterproof, "
          "though it is $9.50 over your $80 budget and isn't the lightest at 289 g per shoe. The TrailFox "
          "Featherlite Running Shoe ($64.99) fits the budget. Heads-up on the Tempo AquaRun Waterproof "
          "Lightweight Runner ($74.99): reviewers say it leaks.")


def test_a_good_reply_passes(a_facts, products):
    assert check_grounding(GOOD_A, a_facts, products) == []


def test_invented_price_is_caught(a_facts, products):
    assert "price $79.99 is not in the facts" in violations(GOOD_A.replace("$64.99", "$79.99"), a_facts, products)


def test_a_hidden_price_overshoot_is_caught(a_facts, products):
    text = GOOD_A.replace("though it is $9.50 over your $80 budget and ", "though it ")
    assert "hides how far over budget" in violations(text, a_facts, products)


def test_a_dropped_caution_is_caught(a_facts, products):
    text = GOOD_A.replace(" Heads-up on the Tempo AquaRun Waterproof Lightweight Runner ($74.99): reviewers say it leaks.", "")
    assert "drops the caution" in violations(text, a_facts, products)


def test_an_omitted_review_caveat_is_caught(a_facts, products):
    text = GOOD_A.replace("reviewers say it leaks", "it is a bit pricey")
    assert "omits a required caveat" in violations(text, a_facts, products)


def test_missing_primary_name_is_caught(a_facts, products):
    assert "does not name the primary pick" in violations("The TrailFox Featherlite Running Shoe ($64.99) is nice.", a_facts, products)


def test_a_product_that_is_not_part_of_the_answer_is_caught(a_facts, products):
    text = GOOD_A + " The SonicPulse Wireless Earbuds Pro would also be great."
    assert "not part of the answer" in violations(text, a_facts, products)


def test_a_brand_that_is_not_part_of_the_answer_is_caught(a_facts, products):
    assert "brand that is not part of the answer (SonicPulse)" in violations(GOOD_A + " SonicPulse is great too.", a_facts, products)


@pytest.mark.parametrize("extra,expect", [
    (" (ref P014)", "internal product id"),
    (" Its score is high.", "internal jargon"),
    ("\n- a bullet", "markdown"),
    (" " + "word " * 140, "too long"),
])
def test_formatting_and_jargon_violations(a_facts, products, extra, expect):
    assert expect in violations(GOOD_A + extra, a_facts, products)


def test_upsell_before_the_primary_is_caught(d_facts, products):
    text = ("If you want noise cancelling, the SonicPulse Wireless Earbuds Pro is $35 more. "
            "I'd go with the SonicPulse Wireless Earbuds Lite ($24.99).")
    assert "upsell before the primary" in violations(text, d_facts, products)


def test_a_dropped_upsell_is_caught(d_facts, products):
    text = "I'd go with the SonicPulse Wireless Earbuds Lite ($24.99). It is wireless."
    assert "drops the upsell" in violations(text, d_facts, products)


def test_clarification_rules(facts_for, products):
    facts = facts_for("Something nice for my mom")
    ok = "Happy to help! What is she into? I'm thinking a blanket, a gift set, or a pour-over coffee set."
    assert check_grounding(ok, facts, products) == []
    assert "exactly one question" in violations(ok.replace("!", "?"), facts, products)
    assert "price" in violations(ok + " Blankets start at $20.", facts, products)
    assert "does not offer the catalog options" in violations("What are you looking for?", facts, products)


def test_no_match_must_not_show_prices(products):
    from app.models.schemas import StructuredQuery
    q = StructuredQuery(raw_message="standing desk", product_type="standing desk", category="office")
    facts = build_no_match_facts(q, ["DeskMate Adjustable Laptop Stand"])
    assert "price" in violations("I have no standing desk, but the DeskMate Adjustable Laptop Stand is $29.99.", facts, products)


# ---------------------------------------------------------------- the LLM path
def test_good_llm_wording_is_used(a_facts, products):
    out = generate(a_facts, FakeClient({"message": GOOD_A}), products)
    assert out.method == "llm" and out.text == GOOD_A and out.guard_failures == []


def test_bad_llm_wording_falls_back_to_the_template_and_records_why(a_facts, products):
    bad = GOOD_A.replace("$64.99", "$79.99")
    out = generate(a_facts, FakeClient({"message": bad}), products)
    assert out.method == "template" and out.text == render(a_facts)
    assert any("$79.99" in f for f in out.guard_failures)


def test_llm_failure_or_garbage_falls_back(a_facts, products):
    assert generate(a_facts, FakeClient(error=LLMError("rate limited")), products).method == "template"
    out = generate(a_facts, FakeClient({"unexpected": "shape"}), products)
    assert out.method == "template" and out.guard_failures == ["empty reply"]


def test_no_client_means_the_template(a_facts, products):
    out = generate(a_facts, None, products)
    assert out.method == "template" and out.text == render(a_facts)


def test_the_llm_never_sees_internal_fields(a_facts):
    view = llm_view(a_facts)
    blob = str(view)
    for hidden in ("critical", "over_budget_by", "'label'"):
        assert hidden not in blob
    assert view["primary"]["title"] and view["primary"]["minuses"]


# ---------------------------------------------------------------- requirement clauses read like English
def _clause(products_by_id, pid, tag, query):
    from app.services.ranker import CatalogStats, evaluate_requirement
    return requirement_clause(evaluate_requirement(tag, products_by_id[pid], CatalogStats()))


def test_clauses_read_as_predicates(products):
    by = {p.product_id: p for p in products}
    assert _clause(by, "P001", "waterproof", None)[:2] == ("plus", "is fully waterproof")
    assert _clause(by, "P002", "waterproof", None)[:2] == ("minus", "doesn't say whether it's waterproof")
    assert _clause(by, "P005", "waterproof", None)[:2] == ("minus", "isn't waterproof")
    assert _clause(by, "P004", "lightweight", None)[1].startswith("is on the heavy side (425 g")
    assert _clause(by, "P026", "wireless", None)[:2] == ("minus", "is wired, not wireless")


def test_an_omitted_lightness_tradeoff_is_caught(a_facts, products):
    text = GOOD_A.replace(" and isn't the lightest at 289 g per shoe", "")
    assert "omits a required caveat" in violations(text, a_facts, products)


def test_a_title_word_does_not_count_as_a_caveat(a_facts, products):
    # "Waterproof" appears inside the caution's TITLE; it must not satisfy "mention the waterproof caveat"
    text = GOOD_A.replace("reviewers say it leaks", "it is a bit pricey")
    assert "omits a required caveat" in violations(text, a_facts, products)


def test_a_near_miss_price_is_not_accepted_for_the_budget(a_facts, products):
    # $79.99 is one cent from the $80 budget; it is still an invented price
    assert "price $79.99 is not in the facts" in violations(GOOD_A.replace("$64.99", "$79.99"), a_facts, products)


def test_typographic_punctuation_is_normalised_before_the_guard(a_facts, products):
    fancy = GOOD_A.replace("It is fully waterproof,", "It is fully waterproof \u2014 really,").replace("isn't", "isn\u2019t")
    out = generate(a_facts, FakeClient({"message": fancy}), products)
    assert out.method == "llm" and "\u2014" not in out.text and "\u2019" not in out.text


def test_commentary_about_its_own_sources_is_rejected(a_facts, products):
    text = GOOD_A + " However, the facts do not explicitly state that."
    assert "comments on its own sources" in violations(text, a_facts, products)
