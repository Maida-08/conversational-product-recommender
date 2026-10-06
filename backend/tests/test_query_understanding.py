"""Tests for query understanding: rules parser, budget parsing, and LLM-merge guardrails."""
import pytest

from app.models.schemas import StructuredQuery
from app.services import query_understanding as qu
from app.services.llm_client import LLMError


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload, self.error = payload, error

    def complete_json(self, system, user, max_tokens=3500):
        if self.error:
            raise self.error
        return self.payload, {"model": "fake", "total_tokens": 50}


# ------------------------------------------------------------------- budget
@pytest.mark.parametrize("text,mx,mn,firm", [
    ("a shoe under $45", 45, None, "standard"),
    ("something less than 30 dollars", 30, None, "standard"),
    ("headphones between $20 and $40", 40, 20, "standard"),
    ("a speaker around $50", 50, None, "soft"),
    ("I can't spend more than $60", 60, None, "firm"),
    ("a laptop under $1,200", 1200, None, "standard"),
    ("a jacket over $100", None, 100, "standard"),
])
def test_budget_forms(text, mx, mn, firm):
    b = qu.parse_budget(text)
    assert (b.max, b.min, b.firmness) == (mx, mn, firm)


@pytest.mark.parametrize("text", [
    "boots under 3 lb", "a kettle that boils in under 4 minutes",
    "earbuds with at least 20 reviews", "a 40L pack", "something light",
])
def test_non_money_numbers_are_not_budgets(text):
    b = qu.parse_budget(text)
    assert b.max is None and b.min is None


def test_ideally_softens_the_budget():
    assert qu.parse_budget("a running shoe, ideally under $80").firmness == "soft"


# --------------------------------------------------- scenario A: specific query
def test_scenario_a_specific_query():
    q = qu.rules_parse("I need a lightweight waterproof running shoe, ideally under $80.")
    assert q.product_type == "running shoe" and q.category == "footwear"
    assert q.budget.max == 80 and q.budget.firmness == "soft"
    assert {"waterproof", "lightweight"} <= set(q.hard_requirements)
    assert q.soft_preferences == []
    assert "no_product_type" not in q.vagueness_signals


# ------------------------------------------------- hard vs soft requirements
def test_ideally_makes_requirement_soft_but_only_in_its_clause():
    q = qu.rules_parse("I need a waterproof jacket, ideally lightweight")
    assert "waterproof" in q.hard_requirements
    assert "lightweight" in q.soft_preferences and "lightweight" not in q.hard_requirements


def test_prefer_marks_soft():
    q = qu.rules_parse("I'd prefer something quiet, a blender")
    assert "quiet" in q.soft_preferences


# ------------------------------------------ scenario C: review-driven rain query
def test_light_rain_is_water_resistance_not_lightweight():
    q = qu.rules_parse("I want a casual sneaker that won't fall apart in light rain.")
    assert q.product_type == "sneaker"
    assert "casual" in q.intended_use
    assert "water_resistant" in q.hard_requirements and "durable" in q.hard_requirements
    assert "lightweight" not in q.hard_requirements + q.soft_preferences


def test_rain_jacket_word_does_not_create_a_rain_requirement():
    q = qu.rules_parse("a rain jacket")
    assert q.product_type == "rain jacket" and "water_resistant" not in q.hard_requirements


# ------------------------------------------------------ scenario D: tone signals
def test_scenario_d_cheap_and_basic():
    q = qu.rules_parse("Looking for cheap wireless earbuds, just need something basic.")
    assert q.product_type == "earbuds" and "wireless" in q.hard_requirements
    assert {"cheap", "basic"} <= set(q.tone_signals)
    assert q.budget.max is None


def test_best_and_large_room():
    q = qu.rules_parse("I want the best air purifier for a large open-plan living room")
    assert q.product_type == "air purifier" and "best" in q.tone_signals
    assert "large_room" in q.hard_requirements


# ----------------------------------------------------- scenario B: ambiguity signals
def test_scenario_b_ambiguous_query_has_vagueness_signals():
    q = qu.rules_parse("Something nice for my mom.")
    assert q.product_type is None and q.category is None
    assert q.recipient == "mom"
    sig = set(q.vagueness_signals)
    assert {"no_product_type", "no_concrete_attributes", "recipient_without_interest"} <= sig
    assert "vague_word:nice" in sig and "vague_word:something" in sig


def test_recipient_plus_interest_is_less_vague():
    q = qu.rules_parse("A birthday gift for my wife who loves yoga")
    assert q.is_gift and q.recipient == "wife"
    assert "yoga" in q.intended_use and "gift" not in q.intended_use
    assert "recipient_without_interest" not in q.vagueness_signals


def test_category_word_counts_as_a_category():
    q = qu.rules_parse("skincare for my mom")
    assert q.category == "beauty" and "no_product_type" not in q.vagueness_signals


# ------------------------------------------------------- product type edge cases
def test_fragrance_free_cleanser_is_not_a_perfume():
    q = qu.rules_parse("a fragrance-free cleanser")
    assert q.product_type == "cleanser" and "fragrance_free" in q.hard_requirements


def test_unknown_product_has_no_type_in_rules():
    q = qu.rules_parse("I need a standing desk for my home office.")
    assert q.product_type is None and q.category is None


def test_brand_detection_uses_known_brands():
    q = qu.rules_parse("sonicpulse earbuds", known_brands=("SonicPulse", "Orbit"))
    assert q.brand == "SonicPulse"


# -------------------------------------------------------------- LLM merge guards
def test_llm_can_fill_product_type_for_unknown_item():
    payload = {"product_type": "standing desk", "category": "office"}
    q = qu.understand_query("I need a standing desk for my home office.", client=FakeClient(payload))
    assert q.product_type == "standing desk" and q.category == "office" and q.parser == "rules+llm"


def test_llm_budget_not_in_message_is_dropped():
    q = qu.understand_query("a nice blender", client=FakeClient({"budget": {"max": 50}}))
    assert q.budget.max is None
    assert any("dropped LLM budget" in n for n in q.notes)


def test_rules_budget_beats_llm_budget():
    q = qu.understand_query("a shoe under $45", client=FakeClient({"budget": {"max": 45}, "product_type": "shoe"}))
    assert q.budget.max == 45


def test_llm_invented_requirement_is_dropped():
    payload = {"product_type": "wallet", "hard_requirements": ["vegan leather", "rfid blocking"]}
    q = qu.understand_query("I need a wallet", client=FakeClient(payload))
    assert q.hard_requirements == []
    assert sum(n.startswith("dropped LLM requirement") for n in q.notes) == 2


def test_llm_supported_new_requirement_is_added():
    payload = {"hard_requirements": ["keeps toes toasty"], "product_type": "boot"}
    q = qu.understand_query("I want a boot that keeps toes toasty", client=FakeClient(payload))
    assert any("toes" in r for r in q.hard_requirements)


def test_llm_category_outside_catalog_is_dropped():
    q = qu.understand_query("I need a standing desk", client=FakeClient({"product_type": "standing desk", "category": "furniture"}))
    assert q.category is None and any("not a catalog category" in n for n in q.notes)


def test_llm_failure_falls_back_to_rules():
    q = qu.understand_query("a lightweight running shoe under $80", client=FakeClient(error=LLMError("rate limited")))
    assert q.parser == "rules" and q.budget.max == 80
    assert any("LLM unavailable" in n for n in q.notes)


def test_output_is_a_valid_schema():
    q = qu.rules_parse("anything")
    StructuredQuery.model_validate(q.model_dump())




def test_llm_intended_use_that_restates_product_type_is_ignored():
    payload = {"product_type": "standing desk", "intended_use": ["standing", "desk", "home office work"]}
    q = qu.understand_query("I need a standing desk for my home office", client=FakeClient(payload))
    assert "desk" not in q.intended_use and "standing" not in q.intended_use


def test_curly_apostrophe_budget_and_requirement():
    b = qu.parse_budget("I can’t spend more than $60")
    assert (b.max, b.min, b.firmness) == (60, None, "firm")
    q = qu.rules_parse("a sneaker that won’t fall apart")
    assert "durable" in q.hard_requirements


def test_more_than_inside_a_cap_phrase_is_not_a_minimum():
    b = qu.parse_budget("I don't want to spend more than $75")
    assert b.max == 75 and b.min is None


def test_cap_and_floor_can_coexist():
    b = qu.parse_budget("something over $20 but under $50")
    assert (b.min, b.max) == (20, 50)    
def test_llm_tone_words_are_not_requirements():
    rq = qu.rules_parse("Looking for cheap wireless earbuds, just need something basic")
    llm = qu._LLMQuery(
        hard_requirements=["wireless", "cheap", "basic"],
        soft_preferences=[],
    )

    q = qu._merge(rq, llm, qu.KNOWN_CATEGORIES, [])

    assert "wireless" in q.hard_requirements
    assert "cheap" not in q.hard_requirements
    assert "basic" not in q.hard_requirements
    assert "cheap" in q.tone_signals
    assert "basic" in q.tone_signals


def test_llm_budget_restatement_is_not_requirement():
    rq = qu.rules_parse("I need a lightweight waterproof running shoe, ideally under $80")
    llm = qu._LLMQuery(
        hard_requirements=["lightweight", "waterproof"],
        soft_preferences=["ideally under $80"],
    )

    q = qu._merge(rq, llm, qu.KNOWN_CATEGORIES, [])

    assert "lightweight" in q.hard_requirements
    assert "waterproof" in q.hard_requirements
    assert "ideally_under_80" not in q.soft_preferences
    assert q.budget.max == 80.0


def test_llm_cannot_turn_price_tone_or_the_product_type_into_requirements():
    # real Groq output put "cheap" and "ideally under $80" into the requirements list
    payload = {"product_type": "wireless earbuds", "hard_requirements": ["cheap", "wireless earbuds", "wireless"],
               "soft_preferences": ["basic"]}
    q = qu.understand_query("Looking for cheap wireless earbuds, just need something basic", client=FakeClient(payload))
    assert q.hard_requirements == ["wireless"] and q.soft_preferences == []
    assert sum("not a requirement" in n for n in q.notes) == 2           # 'cheap' and 'basic'


def test_a_budget_phrase_or_the_product_type_is_not_a_requirement():
    payload = {"hard_requirements": ["lightweight", "running shoe"], "soft_preferences": ["ideally under $80"]}
    q = qu.understand_query("I need a lightweight waterproof running shoe, ideally under $80", client=FakeClient(payload))
    assert q.hard_requirements == ["waterproof", "lightweight"] and q.soft_preferences == []
    assert q.budget.max == 80 and q.budget.firmness == "soft"             # the budget lives in its own field


def test_genuine_attributes_with_numbers_are_still_accepted():
    payload = {"hard_requirements": ["40L capacity"]}
    q = qu.understand_query("a hiking backpack with 40L capacity", client=FakeClient(payload))
    assert any("40l" in r for r in q.hard_requirements)
