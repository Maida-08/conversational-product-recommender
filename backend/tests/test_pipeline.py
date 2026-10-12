"""Tests for the pipeline: the single entry point behind the API (and later Twilio)."""
import json

import pytest

from app.config import get_settings
from app.data.loader import load_catalog
from app.models.schemas import ChatMessage
from app.services import pipeline
from app.services.attribute_extractor import rules_extract
from app.services.llm_client import LLMError
from app.services.pipeline import EXAMPLE_QUERIES, Engine

pytestmark = pytest.mark.skipif(not get_settings().catalog_csv.exists(), reason="synthetic catalog missing")

SCEN_A = "I need a lightweight waterproof running shoe, ideally under $80"
SCEN_C = "I want a casual sneaker that won't fall apart in light rain"
SCEN_D = "Looking for cheap wireless earbuds, just need something basic"


@pytest.fixture(scope="module")
def catalog():
    return [rules_extract(r) for r in load_catalog()]


@pytest.fixture(scope="module")
def engine(catalog):
    return Engine(catalog, client=None, data_source="test")          # no LLM: no test can ever call Groq


def roles(resp):
    return [(c.role, c.product_id) for c in resp.recommendations]


# ------------------------------------------------------------------ scenario A
def test_scenario_a_cards_and_stated_tradeoffs(engine):
    r = engine.respond(SCEN_A)
    assert r.type == "recommendation" and roles(r) == [("primary", "P001"), ("secondary", "P002"), ("caution", "P006")]
    primary = r.recommendations[0]
    assert "$9.50 over your $80 budget" in primary.warnings and "Fully waterproof" in primary.reasons
    assert "$9.50 over your $80 budget" in r.assistant_message
    assert r.upsell is None and "above the stated budget" in r.details.upsell_skipped_reason
    assert r.recommendations[2].match_label == "conflict" and any("reviewers" in w for w in r.recommendations[2].warnings)


# ------------------------------------------------------------------ scenario B
def test_scenario_b_clarification_has_no_products(engine):
    r = engine.respond("Something nice for my mom")
    assert r.type == "clarification" and r.recommendations == [] and r.upsell is None
    assert r.clarification.options and r.clarification.question == r.assistant_message
    assert r.assistant_message.count("?") == 1
    assert r.details.specificity < 0.3 and r.details.parsed_query.recipient == "mom"


# ------------------------------------------------------------------ scenario C
def test_scenario_c_review_flagged_sneaker_is_a_caution_not_the_pick(engine):
    r = engine.respond(SCEN_C)
    primary = r.recommendations[0]
    assert primary.role == "primary" and primary.product_id != "P003"
    caution = next(c for c in r.recommendations if c.role == "caution")
    assert caution.product_id == "P003" and caution.match_label == "poor"
    assert any("reviewers" in w for w in caution.warnings)
    assert r.assistant_message.startswith("The UrbanStep Canvas Casual Sneaker")


# ------------------------------------------------------------------ scenario D
def test_scenario_d_upsell_is_separate_brief_and_concrete(engine):
    r = engine.respond(SCEN_D)
    assert r.recommendations[0].product_id == "P014" and r.recommendations[0].match_label == "strong"
    u = r.upsell
    assert u.product_id == "P015" and u.role == "upsell" and u.style == "mention" and u.price_delta == pytest.approx(35.0)
    assert "Active noise cancellation" in u.benefits
    assert u.product_id not in [c.product_id for c in r.recommendations]
    assert "family" in r.details.upsell_rejected["P016"]


# ------------------------------------------------------------------ no match
class RoutedClient:
    """One fake for both LLM calls: the query parse and the wording."""
    def __init__(self, parse=None, message=None, error=None):
        self.parse, self.message, self.error = parse or {}, message, error

    def complete_json(self, system, user, max_tokens=3500):
        if self.error:
            raise self.error
        if "Shopper message:" in user:
            return self.parse, {"model": "fake"}
        return {"message": self.message}, {"model": "fake"}


def test_named_product_missing_from_the_catalog_is_an_honest_no_match(catalog):
    eng = Engine(catalog, client=RoutedClient(parse={"product_type": "standing desk", "category": "office"},
                                              message="x"))
    r = eng.respond("I need a standing desk for my home office")
    assert r.type == "no_match" and r.recommendations == [] and r.upsell is None
    assert "standing desk" in r.assistant_message and "$" not in r.assistant_message
    assert r.details.match_reasons and r.details.parsed_query.product_type == "standing desk"


def test_without_an_llm_an_unknown_product_degrades_to_a_question(engine):
    assert engine.respond("I need a standing desk for my home office").type == "clarification"

def test_strict_constraints_no_match_does_not_claim_product_type_is_missing(engine):
    r = engine.respond(
        "I need a waterproof running shoe, no more than $30"
    )

    assert r.type == "no_match"
    assert r.recommendations == []
    assert r.upsell is None
    assert "meets all your requirements" in r.assistant_message.lower()
    assert "don't have a running shoe in the catalog" not in r.assistant_message.lower()



# ------------------------------------------------------------------ LLM wording path
GOOD_D = ("I'd go with the SonicPulse Wireless Earbuds Lite ($24.99). It is wireless. If you ever want active noise "
          "cancellation, the SonicPulse Wireless Earbuds Pro has that for $35 more.")


def test_good_llm_wording_is_used_when_it_passes_the_guard(catalog):
    r = Engine(catalog, client=RoutedClient(message=GOOD_D)).respond(SCEN_D)
    assert r.details.wording == "llm" and r.assistant_message == GOOD_D and r.details.guard_failures == []


def test_bad_llm_wording_is_replaced_by_the_template_and_explained(catalog):
    r = Engine(catalog, client=RoutedClient(message=GOOD_D.replace("$35", "$99"))).respond(SCEN_D)
    assert r.details.wording == "template" and r.assistant_message.startswith("I'd go with")
    assert any("$99" in f for f in r.details.guard_failures)


def test_llm_outage_still_produces_a_complete_answer(catalog):
    r = Engine(catalog, client=RoutedClient(error=LLMError("rate limited"))).respond(SCEN_A)
    assert r.type == "recommendation" and r.recommendations and r.details.wording == "template"
    assert any("LLM unavailable" in f for f in r.details.guard_failures)


# ------------------------------------------------------------------ invariants over many queries
BANK = EXAMPLE_QUERIES + [
    "skincare for my mom", "something for hiking", "a gift for my wife who loves yoga", "cheap power bank",
    "cheap basic yoga mat", "waterproof hiking boot", "wireless earbuds that can handle sweat and rain",
    "a fragrance-free cleanser", "waterproof rain jacket", "wireless earbuds under $30", "kettle", "air fryer",
    "hiking backpack under $40", "pet supplies", "running shoe", "zzzz qqqq", "$$$", "a"]


@pytest.mark.parametrize("message", BANK)
def test_every_response_is_well_formed(engine, catalog, message):
    r = engine.respond(message)
    json.loads(r.model_dump_json())                                      # serialisable
    assert r.assistant_message.strip() and r.type in ("recommendation", "clarification", "no_match")
    prices = {p.product_id: p.price_usd for p in catalog}
    if r.type == "recommendation":
        ids = [c.product_id for c in r.recommendations]
        assert r.recommendations[0].role == "primary" and [c.role for c in r.recommendations].count("primary") == 1
        assert len(ids) == len(set(ids)) and (r.upsell is None or r.upsell.product_id not in ids)
        assert all(c.price_usd == prices[c.product_id] for c in r.recommendations + ([r.upsell] if r.upsell else []))
    else:
        assert r.recommendations == [] and r.upsell is None
    if r.type == "clarification":
        assert r.clarification and r.clarification.options and r.assistant_message.count("?") == 1


def test_exactly_one_primary_in_every_recommendation(engine):
    for message in EXAMPLE_QUERIES + ["cheap power bank", "kettle", "running shoe"]:
        r = engine.respond(message)
        if r.type == "recommendation":
            assert sum(1 for c in r.recommendations if c.role == "primary") == 1


# ------------------------------------------------------------------ history and misc
def test_history_is_accepted_and_echoed_but_memory_is_not_applied_yet(engine):
    hist = [ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hello")]
    r = engine.respond(SCEN_D, hist)
    assert r.details.history_turns == 2 and r.details.memory_applied is False
    assert r.recommendations[0].product_id == "P014"


def test_responses_are_deterministic(engine):
    assert engine.respond(SCEN_A).model_dump() == engine.respond(SCEN_A).model_dump()


def test_surrounding_whitespace_is_ignored(engine):
    assert engine.respond("  " + SCEN_D + "  \n").recommendations[0].product_id == "P014"


def test_details_expose_the_ranking_behind_the_answer(engine):
    d = engine.respond(SCEN_A).details
    assert d.ranking and d.ranking[0].rank == 1 and {"type", "hard"} <= set(d.ranking[0].components)
    assert d.mode == "tradeoff" and d.parsed_query.budget.max == 80 and d.data_source == "test"


def test_run_chat_pipeline_is_the_single_entry_point(monkeypatch, engine):
    monkeypatch.setattr(pipeline, "get_engine", lambda: engine)
    assert pipeline.run_chat_pipeline(SCEN_D).recommendations[0].product_id == "P014"


def test_get_engine_falls_back_to_rules_over_the_csv_when_processed_data_is_missing(monkeypatch):
    monkeypatch.setattr(pipeline, "load_processed", lambda: [])
    pipeline.get_engine.cache_clear()
    try:
        eng = pipeline.get_engine()
        assert "rules only" in eng.data_source and len(eng.products) == 100
    finally:
        pipeline.get_engine.cache_clear()