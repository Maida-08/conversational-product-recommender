"""Tests for the ambiguity gate: when to ask, when to infer, when to say no match."""
import pytest

from app.config import get_settings
from app.data.loader import load_catalog
from app.models.schemas import StructuredQuery
from app.services.ambiguity import (CLARIFY_BELOW, canonical_type, gate, match_check, specificity)
from app.services.attribute_extractor import rules_extract
from app.services.query_understanding import rules_parse
from app.services.retriever import Retriever

pytestmark = pytest.mark.skipif(not get_settings().catalog_csv.exists(), reason="synthetic catalog missing")


@pytest.fixture(scope="module")
def products():
    return [rules_extract(r) for r in load_catalog()]


@pytest.fixture(scope="module")
def retriever(products):
    return Retriever(products)


# ------------------------------------------------------------------ specificity score
def test_specific_query_is_well_above_the_threshold():
    score, _ = specificity(rules_parse("I need a lightweight waterproof running shoe, ideally under $80"))
    assert score >= 0.8


def test_the_assignments_ambiguous_example_scores_zero():
    score, why = specificity(rules_parse("Something nice for my mom"))
    assert score == 0.0 and any("recipient" in w for w in why)


def test_attributes_and_budget_without_a_product_cannot_pass():
    score, _ = specificity(rules_parse("waterproof lightweight under $80"))
    assert 0 < score < CLARIFY_BELOW          # capped: attributes alone cannot name a product


def test_a_bare_category_is_not_specific_enough():
    assert specificity(rules_parse("kitchen stuff"))[0] < CLARIFY_BELOW


def test_score_is_clamped_to_unit_interval():
    for msg in ["", "nice good great something", "a waterproof lightweight running shoe under $50 from SonicPulse"]:
        assert 0.0 <= specificity(rules_parse(msg))[0] <= 1.0


# ----------------------------------------------------------- scenario B: clarification
def test_scenario_b_asks_one_question_with_real_gift_options(products):
    g = gate(rules_parse("Something nice for my mom"), products)
    assert g.decision == "clarify" and g.clarifying_question
    assert 1 <= len(g.options) <= 3 and "she" in g.clarifying_question
    gift_types = {canonical_type(p) for p in products if "gift" in p.attributes.intended_use}
    assert set(g.options) <= gift_types           # never invented: all come from catalog gift products
    assert g.clarifying_question.count("?") == 1  # exactly one question


def test_pronoun_follows_the_recipient(products):
    assert "he " in gate(rules_parse("a birthday gift for my dad"), products).clarifying_question


def test_category_only_question_offers_real_types_of_that_category(products):
    g = gate(rules_parse("kitchen stuff"), products)
    kitchen_types = {canonical_type(p) for p in products if p.category == "kitchen"}
    assert g.decision == "clarify" and set(g.options) <= kitchen_types and "kitchen" in g.clarifying_question


def test_skincare_does_not_offer_a_hair_dryer(products):
    g = gate(rules_parse("skincare for my mom"), products)
    assert g.decision == "clarify" and "hair dryer" not in g.options


def test_use_only_question_offers_types_that_serve_that_use(products):
    g = gate(rules_parse("something for hiking"), products)
    assert g.decision == "clarify" and {"backpack", "hiking shoe"} & set(g.options)


def test_no_information_at_all_falls_back_to_real_categories(products):
    g = gate(rules_parse("cheap stuff"), products)
    cats = {p.category for p in products}
    assert g.decision == "clarify" and set(g.options) <= cats and len(g.options) >= 3


def test_attributes_and_budget_without_a_product_still_asks(products):
    assert gate(rules_parse("waterproof lightweight under $80"), products).decision == "clarify"


# ------------------------------------------------- resolving from the catalog instead
def test_single_product_type_for_a_use_is_inferred_without_asking(products):
    g = gate(rules_parse("a gift for my wife who loves yoga"), products)
    assert g.decision == "proceed" and g.query.product_type == "yoga mat"
    assert any("inferred" in n for n in g.query.notes)


def test_single_product_type_in_a_category_is_inferred(products):
    g = gate(rules_parse("pet supplies"), products)
    assert g.decision == "proceed" and g.query.product_type == "dog bed"


def test_inference_does_not_mutate_the_original_query(products):
    q = rules_parse("pet supplies")
    gate(q, products)
    assert q.product_type is None


def test_specific_query_proceeds_unchanged(products):
    q = rules_parse("running shoe")
    g = gate(q, products)
    assert g.decision == "proceed" and g.query.product_type == "running shoe" and not g.options


# ------------------------------------------------------------ stage 2: no match
def test_named_product_missing_from_the_catalog_is_no_match(products, retriever):
    q = StructuredQuery(raw_message="I need a standing desk for my home office",
                        product_type="standing desk", category="office")
    g = gate(q, products)
    assert g.decision == "proceed"                       # specific enough to try
    m = match_check(g.query, retriever.retrieve(g.query))
    assert m.decision == "no_match" and "standing desk" in m.reasons[0]


def test_rules_only_unknown_product_degrades_to_a_question(products):
    assert gate(rules_parse("I need a standing desk for my home office"), products).decision == "clarify"


def test_real_product_is_answered(products, retriever):
    q = rules_parse("running shoe")
    assert match_check(q, retriever.retrieve(q)).decision == "answer"


def test_empty_retrieval_is_no_match(products, retriever):
    q = rules_parse("zzzz qqqq")
    assert match_check(q, retriever.retrieve(q)).decision == "no_match"