"""Tests for the loader and attribute extractor (rules, LLM guardrails, fallback)."""
import pytest

from app.config import get_settings
from app.data.loader import load_catalog
from app.models.schemas import ProcessedProduct, RawProduct
from app.services import attribute_extractor as ax
from app.services.llm_client import LLMError

CATALOG_EXISTS = get_settings().catalog_csv.exists()


def make_raw(pid="T1", title="Test Product", brand="Acme", desc="", category="footwear") -> RawProduct:
    return RawProduct(product_id=pid, category=category, brand=brand, title=title,
                      description=desc, price_usd=10.0, average_rating=4.0, num_reviews=10,
                      review_snippets=["ok", "fine"])


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, 0

    def complete_json(self, system, user, max_tokens=3500):
        self.calls += 1
        if self.error:
            raise self.error
        return self.payload, {"model": "fake-model", "total_tokens": 100}


# ----------------------------------------------------------------- weight
def test_weight_per_shoe_with_parenthetical_grams():
    w = ax.extract_weight("Weighs 10.2 oz (289 g) per shoe in men's size 9.")
    assert w.basis == "per_shoe" and w.grams == pytest.approx(289.2, abs=1)


def test_weight_pounds_and_ounces_per_pair():
    w = ax.extract_weight("Weighs 1 lb 14 oz (850 g) per pair.")
    assert w.basis == "per_pair" and w.grams == pytest.approx(850.5, abs=1)


def test_weight_at_just_pattern():
    assert ax.extract_weight("Ultralight runner at just 7.8 oz (221 g).").grams == pytest.approx(221, abs=1)


def test_weight_ignores_numbers_that_are_not_the_product_weight():
    assert ax.extract_weight("Fits dogs up to 50 lb.").grams is None
    assert ax.extract_weight("Resistance bands from 10 to 50 lb with handles.").grams is None


def test_missing_weight_is_unknown_not_zero():
    w = ax.extract_weight("A dependable entry-level trainer for casual joggers.")
    assert w.grams is None and w.basis is None


# ------------------------------------------------------------ water resistance
def test_waterproof_detected_with_evidence():
    w = ax.extract_water_resistance("Fully waterproof trail runner with a sealed-seam membrane.")
    assert w.status is True and w.level == "waterproof" and "waterproof" in w.evidence


def test_explicit_negation_is_false():
    w = ax.extract_water_resistance("Not waterproof; designed for breathability in warm weather.")
    assert w.status is False


def test_silence_is_unknown_not_false():
    w = ax.extract_water_resistance("Breathable engineered mesh upper with a responsive foam midsole.")
    assert w.status is None and w.source == "none"


def test_ipx_rating_maps_to_level():
    w = ax.extract_water_resistance("Earbuds with Bluetooth 5.3. IPX4 splash resistance.")
    assert w.status is True and w.level == "splash_resistant" and w.rating == "IPX4"


def test_atm_rating():
    w = ax.extract_water_resistance("Fitness tracker with 5ATM water resistance.")
    assert w.status is True and w.rating == "5ATM"


def test_leakproof_is_not_water_resistance():
    assert ax.extract_water_resistance("Double-wall bottle with a leakproof lid.").status is None


def test_evidence_is_verbatim_substring():
    text = "Great shoe. Waterproof membrane keeps feet dry. Light too."
    w = ax.extract_water_resistance(text)
    assert w.evidence in text


# ---------------------------------------------------------------- feature flags
def test_negated_noise_cancellation_goes_to_lacks():
    feats, lacks = ax.extract_feature_flags("True-wireless earbuds. No active noise cancellation.")
    assert "active_noise_cancellation" in lacks and "active_noise_cancellation" not in feats
    assert "wireless" in feats


def test_wired_is_not_wireless():
    feats, lacks = ax.extract_feature_flags("Wired in-ear earphones. Not wireless.")
    assert "wireless" in lacks and "wireless" not in feats


# --------------------------------------------------------------- LLM guardrails
def test_llm_unsupported_claims_are_dropped_and_logged():
    raw = make_raw(desc="Breathable mesh upper with a foam midsole. Weighs 8 oz per shoe.")
    p = ax.rules_extract(raw)
    llm = ax._LLMProduct(
        product_id="T1", product_type="running shoe",
        materials=["mesh", "leather"],                                  # leather not in text
        specs={"battery_hours": 99, "midsole_mm": 8},                   # 99 not in text; 8 is
        water_resistance=ax._LLMWater(status=True, level="waterproof",
                                      evidence="Totally waterproof design"),   # not verbatim
        features=["gps_tracking"],                                      # not in text
    )
    ax._apply_llm(p, llm)
    assert "leather" not in p.attributes.materials and "mesh" in p.attributes.materials
    assert "battery_hours" not in p.attributes.specs and p.attributes.specs.get("midsole_mm") == 8
    assert p.attributes.water_resistance.status is None
    assert "gps_tracking" not in p.attributes.features
    assert sum(n.startswith("dropped") for n in p.extraction.notes) >= 4


def test_llm_water_claim_accepted_with_verbatim_quote():
    raw = make_raw(desc="Sealed seams keep water out completely. Light trail runner.")
    p = ax.rules_extract(raw)
    assert p.attributes.water_resistance.status is None   # rules found no keyword
    llm = ax._LLMProduct(product_id="T1", water_resistance=ax._LLMWater(
        status=True, level="waterproof", evidence="Sealed seams keep water out completely."))
    ax._apply_llm(p, llm)
    assert p.attributes.water_resistance.status is True
    assert p.attributes.water_resistance.source == "llm"


def test_llm_false_claim_requires_negation_in_quote():
    raw = make_raw(desc="Breathable mesh runner for summer.")
    p = ax.rules_extract(raw)
    llm = ax._LLMProduct(product_id="T1", water_resistance=ax._LLMWater(
        status=False, evidence="Breathable mesh runner for summer."))
    ax._apply_llm(p, llm)
    assert p.attributes.water_resistance.status is None


def test_rules_win_over_llm_on_water_resistance():
    raw = make_raw(desc="Not waterproof; designed for warm weather.")
    p = ax.rules_extract(raw)
    llm = ax._LLMProduct(product_id="T1", water_resistance=ax._LLMWater(
        status=True, level="waterproof", evidence="designed for warm weather"))
    ax._apply_llm(p, llm)
    assert p.attributes.water_resistance.status is False and p.attributes.water_resistance.source == "rules"


def test_spec_numbers_with_thousands_separator():
    assert ax._number_in_text(20000, "Slim 20,000mAh power bank")
    assert not ax._number_in_text(2000, "Slim 20,000mAh power bank")


# ---------------------------------------------------------------- orchestration
def test_llm_failure_falls_back_to_rules_without_raising():
    raws = [make_raw("T1", desc="Waterproof boot. Weighs 2 lb (907 g)."), make_raw("T2", desc="Plain shoe.")]
    out = ax.build_processed_products(raws, client=FakeClient(error=LLMError("rate limited")), log=lambda *_: None)
    assert len(out) == 2 and all(p.extraction.method == "rules" for p in out)
    assert out[0].attributes.water_resistance.status is True
    assert any("LLM unavailable" in n for n in out[0].extraction.notes)


def test_llm_success_marks_method_and_model():
    raws = [make_raw("T1", desc="Cotton canvas sneaker for casual wear.")]
    payload = {"products": [{"product_id": "T1", "product_type": "Canvas Sneaker",
                             "materials": ["cotton", "canvas"], "intended_use": ["casual wear"]}]}
    out = ax.build_processed_products(raws, client=FakeClient(payload=payload), log=lambda *_: None)
    p = out[0]
    assert p.extraction.method == "rules+llm" and p.extraction.model == "fake-model"
    assert p.attributes.product_type == "canvas sneaker"
    assert "casual wear" in p.attributes.intended_use


def test_cache_skips_llm_for_unchanged_products():
    raws = [make_raw("T1", desc="Cotton canvas sneaker.")]
    payload = {"products": [{"product_id": "T1", "product_type": "sneaker"}]}
    first = ax.build_processed_products(raws, client=FakeClient(payload=payload), log=lambda *_: None)
    fake2 = FakeClient(payload=payload)
    second = ax.build_processed_products(raws, client=fake2, existing=first, log=lambda *_: None)
    assert fake2.calls == 0 and second[0].extraction.method == "rules+llm"


def test_changed_description_invalidates_cache():
    r1 = make_raw("T1", desc="Cotton canvas sneaker.")
    first = ax.build_processed_products([r1], client=FakeClient(payload={"products": [{"product_id": "T1"}]}),
                                        log=lambda *_: None)
    r2 = make_raw("T1", desc="Leather boot, waterproof.")
    fake2 = FakeClient(payload={"products": [{"product_id": "T1"}]})
    ax.build_processed_products([r2], client=fake2, existing=first, log=lambda *_: None)
    assert fake2.calls == 1


# ------------------------------------------------- real (synthetic) catalog checks
@pytest.mark.skipif(not CATALOG_EXISTS, reason="synthetic catalog CSV not present")
class TestOnCatalog:
    @pytest.fixture()
    def processed(self):
        raw = load_catalog()
        return {p.product_id: p for p in (ax.rules_extract(r) for r in raw)}

    def test_loader_returns_100_valid_products(self):
        raw = load_catalog()
        assert len(raw) == 100 and len({r.product_id for r in raw}) == 100
        assert all(len(r.review_snippets) >= 3 for r in raw)

    def test_every_product_validates_against_schema(self, processed):
        assert len(processed) == 100
        for p in processed.values():
            ProcessedProduct.model_validate(p.model_dump())

    def test_waterproof_shoe_p001(self, processed):
        a = processed["P001"].attributes
        assert a.water_resistance.status is True and a.weight.basis == "per_shoe"

    def test_p002_silent_on_water_so_unknown(self, processed):
        assert processed["P002"].attributes.water_resistance.status is None

    def test_p003_description_silent_on_rain(self, processed):
        """The rain problem exists only in reviews; description extraction must not invent it."""
        assert processed["P003"].attributes.water_resistance.status is None

    def test_p005_explicitly_not_waterproof(self, processed):
        assert processed["P005"].attributes.water_resistance.status is False

    def test_p006_claims_waterproof_in_description(self, processed):
        """Description claims waterproof; reviews contradict (handled in Phase 7)."""
        assert processed["P006"].attributes.water_resistance.status is True

    def test_p011_weight_and_water_unknown(self, processed):
        a = processed["P011"].attributes
        assert a.weight.grams is None and a.water_resistance.status is None

    def test_p014_lite_has_no_anc_but_pro_does(self, processed):
        assert "active_noise_cancellation" in processed["P014"].attributes.lacks
        assert "active_noise_cancellation" in processed["P015"].attributes.features

    def test_p026_wired_earphones_not_wireless(self, processed):
        assert "wireless" in processed["P026"].attributes.lacks

    def test_p099_dog_weight_limit_not_mistaken_for_product_weight(self, processed):
        assert processed["P099"].attributes.weight.grams is None

    def test_p024_water_unknown_in_description(self, processed):
        """The 'survived rain' signal is review-only."""
        assert processed["P024"].attributes.water_resistance.status is None


def test_llm_intended_use_must_be_grounded_in_the_listing():
    raw = make_raw(desc="Waterproof leather hiking shoe with a supportive midsole for day hikes.")
    p = ax.rules_extract(raw)
    llm = ax._LLMProduct(product_id="T1", intended_use=["gift", "day hiking", "mountaineering"])
    ax._apply_llm(p, llm)
    assert "gift" not in p.attributes.intended_use and "mountaineering" not in p.attributes.intended_use
    assert "day hiking" in p.attributes.intended_use
    assert sum(n.startswith("dropped unsupported intended_use") for n in p.extraction.notes) == 2


def test_revalidate_drops_unsupported_tags_and_keeps_rule_derived_ones():
    raw = make_raw(desc="Waterproof leather hiking shoe for day hikes.")
    p = ax.rules_extract(raw)
    p.attributes.intended_use.append("gift")
    assert ax.revalidate_intended_use([p]) == 1
    assert "gift" not in p.attributes.intended_use and "hiking" in p.attributes.intended_use
    assert ax.revalidate_intended_use([p]) == 0                      # idempotent
