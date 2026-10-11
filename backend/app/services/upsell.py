"""
Upsell: offer a step-up product ONLY when it genuinely fits. Never a redirect, never forced.

A candidate becomes an upsell only if EVERY check passes (each rejection is recorded with its
reason in UpsellDecision.rejected, so behaviour is explainable and testable):

  0  the recommendation is a real match, and the primary is not already over budget
  1  same FAMILY as the primary: same brand + category + product type
     (so over-ear headphones, a different brand's buds, or a racing shoe are never "upsells")
  2  higher price than the primary
  3  reasonable price gap: <= 3.5x and <= $100  (<= $60 for a "cheap / basic" ask)
  4  within the stated budget stretch: firm = none, standard = +25%, soft = +60%
  5  still satisfies the shopper's requirements (label strong or partial)
  6  rating at least as good as the primary's, and >= 4.0
  7  a CONCRETE, data-backed benefit (see benefits())

Of the survivors the smallest price jump wins (the natural next tier). Tone sets the style:
"cheap" / "basic" -> a one-line optional MENTION; otherwise a SUGGESTION.

Benefits are derived from data, never from a vague "it is better":
  * a feature the candidate states and the primary does not (ANC, GPS); explicit absence first
  * a >= 30% numeric gain: battery life, mAh, room coverage, quarts, watts, thickness
  * a new capability stated in the description (lexicon below), absent from the primary's
  * a water-resistance upgrade
The lexicon is explicit and catalog-tuned on purpose (explainable); on a larger catalog it would be
complemented by LLM-extracted specs.

Try it (from backend/):
    python -m app.services.upsell 'Looking for cheap wireless earbuds, just need something basic'
"""
from __future__ import annotations

import argparse
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.models.schemas import ProcessedProduct, StructuredQuery
from app.services.ambiguity import canonical_type
from app.services.ranker import RankedProduct, shrunk_rating
from app.services.recommender import Recommendation, recommend

# ----------------------------------------------------------------- tunables
MAX_RATIO = 3.5
MAX_DELTA = 100.0
CHEAP_MAX_DELTA = 60.0
BUDGET_STRETCH = {"firm": 1.0, "standard": 1.25, "soft": 1.6}
MIN_RATING = 4.0
RATING_TOLERANCE = 0.05
GAIN_RATIO = 1.3
MAX_BENEFITS = 3

_FEATURE_LABEL = {"active_noise_cancellation": "active noise cancellation", "gps": "built-in GPS",
                  "wireless": "wireless"}

# (regex, label): a capability stated in the candidate's description that the primary's lacks.
_LEXICON = [
    (r"non[\s-]?slip", "a non-slip surface"),
    (r"brushless", "a brushless motor"),
    (r"\bwith diffuser\b|\bdiffuser\b", "a diffuser attachment"),
    (r"keep[\s-]warm", "a keep-warm function"),
    (r"adjustable temperatures?", "adjustable temperature control"),
    (r"two independent baskets|dual[\s-]basket", "two independent baskets"),
    (r"wireless charging", "wireless charging"),
    (r"power delivery", "USB-C Power Delivery (can charge laptops)"),
    (r"air[\s-]quality sensor", "an air-quality sensor"),
    (r"app control", "app control"),
    (r"self[\s-]clean", "a self-clean cycle"),
    (r"digital (?:battery )?display", "a digital battery display"),
    (r"suspension frame|adjustable frame", "an adjustable frame for heavier loads"),
    (r"all[\s-]terrain", "all-terrain tires"),
    (r"transparency mode", "transparency mode"),
    (r"two (?:rechargeable )?batteries", "two batteries"),
    (r"spo2", "SpO2 tracking"),
]
_LEXICON_C = [(re.compile(p, re.I), label) for p, label in _LEXICON]

_WATER_RANK = {None: 0, "splash_resistant": 1, "water_resistant": 2, "waterproof": 3}


# ------------------------------------------------------------------ result models
class UpsellSuggestion(BaseModel):
    product: RankedProduct
    style: Literal["suggest", "mention"]
    price_delta: float
    benefits: list[str]


class UpsellDecision(BaseModel):
    suggestion: Optional[UpsellSuggestion] = None
    skipped_reason: Optional[str] = None
    rejected: dict[str, str] = Field(default_factory=dict, description="product_id -> why it was not offered")


# ------------------------------------------------------------------------ family
def same_family(a: ProcessedProduct, b: ProcessedProduct) -> bool:
    ta, tb = canonical_type(a), canonical_type(b)
    return (a.brand.lower() == b.brand.lower() and a.category == b.category
            and ta is not None and ta == tb)


# ---------------------------------------------------------------------- benefits
def _metrics(desc: str) -> dict[str, tuple[float, str]]:
    """Comparable numbers from a description: key -> (comparable value, display text)."""
    out: dict[str, tuple[float, str]] = {}

    battery: list[tuple[float, str]] = []
    for n in re.findall(r"(\d+(?:\.\d+)?)\s*[- ]?hours?", desc, re.I):
        battery.append((float(n), f"{float(n):g} hours"))
    for n in re.findall(r"(\d+(?:\.\d+)?)\s*[- ]?days?\b", desc, re.I):
        battery.append((float(n) * 24, f"{float(n):g} days"))
    for n in re.findall(r"(\d+)\s*minutes? (?:of )?(?:battery|play|runtime)", desc, re.I):
        battery.append((float(n) / 60, f"{n} minutes"))
    if battery:
        out["battery life"] = max(battery)

    for key, pattern, fmt in [
        ("battery capacity", r"(\d[\d,]*)\s*mAh", "{:,.0f} mAh"),
        ("room coverage", r"(\d[\d,]*)\s*sq\.?\s*ft", "{:,.0f} sq ft"),
        ("capacity", r"(\d+(?:\.\d+)?)\s*-?\s*(?:qt|quart)", "{:g} qt"),
        ("power", r"(\d{2,4})\s*W\b", "{:g} W"),
        ("thickness", r"(\d+(?:\.\d+)?)\s*mm", "{:g} mm"),
    ]:
        vals = [float(v.replace(",", "")) for v in re.findall(pattern, desc, re.I)]
        if vals:
            out[key] = (max(vals), fmt.format(max(vals)))
    return out


def benefits(primary: ProcessedProduct, cand: ProcessedProduct) -> list[str]:
    a, b = primary.attributes, cand.attributes
    out: list[str] = []

    # 1. a feature the candidate states and the primary does not (explicit absence, e.g. "No active
    #    noise cancellation", is the strongest case and is listed first)
    for f in sorted((f for f in b.features if f in _FEATURE_LABEL and f not in a.features),
                    key=lambda f: f not in a.lacks):
        out.append(f"adds {_FEATURE_LABEL[f]}")

    mp, mc = _metrics(primary.description), _metrics(cand.description)   # 2. clear numeric gains
    for key, (vc, text_c) in mc.items():
        if key in mp and mp[key][0] > 0 and vc / mp[key][0] >= GAIN_RATIO:
            prefix = "up to " if key == "battery life" else ""
            out.append(f"{key}: {prefix}{text_c} vs {mp[key][1]}")

    for rx, label in _LEXICON_C:                                # 3. a new stated capability
        if rx.search(cand.description) and not rx.search(primary.description):
            out.append(f"adds {label}")

    wp, wc = a.water_resistance, b.water_resistance             # 4. water-resistance upgrade
    if wc.status and _WATER_RANK.get(wc.level, 0) > _WATER_RANK.get(wp.level if wp.status else None, 0):
        was = (wp.level or "water resistant").replace("_", " ") if wp.status else "not stated"
        out.append(f"water resistance: {(wc.level or 'yes').replace('_', ' ')} vs {was}")

    seen, unique = set(), []
    for line in out:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return unique[:MAX_BENEFITS]


# ------------------------------------------------------------------------ decision
def suggest_upsell(query: StructuredQuery, rec: Recommendation, ranked: list[RankedProduct]) -> UpsellDecision:
    primary = rec.primary
    if rec.mode == "closest_available":
        return UpsellDecision(skipped_reason="there is no confident match to upgrade from")
    if primary.over_budget_by:
        return UpsellDecision(skipped_reason="the recommended product is already above the stated budget")

    cheap_ask = bool({"cheap", "basic"} & set(query.tone_signals))
    pp, p_rating = primary.product.price_usd, shrunk_rating(primary.product)
    rejected: dict[str, str] = {}
    passing: list[tuple[RankedProduct, list[str]]] = []

    for r in ranked:
        p = r.product
        if p.product_id == primary.product.product_id:
            continue
        if not same_family(primary.product, p):
            rejected[p.product_id] = "different product family (brand or product type)"
            continue
        if pp is None or p.price_usd is None or p.price_usd <= pp:
            rejected[p.product_id] = "not a step up in price"
            continue
        delta, ratio = p.price_usd - pp, p.price_usd / pp
        max_delta = CHEAP_MAX_DELTA if cheap_ask else MAX_DELTA
        if ratio > MAX_RATIO or delta > max_delta:
            rejected[p.product_id] = (f"price jump of ${delta:.2f} is too big"
                                      + (" for a cheap/basic ask" if cheap_ask else ""))
            continue
        cap = query.budget.max
        if cap is not None and p.price_usd > cap * BUDGET_STRETCH[query.budget.firmness]:
            rejected[p.product_id] = f"${p.price_usd:.2f} is above your stated ${cap:.0f} budget"
            continue
        if r.label not in ("strong", "partial"):
            rejected[p.product_id] = f"it would not meet your requirements (label: {r.label})"
            continue
        c_rating = shrunk_rating(p)
        if (p.average_rating or 0) < MIN_RATING or (c_rating is not None and p_rating is not None
                                                    and c_rating < p_rating - RATING_TOLERANCE):
            rejected[p.product_id] = "its rating is not at least as good as the recommended product's"
            continue
        gains = benefits(primary.product, p)
        if not gains:
            rejected[p.product_id] = "no concrete, data-backed benefit over the recommended product"
            continue
        passing.append((r, gains))

    if not passing:
        return UpsellDecision(skipped_reason="no suitable step-up found", rejected=rejected)
    best, gains = min(passing, key=lambda x: (x[0].product.price_usd, -len(x[1]), x[0].product.product_id))
    return UpsellDecision(
        suggestion=UpsellSuggestion(product=best, style="mention" if cheap_ask else "suggest",
                                    price_delta=round(best.product.price_usd - pp, 2), benefits=gains),
        rejected=rejected)


def plan_recommendation(
    query: StructuredQuery,
    ranked: list[RankedProduct],
) -> tuple[Optional[Recommendation], UpsellDecision]:
    """Build a recommendation, enforcing explicit hard budget limits."""

    candidates = ranked
    budget = query.budget

    # A firm maximum is a hard constraint, not just a ranking preference.
    if budget.firmness == "firm" and budget.max is not None:
        candidates = [
            item for item in ranked
            if item.product.price_usd <= budget.max
        ]

        if not candidates:
            return None, UpsellDecision(
                skipped_reason="no products within the firm budget"
            )

    rec = recommend(candidates)

    if rec is None:
        return None, UpsellDecision(
            skipped_reason="nothing suitable to recommend"
        )

    decision = suggest_upsell(query, rec, candidates)

    if decision.suggestion:
        rec = recommend(
            candidates,
            exclude_ids={decision.suggestion.product.product.product_id},
        )

    return rec, decision


# ================================================================================= CLI
def main() -> None:
    from app.config import get_settings
    from app.data.loader import load_catalog, load_processed
    from app.services.ambiguity import gate, match_check
    from app.services.attribute_extractor import rules_extract
    from app.services.llm_client import LLMClient
    from app.services.query_understanding import understand_query
    from app.services.ranker import Ranker
    from app.services.retriever import Retriever
    from app.services.review_analyzer import ensure_review_evidence

    ap = argparse.ArgumentParser(description="Show the recommendation and the upsell decision")
    ap.add_argument("message")
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    products = ensure_review_evidence(load_processed() or [rules_extract(r) for r in load_catalog()])
    client = LLMClient(settings) if (args.llm and settings.llm_enabled) else None
    q = understand_query(args.message, client=client, known_brands=sorted({p.brand for p in products}))
    g = gate(q, products)
    if g.decision == "clarify":
        print("ASK:", g.clarifying_question)
        return
    retrieval = Retriever(products).retrieve(g.query)
    if match_check(g.query, retrieval).decision == "no_match":
        print("NO MATCH")
        return
    rec, up = plan_recommendation(g.query, Ranker(products).rank(g.query, retrieval.candidates))

    def line(r: RankedProduct) -> str:
        return f"{r.product.product_id} ${r.product.price_usd:.2f} [{r.label}] {r.product.title}"

    print("PRIMARY  ", line(rec.primary))
    for r in rec.secondary:
        print("SECONDARY", line(r))
    if up.suggestion:
        s = up.suggestion
        print(f"\nUPSELL ({s.style}): {line(s.product)}  +${s.price_delta:.2f}")
        for b in s.benefits:
            print("      +", b)
    else:
        print("\nNO UPSELL:", up.skipped_reason)
    if up.rejected:
        print("\nconsidered and rejected:")
        for pid, why in sorted(up.rejected.items()):
            print(f"   {pid}: {why}")


if __name__ == "__main__":
    main()