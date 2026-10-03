"""
Recommendation selection: ranked products -> primary pick, secondary options, cautions.

  primary    best 'strong' or 'partial' product. When scores are nearly tied (within
             PRIMARY_TIE_WINDOW) prefer the option WITHOUT review warnings.
  secondary  up to 2. First a product that fixes the primary's main tradeoff (primary over
             budget -> an in-budget option), then close runners-up (within SECONDARY_WINDOW)
             that are not worse than the primary on the shopper's hard requirements.
  cautions   products that match the shopper's words but are held back by REVIEW evidence
             (e.g. the canvas sneaker that soaks in light rain). They are surfaced with their
             warning instead of being silently dropped, so the response can say so.
  mode       confident (primary is strong) | tradeoff (partial) |
             closest_available (nothing eligible: be honest that nothing matches well)

Upsell is a separate decision (upsell.py); pass its product id in `exclude_ids` so the same
product is not shown twice.

Try the whole chain (from backend/):
    python -m app.services.recommender 'I need a lightweight waterproof running shoe, ideally under $80'
"""
from __future__ import annotations

import argparse
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.services.ranker import RankedProduct

PRIMARY_TIE_WINDOW = 0.02
SECONDARY_WINDOW = 0.08
SECONDARY_MAX = 2
CAUTION_MAX = 2
ELIGIBLE = ("strong", "partial")


class Recommendation(BaseModel):
    mode: Literal["confident", "tradeoff", "closest_available"]
    primary: RankedProduct
    secondary: list[RankedProduct] = Field(default_factory=list)
    cautions: list[RankedProduct] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _type_relevance(r: RankedProduct) -> float:
    return next((c.value for c in r.components if c.name == "type"), 0.0)


def review_flags(r: RankedProduct) -> int:
    """How many requirements carry a review-based warning or a review-derived shortfall."""
    n = 0
    for q in r.requirements:
        if q.review_note:
            n += 1
        elif q.source != "description" and q.state in ("unmet", "partial", "conflict"):
            n += 1
    return n


def _unmet_hard(r: RankedProduct) -> int:
    """Hard requirements not fully met (unknown, partial, unmet or conflicting)."""
    return sum(1 for q in r.requirements if q.kind == "hard" and q.state != "met")


def _held_back_by_reviews(r: RankedProduct) -> bool:
    if r.label == "conflict":
        return True
    return any(q.source != "description" and q.state in ("unmet", "conflict") for q in r.requirements)


def recommend(ranked: list[RankedProduct], exclude_ids: Optional[set[str]] = None) -> Optional[Recommendation]:
    if not ranked:
        return None
    exclude = exclude_ids or set()
    eligible = [r for r in ranked if r.label in ELIGIBLE]
    notes: list[str] = []

    if eligible:
        top = eligible[0].score
        near = [r for r in eligible if r.score >= top - PRIMARY_TIE_WINDOW]
        primary = min(near, key=lambda r: (review_flags(r), -r.score, r.product.product_id))
        if primary is not eligible[0]:
            notes.append(f"{primary.product.product_id} chosen over {eligible[0].product.product_id}: "
                         "near-tied, and it carries fewer review warnings")
        mode = "confident" if primary.label == "strong" else "tradeoff"
    else:
        fallback = [r for r in ranked if r.label == "conflict"] or ranked
        primary, mode = fallback[0], "closest_available"
        notes.append("no product satisfies the request well; this is the closest available")

    secondary: list[RankedProduct] = []
    if mode != "closest_available":
        pool = [r for r in eligible if r is not primary and r.product.product_id not in exclude]
        if primary.over_budget_by:                       # fix the main tradeoff: show an in-budget option
            fix = next((r for r in pool if not r.over_budget_by), None)
            if fix:
                secondary.append(fix)
        for r in pool:
            if len(secondary) >= SECONDARY_MAX:
                break
            # a runner-up must not be worse than the primary on the shopper's hard requirements
            if r not in secondary and r.score >= primary.score - SECONDARY_WINDOW \
                    and _unmet_hard(r) <= _unmet_hard(primary):
                secondary.append(r)

    shown = {primary.product.product_id, *(r.product.product_id for r in secondary)}
    cautions = [r for r in ranked
                if r.product.product_id not in shown and _type_relevance(r) >= 0.9 and _held_back_by_reviews(r)
                ][:CAUTION_MAX]
    return Recommendation(mode=mode, primary=primary, secondary=secondary, cautions=cautions, notes=notes)


# ======================================================================== CLI (whole chain)
def _line(r: RankedProduct) -> str:
    return f"{r.product.product_id} ${r.product.price_usd:.2f} [{r.label}, score {r.score:.2f}] {r.product.title}"


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

    ap = argparse.ArgumentParser(description="Run the whole recommendation chain for one message")
    ap.add_argument("message")
    ap.add_argument("--llm", action="store_true", help="use Groq for query understanding")
    args = ap.parse_args()

    settings = get_settings()
    products = ensure_review_evidence(load_processed() or [rules_extract(r) for r in load_catalog()])
    client = LLMClient(settings) if (args.llm and settings.llm_enabled) else None
    q = understand_query(args.message, client=client, known_brands=sorted({p.brand for p in products}))

    g = gate(q, products)
    print(f"specificity {g.specificity} -> {g.decision}   ({'; '.join(g.reasons)})")
    if g.decision == "clarify":
        print("\nASK:", g.clarifying_question, "\noptions:", g.options)
        return
    q = g.query
    retrieval = Retriever(products).retrieve(q)
    m = match_check(q, retrieval)
    print(f"match check -> {m.decision}: {m.reasons[0]}")
    if m.decision == "no_match":
        print("\nNO MATCH. Nearest (loosely related) items:",
              [c.product.title for c in retrieval.candidates[:3]])
        return

    rec = recommend(Ranker(products).rank(q, retrieval.candidates))
    print(f"\nmode: {rec.mode}")
    print("PRIMARY  ", _line(rec.primary))
    for t in rec.primary.reasons:
        print("      +", t)
    for t in rec.primary.tradeoffs:
        print("      -", t)
    for r in rec.secondary:
        print("SECONDARY", _line(r))
        for t in r.tradeoffs:
            print("      -", t)
    for r in rec.cautions:
        print("CAUTION  ", _line(r))
        for t in r.tradeoffs:
            print("      !", t)
    for n in rec.notes:
        print("note:", n)


if __name__ == "__main__":
    main()