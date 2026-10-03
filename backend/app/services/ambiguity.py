"""
Ambiguity handling: decide whether to ANSWER, ASK one clarifying question, or say NO MATCH.

Two stages, both deterministic (the LLM never decides this):

  Stage 1  gate()         BEFORE retrieval. Is the request specific enough to rank anything?
                          A transparent specificity score; below CLARIFY_BELOW we first try to
                          resolve the request from the catalog (e.g. "yoga" only maps to yoga
                          mats), otherwise we ask ONE question whose options come from real
                          catalog data, never invented.
  Stage 2  match_check()  AFTER retrieval. The shopper named a specific product type, but is
                          there anything real in the catalog? If not: honest NO MATCH, not a
                          forced list of unrelated products.

Specificity score (0..1):
  product type named ........ +0.55   (only a category named: +0.25)
  any concrete attribute/use  +0.20   (+0.05 more for two or more)
  budget given ............... +0.10
  vague words, no product type  -0.05 each (max 3)
  recipient but no interest .. -0.25   ("something nice for my mom")
  no product type AND no category -> capped at 0.29: attributes alone cannot name a product
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.models.schemas import ProcessedProduct, StructuredQuery
from app.services.query_understanding import detect_product_type
from app.services.retriever import RetrievalResult

CLARIFY_BELOW = 0.30
NO_TYPE_CAP = 0.29
MAX_OPTIONS = 3

# The catalog category "beauty" is broader than what shoppers say. A narrower word narrows the options.
_SKINCARE_TYPES = {"moisturizer", "night cream", "sunscreen", "serum", "cleanser", "lip balm"}

_FEMALE = {"mom", "mother", "mum", "wife", "girlfriend", "sister", "grandma", "grandmother", "daughter", "niece", "aunt"}
_MALE = {"dad", "father", "husband", "boyfriend", "brother", "grandpa", "grandfather", "son", "nephew", "uncle"}


# ------------------------------------------------------------------- result models
class GateDecision(BaseModel):
    decision: Literal["clarify", "proceed"]
    specificity: float
    reasons: list[str] = Field(default_factory=list)
    clarifying_question: Optional[str] = None
    options: list[str] = Field(default_factory=list, description="Real catalog product types / categories")
    query: StructuredQuery = Field(..., description="The query to use next (may carry an inferred product type)")


class MatchDecision(BaseModel):
    decision: Literal["answer", "no_match"]
    reasons: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------------ scoring
def specificity(q: StructuredQuery) -> tuple[float, list[str]]:
    score, why = 0.0, []
    if q.product_type:
        score += 0.55
        why.append(f"product type named ('{q.product_type}')")
    elif q.category:
        score += 0.25
        why.append(f"only a category named ('{q.category}')")
    attrs = len(q.hard_requirements) + len(q.soft_preferences) + len(q.intended_use) + (1 if q.brand else 0)
    if attrs >= 1:
        score += 0.20 + (0.05 if attrs >= 2 else 0.0)
        why.append(f"{attrs} concrete detail(s) given")
    if q.budget.max is not None or q.budget.min is not None:
        score += 0.10
        why.append("budget given")
    n_vague = sum(1 for s in q.vagueness_signals if s.startswith("vague_word:"))
    if not q.product_type and n_vague:
        score -= 0.05 * min(3, n_vague)
        why.append(f"{n_vague} vague word(s)")
    if "recipient_without_interest" in q.vagueness_signals:
        score -= 0.25
        why.append("a recipient but no hint of what they like")
    if not (q.product_type or q.category):
        score = min(score, NO_TYPE_CAP)
        why.append("no product type or category: attributes alone cannot name a product")
    return round(max(0.0, min(1.0, score)), 2), why


# ----------------------------------------------------------- options from the catalog
def canonical_type(p: ProcessedProduct) -> Optional[str]:
    ptype, _ = detect_product_type(p.title)
    return ptype


def _ranked_types(products: list[ProcessedProduct]) -> list[tuple[str, str]]:
    """Distinct canonical types as (type, category), most common first, then best rated."""
    count: dict[tuple[str, str], int] = defaultdict(int)
    rating: dict[tuple[str, str], float] = defaultdict(float)
    for p in products:
        t = canonical_type(p)
        if t:
            key = (t, p.category)
            count[key] += 1
            rating[key] = max(rating[key], p.average_rating or 0.0)
    return sorted(count, key=lambda k: (-count[k], -rating[k], k[0]))


def _diverse(types: list[tuple[str, str]], n: int = MAX_OPTIONS) -> list[tuple[str, str]]:
    """Prefer options from different categories, so the question spans the real choices."""
    chosen, seen = [], set()
    for t in types:
        if t[1] not in seen:
            chosen.append(t)
            seen.add(t[1])
    for t in types:
        if t not in chosen:
            chosen.append(t)
    return chosen[:n]


def _join(options: list[str]) -> str:
    if len(options) <= 1:
        return "".join(options)
    return ", ".join(options[:-1]) + f", or {options[-1]}"


def _pronoun(recipient: Optional[str]) -> str:
    return "she" if recipient in _FEMALE else "he" if recipient in _MALE else "they"


def _use_matches(p: ProcessedProduct, uses: list[str]) -> bool:
    have = {u.lower() for u in p.attributes.intended_use}
    blob = (p.title + " " + p.description).lower()
    return any(u.lower() in have or u.lower() in blob for u in uses)


# ------------------------------------------------------------------------- stage 1
def gate(q: StructuredQuery, catalog: list[ProcessedProduct]) -> GateDecision:
    score, why = specificity(q)
    if score >= CLARIFY_BELOW:
        return GateDecision(decision="proceed", specificity=score, reasons=why, query=q)

    # Too vague to rank. First try to resolve it from the catalog before bothering the shopper.
    source, pool = None, []
    if q.category:
        source, pool = f"{q.category}", [p for p in catalog if p.category == q.category]
    elif q.intended_use:
        source, pool = f"{q.intended_use[0]}", [p for p in catalog if _use_matches(p, q.intended_use)]
    elif q.is_gift or q.recipient:
        source, pool = "gift", [p for p in catalog if "gift" in {u.lower() for u in p.attributes.intended_use}]

    types = _ranked_types(pool)
    if re.search(r"skin\s?care", q.raw_message, re.I):
        types = [t for t in types if t[0] in _SKINCARE_TYPES] or types
    if len(types) == 1 and source != "gift":
        t, cat = types[0]
        enriched = q.model_copy(update={"product_type": t, "category": q.category or cat})
        enriched.notes = q.notes + [f"inferred product type '{t}': the only {source} product type in the catalog"]
        return GateDecision(decision="proceed", specificity=score, query=enriched,
                            reasons=why + [f"resolved from the catalog: the only '{source}' product type is '{t}'"])

    if types:
        options = [t for t, _ in _diverse(types)]
        if source == "gift":
            question = (f"Happy to help. What kind of thing is {_pronoun(q.recipient)} into? "
                        f"I'm thinking {_join(options)}.")
        elif q.category:
            question = f"Happy to help with {q.category}. Are you after something specific, like {_join(options)}?"
        else:
            question = f"For {source}, what are you looking for: {_join(options)}?"
    else:
        cats = sorted({p.category for p in catalog}, key=lambda c: -sum(1 for p in catalog if p.category == c))
        options = cats[:4]
        question = f"Happy to help. What kind of product are you looking for? For example {_join(options)}."
    return GateDecision(decision="clarify", specificity=score, reasons=why, clarifying_question=question,
                        options=options, query=q)


# ------------------------------------------------------------------------- stage 2
def match_check(q: StructuredQuery, retrieval: RetrievalResult) -> MatchDecision:
    d = retrieval.diagnostics
    if d.n_candidates == 0:
        return MatchDecision(decision="no_match", reasons=["nothing in the catalog resembles this request"])
    if q.product_type and not d.has_type_match:
        return MatchDecision(decision="no_match",
                             reasons=[f"no '{q.product_type}' (or close relative) in the catalog; "
                                      f"the nearest items are only loosely related"])
    return MatchDecision(decision="answer", reasons=["the catalog has real matches for the requested type"])