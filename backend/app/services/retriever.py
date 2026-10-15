"""
Candidate retrieval: StructuredQuery -> a short list of candidate products + diagnostics.

Two signals, combined explicitly (no black box):

  type_relevance (0.55) .. is this the KIND of product requested? Graded 0..1 from the
                           query's product type vs the product's type/title tokens,
                           with small "families" (earbuds ~ headphones, sneaker ~ boot).
  text_similarity (0.45) . TF-IDF cosine between the query text and the product's
                           title + description + extracted attributes.
  brand match (+0.10) .... only when the shopper named a brand.

Deliberate choices:
  * BUDGET NEVER FILTERS. Filtering would hide the best match when it is a little over
    budget. The ranker penalises price and the response explains the tradeoff.
  * Related types stay in the pool at lower relevance; hard requirements are enforced by
    the ranker, so nothing is silently dropped here.
  * Diagnostics (top score, whether any real type match exists, category spread) are
    returned for the ambiguity gate, which decides "ask a question" vs "answer".

Try it (from backend/):
    python -m app.services.retriever 'lightweight waterproof running shoe under $80'
"""
from __future__ import annotations

import argparse
import re
from typing import Optional

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.schemas import ProcessedProduct, StructuredQuery

# ----------------------------------------------------------------- tunables
W_TYPE = 0.55
W_TEXT = 0.45
BRAND_BOOST = 0.10
SIM_SCALE = 2.5          # TF-IDF cosines for short queries are small; scale into ~0..1
MIN_SCORE = 0.15         # below this a product is not a candidate at all
DEFAULT_K = 12
STRONG_TYPE = 0.6        # type_relevance at or above this counts as a "real" type match

# Product "families": different members are related products (score 0.6), not unrelated.
_FAMILIES = [
    {"shoe", "sneaker", "boot", "sandal", "trainer", "runner", "loafer"},
    {"earbud", "earphone", "headphone", "headset"},
    {"backpack", "rucksack", "daypack"},
    {"blanket", "throw"},
]
_FOOTWEAR = _FAMILIES[0]

# Requirement tag -> words that help TF-IDF find products describing it.
_REQ_TERMS = {
    "waterproof": "waterproof water resistant sealed rain",
    "water_resistant": "water resistant waterproof rain splash",
    "lightweight": "lightweight light ultralight",
    "wireless": "wireless bluetooth",
    "cordless": "cordless battery",
    "noise_cancelling": "noise cancellation anc",
    "gps": "gps",
    "non_slip": "non slip grip",
    "quiet": "quiet low noise sleep mode",
    "long_battery": "battery hours long battery",
    "fast_charging": "fast charge power delivery",
    "cushioned": "cushion cushioned foam",
    "comfortable": "comfortable comfort soft",
    "durable": "durable sturdy tough",
    "breathable": "breathable mesh",
    "warm": "warm insulated",
    "cozy": "cozy soft warm",
    "fragrance_free": "fragrance free unscented",
    "sensitive_skin": "sensitive skin gentle",
    "large_room": "large room sq ft coverage",
    "compact": "compact portable travel slim",
}


# ------------------------------------------------------------------ text utils
def _stem(tok: str) -> str:
    """Tiny plural stripper: earbuds->earbud, shoes->shoe. Good enough for a product catalog."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in ENGLISH_STOP_WORDS]


def _analyzer(text: str) -> list[str]:
    toks = _tokens(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


def _doc_text(p: ProcessedProduct) -> str:
    a = p.attributes
    return " ".join([
        p.title, p.title, a.product_type or "", p.category, p.description,
        " ".join(a.intended_use), " ".join(a.materials),
        " ".join(f.replace("_", " ") for f in a.features),
    ])


def _query_text(q: StructuredQuery) -> str:
    parts = [q.product_type or "", q.category or "", " ".join(q.intended_use), q.brand or ""]
    for tag in q.hard_requirements + q.soft_preferences:
        parts.append(_REQ_TERMS.get(tag, tag.replace("_", " ")))
    parts.append(q.raw_message)
    return " ".join(parts)


# -------------------------------------------------------------- type relevance
def type_relevance(q: StructuredQuery, p: ProcessedProduct) -> tuple[float, str]:
    """How much is p the kind of product the shopper asked for? Returns (score, reason)."""
    if q.product_type:
        qt = set(_tokens(q.product_type))
        pt = set(_tokens((p.attributes.product_type or "") + " " + p.title))
        if pt & {"runner", "trainer"}:          # "AquaRun Lightweight Runner" IS a running shoe
            pt |= {"running", "shoe"}
        if {"backpacking", "pack"} <= pt:       # "55L Backpacking Pack" IS a backpack (a tent is not)
            pt.add("backpack")
        # Match product identity using the type's head noun; descriptive modifiers
        # (e.g. wireless, waterproof, model names) must not downgrade the match.
        p_type_tokens = _tokens(p.attributes.product_type or "")
        query_tokens = _tokens(q.product_type)
        query_head = query_tokens[-1] if query_tokens else None

# Ignore numeric dimensions such as 6mm, 8mm, 40L, or 1TB.
        p_type_tokens = [
        token for token in p_type_tokens
             if not re.fullmatch(r"\d+(?:mm|cm|in|inch|inches|ft|gb|tb|l)", token)
]
        product_head = p_type_tokens[-1] if p_type_tokens else None
        # Treat common model suffixes as variants, not as the product type.
        model_suffixes = {"lite", "pro", "plus", "max", "mini", "se"}
        if product_head in model_suffixes and len(p_type_tokens) > 1:
            product_head = p_type_tokens[-2]
        if qt and qt <= pt and query_head == product_head:
            return 1.0, "same product type"
        if qt and qt <= pt and query_head == "shoe" and pt & {"runner", "trainer"}:
            return 1.0, "same product type"
        if qt == {"shoe"} and pt & _FOOTWEAR:
            return 0.9, "footwear (generic 'shoe' request)"
        # Shared HEAD noun: the query's head must also be the head (last word) of the product's own
        # type. "desk" in "desk lamp" is a modifier, so a lamp is not a desk.
        head = _tokens(q.product_type)[-1] if qt else None
        p_type_tokens = _tokens(p.attributes.product_type or "")
        p_head = p_type_tokens[-1] if p_type_tokens else None
        if head and head == p_head:
            return 0.7, f"shares head noun '{head}'"
        for fam in _FAMILIES:
            if qt & fam and pt & fam:
                return 0.6, "related product family"
        if q.category and p.category == q.category:
            return 0.3, "same category only"
        return 0.0, "different product type"
    if q.category:
        return (0.8, "category match") if p.category == q.category else (0.0, "different category")
    return 0.0, "no product type given"


# ------------------------------------------------------------------ result models
class Candidate(BaseModel):
    product: ProcessedProduct
    type_relevance: float
    type_reason: str
    text_similarity: float = Field(..., description="Raw TF-IDF cosine similarity")
    brand_match: bool = False
    retrieval_score: float


class RetrievalDiagnostics(BaseModel):
    n_candidates: int
    top_score: float
    margin: float = Field(..., description="Top score minus the 5th (or last) candidate's score")
    has_type_match: bool = Field(..., description="Any candidate with type_relevance >= 0.6")
    top_type_relevance: float
    max_similarity: float
    top5_categories: list[str]
    n_categories_top5: int


class RetrievalResult(BaseModel):
    candidates: list[Candidate]
    diagnostics: RetrievalDiagnostics


# --------------------------------------------------------------------- retriever
class Retriever:
    def __init__(self, products: list[ProcessedProduct]):
        self.products = products
        self._vec: Optional[TfidfVectorizer] = None
        self._matrix = None
        if products:
            self._vec = TfidfVectorizer(analyzer=_analyzer, sublinear_tf=True)
            self._matrix = self._vec.fit_transform([_doc_text(p) for p in products])

    def retrieve(self, query: StructuredQuery, k: int = DEFAULT_K) -> RetrievalResult:
        if not self.products:
            return RetrievalResult(candidates=[], diagnostics=_diagnostics([]))

        sims = cosine_similarity(self._vec.transform([_query_text(query)]), self._matrix).ravel()
        scored: list[Candidate] = []
        for p, sim in zip(self.products, sims):
            rel, reason = type_relevance(query, p)
            brand_match = bool(query.brand) and p.brand.lower() == query.brand.lower()
            score = W_TYPE * rel + W_TEXT * min(1.0, float(sim) * SIM_SCALE) + (BRAND_BOOST if brand_match else 0.0)
            score = min(1.0, score)
            if score >= MIN_SCORE:
                scored.append(Candidate(product=p, type_relevance=rel, type_reason=reason,
                                        text_similarity=round(float(sim), 4), brand_match=brand_match,
                                        retrieval_score=round(score, 4)))
        if any(c.type_relevance >= STRONG_TYPE for c in scored):
            # Real matches exist: same-category-only items (e.g. a tablet for "earbuds") are noise
            # unless their text strongly matches anyway.
            scored = [c for c in scored if c.type_relevance >= 0.5 or c.text_similarity >= 0.15]
        scored.sort(key=lambda c: (-c.retrieval_score, c.product.product_id))
        top = scored[:k]
        return RetrievalResult(candidates=top, diagnostics=_diagnostics(top))


def _diagnostics(cands: list[Candidate]) -> RetrievalDiagnostics:
    if not cands:
        return RetrievalDiagnostics(n_candidates=0, top_score=0.0, margin=0.0, has_type_match=False,
                                    top_type_relevance=0.0, max_similarity=0.0,
                                    top5_categories=[], n_categories_top5=0)
    ref = cands[min(4, len(cands) - 1)]
    cats = [c.product.category for c in cands[:5]]
    return RetrievalDiagnostics(
        n_candidates=len(cands),
        top_score=cands[0].retrieval_score,
        margin=round(cands[0].retrieval_score - ref.retrieval_score, 4),
        has_type_match=any(c.type_relevance >= STRONG_TYPE for c in cands),
        top_type_relevance=max(c.type_relevance for c in cands),
        max_similarity=max(c.text_similarity for c in cands),
        top5_categories=cats,
        n_categories_top5=len(set(cats)),
    )


# ------------------------------------------------------------------------ CLI
def main() -> None:
    from app.data.loader import load_catalog, load_processed
    from app.services.attribute_extractor import rules_extract
    from app.services.query_understanding import understand_query

    ap = argparse.ArgumentParser(description="Show retrieval candidates for a shopper message")
    ap.add_argument("message")
    ap.add_argument("-k", type=int, default=DEFAULT_K)
    args = ap.parse_args()

    products = load_processed() or [rules_extract(r) for r in load_catalog()]
    brands = sorted({p.brand for p in products})
    q = understand_query(args.message, known_brands=brands)
    res = Retriever(products).retrieve(q, k=args.k)

    print(f"query: {q.product_type!r} / {q.category!r} | hard={q.hard_requirements} soft={q.soft_preferences} "
          f"| budget<= {q.budget.max}")
    print(f"{'id':5} {'price':>7} {'type':>5} {'sim':>6} {'score':>6}  title")
    for c in res.candidates:
        print(f"{c.product.product_id:5} {c.product.price_usd or 0:7.2f} {c.type_relevance:5.2f} "
              f"{c.text_similarity:6.3f} {c.retrieval_score:6.3f}  {c.product.title}")
    print("\ndiagnostics:", res.diagnostics.model_dump())


if __name__ == "__main__":
    main()