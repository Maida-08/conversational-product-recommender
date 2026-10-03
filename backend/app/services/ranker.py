"""
Ranking: retrieval candidates -> explicitly scored, labelled, explained products.

score = sum(weight_i * component_i) / sum(weight_i), over the components that APPLY to this
query (no budget given -> no budget component; weights renormalise). Then explicit penalties
multiply the result. Every component, requirement state and penalty is returned, so the
response can state tradeoffs instead of hiding them.

Component weights (why they exist):
  type      0.25  wrong product type is the worst failure
  hard      0.30  explicit demands matter most; twice the budget weight on purpose, so a
                  product that MEETS the hard requirement but is slightly over budget beats
                  an in-budget product that does not
  budget    0.15  1.0 within budget, then linear decay; speed depends on firmness
  use_text  0.15  intended-use overlap + text similarity (fuzzy fit: "casual", "trail")
  soft      0.05  nice-to-haves only
  quality   0.05  shrunk rating (0.15 for "best/premium" queries)

Requirement states: met 1.0 | partial 0.3-0.8 | unknown 0.4 | unmet 0.0.
Unknown is deliberately NOT zero: silence in a description is not evidence of absence.

Labels come from states, not the score:
  strong   right type, every hard requirement met, no penalties (budget may be slightly over)
  partial  nothing explicitly failed, but something is unknown / partial / over budget / penalised
  poor     wrong product type, OR any hard requirement explicitly unmet (score x0.75, so it
           never outranks a product that fails nothing)
  conflict description says yes, but 2+ reviewers disagree (and outnumber supporters)

Review evidence (Phase 7) is combined with the description PER REQUIREMENT:
  description claims it ..... 2+ dissenting reviews -> CONFLICT (0.3) | one dissent -> soft warning only
  description is silent ..... 2+ negative -> unmet (0.1) | 1 negative -> partial (0.25, unverified)
                              1 positive -> partial (0.6) | 2+ positive -> met (0.8) | mixed -> unknown
A lone review never overturns an explicit description claim.
"""
from __future__ import annotations

import re
import statistics
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.models.schemas import Budget, ProcessedProduct, StructuredQuery
from app.services.attribute_extractor import _sentence_around
from app.services.retriever import Candidate
from app.services.review_analyzer import (ASPECT_PHRASES, TAG_TO_ASPECT, AspectSummary,
                                          ensure_review_evidence, summarize_aspect)

# ----------------------------------------------------------------- tunables
WEIGHTS = {"type": 0.25, "hard": 0.30, "budget": 0.15, "use_text": 0.15, "soft": 0.05, "quality": 0.05}
QUALITY_WEIGHT_PREMIUM = 0.15          # when the shopper asks for "best" / "premium"
UNKNOWN = 0.4                          # satisfaction of a requirement the data is silent on
BUDGET_TOLERANCE = {"soft": 1.0, "standard": 0.6, "firm": 0.25}   # overshoot fraction that reaches 0
SIM_SCALE = 2.5
PRIOR_MEAN, PRIOR_STRENGTH = 4.0, 50   # rating shrinkage: (r*n + 4.0*50) / (n + 50)
LOW_RATING = 3.8
LOW_RATING_PENALTY = 0.88
AUDIENCE_PENALTY = 0.80
POOR_DISCOUNT = 0.75                   # matches that explicitly fail a requirement sort below the rest
FOOTWEAR_LIGHT_G = (250.0, 320.0)      # per shoe: <=250 g light, <=320 g middling
KID_WORDS = re.compile(r"\b(kids?|child(?:ren)?|toddlers?|baby|babies)\b", re.I)
KID_RECIPIENTS = {"kid", "child", "son", "daughter", "niece", "nephew"}

State = Literal["met", "partial", "unmet", "unknown", "conflict"]


# ------------------------------------------------------------------ output models
class RequirementResult(BaseModel):
    requirement: str
    kind: Literal["hard", "soft"]
    state: State
    satisfaction: float
    note: str = Field(..., description="Short factual note (what the data says)")
    evidence: Optional[str] = Field(None, description="Verbatim description sentence, if any")
    source: Literal["description", "reviews", "description+reviews"] = "description"
    review_note: Optional[str] = Field(None, description="Soft review warning that did not change the state")
    review_quotes: list[str] = Field(default_factory=list, description="Verbatim review snippets used")


class ScoreComponent(BaseModel):
    name: str
    weight: float = Field(..., description="Normalised weight (applicable components sum to 1)")
    value: float
    contribution: float


class RankedProduct(BaseModel):
    product: ProcessedProduct
    rank: int
    score: float
    label: Literal["strong", "partial", "conflict", "poor"]
    retrieval_score: float
    components: list[ScoreComponent]
    requirements: list[RequirementResult]
    penalties: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    over_budget_by: Optional[float] = None


# ------------------------------------------------------------ requirement checks
def _human(tag: str) -> str:
    return tag.replace("_", " ")


def _item_grams(p: ProcessedProduct) -> Optional[float]:
    w = p.attributes.weight
    if w.grams is None:
        return None
    return w.grams / 2 if w.basis == "per_pair" else w.grams


_LIGHT_CLAIM = re.compile(r"light[\s-]?weight|ultra[\s-]?light|feather[\s-]?light|featherlite", re.I)


def _doc(p: ProcessedProduct) -> str:
    return " ".join([p.title, p.description, " ".join(f.replace("_", " ") for f in p.attributes.features)])


# tag -> regex that counts as positive TEXT evidence. Silence stays "unknown", never "unmet".
_TEXT_EVIDENCE = {
    "quiet": r"\bquiet\b|\d+\s*dB|\bsilent\b|sleep mode",
    "non_slip": r"non[\s-]?slip|anti[\s-]?slip|\bgrip\w*",
    "durable": r"\bdurable\b|\bsturdy\b|full[\s-]grain|cast iron|rotomolded|stainless|long[\s-]?lasting",
    "cozy": r"\bcoz[yi]\w*|\bsoft\b|plush|fleece|chunky[\s-]knit",
    "warm": r"\bwarm\b|insulated|rated to -?\d+\s*f",
    "breathable": r"breathab\w*|\bmesh\b|\bknit\b",
    "compact": r"compact|\bslim\b|folds? flat|packable|pocket|portable|stuffs into",
    "cushioned": r"cushion\w*|memory[\s-]foam|padded|plush",
    "comfortable": r"comfort\w*|cushion\w*|memory[\s-]foam|\bsoft\b",
    "fast_charging": r"fast[\s-]?charg\w*|\b\d+\s*w\b|power delivery",
    "fragrance_free": r"fragrance[\s-]?free|unscented",
    "sensitive_skin": r"sensitive skin",
}


def _text_state(tag: str, p: ProcessedProduct, pattern: str) -> RequirementResult:
    """Positive text evidence -> met (0.85). Silence -> unknown, never unmet."""
    rx = re.compile(pattern, re.I)
    m = rx.search(p.description)
    if m:   # evidence is a verbatim sentence from the description
        return RequirementResult(requirement=tag, kind="hard", state="met", satisfaction=0.85,
                                 note=f"description mentions '{m.group(0).strip()}'",
                                 evidence=_sentence_around(p.description, m.start(), m.end()))
    m = rx.search(p.title) or rx.search(" ".join(f.replace("_", " ") for f in p.attributes.features))
    if m:
        return RequirementResult(requirement=tag, kind="hard", state="met", satisfaction=0.85,
                                 note=f"listing mentions '{m.group(0).strip()}'")
    return RequirementResult(requirement=tag, kind="hard", state="unknown", satisfaction=UNKNOWN,
                             note="not stated in the description")


def _hours_stated(text: str) -> Optional[float]:
    vals = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)\s*[- ]?hours?", text, re.I)]
    vals += [float(n) * 24 for n in re.findall(r"(\d+(?:\.\d+)?)\s*[- ]?days?\b", text, re.I)]
    return max(vals) if vals else None


class CatalogStats(BaseModel):
    median_item_grams: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_products(cls, products: list[ProcessedProduct]) -> "CatalogStats":
        by_cat: dict[str, list[float]] = {}
        for p in products:
            g = _item_grams(p)
            if g:
                by_cat.setdefault(p.category, []).append(g)
        return cls(median_item_grams={c: statistics.median(v) for c, v in by_cat.items() if len(v) >= 3})


def evaluate_requirement(tag: str, p: ProcessedProduct, stats: CatalogStats) -> RequirementResult:
    a = p.attributes
    wr = a.water_resistance

    def res(state: State, sat: float, note: str, evidence: Optional[str] = None) -> RequirementResult:
        return RequirementResult(requirement=tag, kind="hard", state=state, satisfaction=sat,
                                 note=note, evidence=evidence)

    if tag == "waterproof":
        if wr.status is True:
            rating = f" ({wr.rating})" if wr.rating else ""
            if wr.level == "waterproof":
                return res("met", 1.0, f"described as waterproof{rating}", wr.evidence)
            if wr.level == "splash_resistant":
                return res("partial", 0.3, f"only splash resistant{rating}, not fully waterproof", wr.evidence)
            return res("partial", 0.6, f"water resistant{rating}, not described as fully waterproof", wr.evidence)
        if wr.status is False:
            return res("unmet", 0.0, "the description says it is not waterproof", wr.evidence)
        return res("unknown", UNKNOWN, "waterproofing is not stated in the description")

    if tag == "water_resistant":
        if wr.status is True:
            if wr.level == "splash_resistant":
                return res("partial", 0.6, f"splash resistant only ({wr.rating or 'no rating'})", wr.evidence)
            return res("met", 1.0, f"described as {wr.level.replace('_', ' ') if wr.level else 'water resistant'}", wr.evidence)
        if wr.status is False:
            return res("unmet", 0.0, "the description says it is not water resistant", wr.evidence)
        return res("unknown", UNKNOWN, "water resistance is not stated in the description")

    if tag == "lightweight":
        g = _item_grams(p)
        if g is not None:
            per = " per shoe" if a.weight.basis in ("per_shoe", "per_pair") else ""
            if p.category == "footwear":
                lo, mid = FOOTWEAR_LIGHT_G
                if g <= lo:
                    return res("met", 1.0, f"{g:.0f} g{per}: light")
                if g <= mid:
                    return res("partial", 0.7, f"{g:.0f} g{per}: not the lightest")
                return res("unmet", 0.1, f"{g:.0f} g{per}: on the heavy side")
            med = stats.median_item_grams.get(p.category)
            if med:
                ratio = g / med
                if ratio <= 0.75:
                    return res("met", 1.0, f"{g:.0f} g: lighter than most {p.category} items")
                if ratio <= 1.0:
                    return res("partial", 0.6, f"{g:.0f} g: about average for {p.category}")
                return res("unmet", 0.15, f"{g:.0f} g: heavier than most {p.category} items")
        m = _LIGHT_CLAIM.search(p.title + " " + p.description)
        if m:
            return res("met", 0.8, "described as lightweight, but no weight is given")
        return res("unknown", UNKNOWN, "weight is not stated")

    if tag == "wireless":
        if "wireless" in a.features:
            return res("met", 1.0, "wireless")
        if "wireless" in a.lacks:
            return res("unmet", 0.0, "the description says it is wired / not wireless")
        return res("unknown", UNKNOWN, "wireless connectivity is not stated")

    if tag == "noise_cancelling":
        if "active_noise_cancellation" in a.features:
            return res("met", 1.0, "has active noise cancellation")
        if "active_noise_cancellation" in a.lacks:
            return res("unmet", 0.0, "the description says it has no noise cancellation")
        return res("unknown", UNKNOWN, "noise cancellation is not stated")

    if tag == "gps":
        if "gps" in a.features:
            return res("met", 1.0, "built-in GPS")
        return res("unknown", UNKNOWN, "GPS is not mentioned")

    if tag == "cordless":
        if re.search(r"\bcordless\b", p.title + " " + p.description, re.I):
            return res("met", 1.0, "cordless")
        return res("unknown", UNKNOWN, "not stated to be cordless")

    if tag == "large_room":
        sq = [float(x.replace(",", "")) for x in re.findall(r"(\d[\d,]*)\s*sq\.?\s*ft", p.description, re.I)]
        if sq:
            best = max(sq)
            if best >= 500:
                return res("met", 1.0, f"covers rooms up to {best:.0f} sq ft")
            return res("unmet", 0.2, f"only covers rooms up to {best:.0f} sq ft")
        return res("unknown", UNKNOWN, "room coverage is not stated")

    if tag == "long_battery":
        h = _hours_stated(p.description)
        if h is None:
            return res("unknown", UNKNOWN, "battery life is not stated")
        if h >= 24:
            return res("met", 1.0, f"up to {h:.0f} hours of battery")
        if h >= 10:
            return res("partial", 0.6, f"up to {h:.0f} hours of battery")
        return res("unmet", 0.2, f"only about {h:.0f} hours of battery")

    if tag in _TEXT_EVIDENCE:
        return _text_state(tag, p, _TEXT_EVIDENCE[tag])

    # free-text requirement from the LLM: token overlap with the product text
    toks = [t for t in re.findall(r"[a-z]{4,}", tag.replace("_", " ").lower())]
    text = _doc(p).lower()
    if toks and sum(1 for t in toks if t[:5] in text) / len(toks) >= 0.5:
        return res("met", 0.7, f"description mentions {_human(tag)}")
    return res("unknown", UNKNOWN, f"{_human(tag)} is not stated in the description")


# ------------------------------------------------------------- review evidence
def _reviewers(n: int, m: int, phrase: str) -> str:
    return f"{n} of {m} reviewers {phrase}" if m else f"{n} reviewers {phrase}"


def _apply_reviews(r: RequirementResult, s: AspectSummary) -> RequirementResult:
    """Combine the description-based result with aggregated review evidence (see module docstring)."""
    if not (s.n_support or s.n_negative):
        return r
    pos_phrase, neg_phrase = ASPECT_PHRASES[s.aspect]
    neg = _reviewers(s.n_negative, s.n_reviews, neg_phrase)
    pos = _reviewers(s.n_support, s.n_reviews, pos_phrase)
    firm_negative = s.negative >= 2 and s.negative > s.support
    out = r.model_copy(deep=True)

    if r.state in ("met", "partial"):                     # the description makes a claim
        if firm_negative:
            out.state, out.satisfaction, out.source = "conflict", 0.3, "description+reviews"
            out.note = f"{r.note}, but {neg}"
            out.review_quotes = s.quotes_negative[:2]
        elif s.n_negative:                                # a lone dissent never overturns the claim
            out.review_note, out.source = neg, "description+reviews"
            out.review_quotes = s.quotes_negative[:1]
    elif r.state == "unknown":                            # the description is silent
        if firm_negative:
            out.state, out.satisfaction, out.source = "unmet", 0.1, "reviews"
            out.note, out.review_quotes = f"not stated in the description, and {neg}", s.quotes_negative[:2]
        elif s.n_negative and not s.n_support:
            out.state, out.satisfaction, out.source = "partial", 0.25, "reviews"
            out.note, out.review_quotes = f"not stated in the description; {neg} (unverified)", s.quotes_negative[:1]
        elif s.n_negative and s.n_support:
            out.source = "description+reviews"
            out.note = f"{r.note}; reviews are mixed ({pos}; {neg})"
            out.review_quotes = (s.quotes_negative[:1] + s.quotes_support[:1])
        elif s.support >= 2:
            out.state, out.satisfaction, out.source = "met", 0.8, "reviews"
            out.note, out.review_quotes = f"not stated in the description, but {pos}", s.quotes_support[:2]
        else:
            out.state, out.satisfaction, out.source = "partial", 0.6, "reviews"
            out.note, out.review_quotes = f"not stated in the description; {pos}", s.quotes_support[:1]
    return out


# ------------------------------------------------------------------- components
def budget_fit(price: Optional[float], b: Budget) -> tuple[Optional[float], Optional[float]]:
    """Return (fit 0..1, amount over budget). None fit when the shopper gave no budget."""
    if b.max is None and b.min is None:
        return None, None
    if price is None:
        return UNKNOWN, None
    tol = BUDGET_TOLERANCE[b.firmness]
    fit, over = 1.0, None
    if b.max is not None and price > b.max:
        fit = min(fit, max(0.0, 1.0 - (price / b.max - 1.0) / tol))
        over = round(price - b.max, 2)
    if b.min is not None and price < b.min:
        fit = min(fit, max(0.0, 1.0 - (1.0 - price / b.min) / tol))
    return fit, over


def shrunk_rating(p: ProcessedProduct) -> Optional[float]:
    if p.average_rating is None:
        return None
    n = p.num_reviews or 0
    return (p.average_rating * n + PRIOR_MEAN * PRIOR_STRENGTH) / (n + PRIOR_STRENGTH)


def quality_value(p: ProcessedProduct) -> float:
    r = shrunk_rating(p)
    return UNKNOWN if r is None else max(0.0, min(1.0, (r - 3.0) / 2.0))


def _use_text_value(q: StructuredQuery, c: Candidate) -> float:
    text_sim = min(1.0, c.text_similarity * SIM_SCALE)
    if not q.intended_use:
        return text_sim
    p = c.product
    pu = {u.lower() for u in p.attributes.intended_use}
    if not pu:
        return 0.5 * UNKNOWN + 0.5 * text_sim
    blob = (p.title + " " + p.description).lower()
    hit = sum(1 for u in q.intended_use if u.lower() in pu or u.lower() in blob)
    return 0.5 * (hit / len(q.intended_use)) + 0.5 * text_sim


def _audience_mismatch(q: StructuredQuery, p: ProcessedProduct) -> bool:
    if p.category == "toys":
        return False
    kid_product = bool(KID_WORDS.search(p.title)) or "kids" in p.attributes.intended_use
    if not kid_product:
        return False
    asked = (q.recipient in KID_RECIPIENTS) or "kids" in q.intended_use or bool(KID_WORDS.search(q.raw_message))
    return not asked


# ----------------------------------------------------------------------- ranker
class Ranker:
    def __init__(self, catalog: list[ProcessedProduct]):
        self.stats = CatalogStats.from_products(catalog)

    def rank(self, query: StructuredQuery, candidates: list[Candidate]) -> list[RankedProduct]:
        if not candidates:
            return []
        premium = bool({"best", "premium"} & set(query.tone_signals))
        cheap_only = "cheap" in query.tone_signals and query.budget.max is None and query.budget.min is None
        pool = [c.product.price_usd for c in candidates if c.type_relevance >= 0.6 and c.product.price_usd is not None]
        lo, hi = (min(pool), max(pool)) if pool else (None, None)

        ranked = [self._score(query, c, premium, cheap_only, lo, hi) for c in candidates]
        ranked.sort(key=lambda r: (-r.score, r.product.price_usd if r.product.price_usd is not None else 1e9,
                                   r.product.product_id))
        for i, r in enumerate(ranked, 1):
            r.rank = i
        return ranked

    # ------------------------------------------------------------------ one product
    def _score(self, q: StructuredQuery, c: Candidate, premium: bool, cheap_only: bool,
               lo: Optional[float], hi: Optional[float]) -> RankedProduct:
        p = c.product
        hard = [self._eval(t, p, "hard") for t in q.hard_requirements]
        soft = [self._eval(t, p, "soft") for t in q.soft_preferences]
        fit, over = budget_fit(p.price_usd, q.budget)

        comps: dict[str, float] = {"type": c.type_relevance}
        weights = {"type": WEIGHTS["type"]}
        if hard:
            comps["hard"] = sum(r.satisfaction for r in hard) / len(hard)
            weights["hard"] = WEIGHTS["hard"]
        if fit is not None:
            comps["budget"], weights["budget"] = fit, WEIGHTS["budget"]
        elif cheap_only and p.price_usd is not None and lo is not None:
            cheapness = 1.0 if hi == lo else 1.0 - (p.price_usd - lo) / (hi - lo)
            comps["budget"], weights["budget"] = cheapness, WEIGHTS["budget"]
        comps["use_text"], weights["use_text"] = _use_text_value(q, c), WEIGHTS["use_text"]
        if soft:
            comps["soft"] = sum(r.satisfaction for r in soft) / len(soft)
            weights["soft"] = WEIGHTS["soft"]
        comps["quality"] = quality_value(p)
        weights["quality"] = QUALITY_WEIGHT_PREMIUM if premium else WEIGHTS["quality"]

        total_w = sum(weights.values())
        components = [ScoreComponent(name=n, weight=round(weights[n] / total_w, 4), value=round(v, 4),
                                     contribution=round(weights[n] / total_w * v, 4))
                      for n, v in comps.items()]
        score = sum(x.contribution for x in components)

        penalties: list[str] = []
        sr = shrunk_rating(p)
        if sr is not None and sr < LOW_RATING:
            score *= LOW_RATING_PENALTY
            penalties.append(f"low rating: {p.average_rating:.1f} stars from {p.num_reviews} reviews")
        if _audience_mismatch(q, p):
            score *= AUDIENCE_PENALTY
            penalties.append("marketed for kids, and you did not mention a child")
        score = round(max(0.0, min(1.0, score)), 4)

        label = self._label(c.type_relevance, hard, fit, penalties)
        reasons, tradeoffs = self._explain(q, p, hard + soft, fit, over, comps, penalties, sr)
        if label == "poor":      # visible in `penalties`, not repeated in tradeoffs (the unmet note says why)
            score = round(score * POOR_DISCOUNT, 4)
            penalties = penalties + ["explicitly fails a requirement or is the wrong product type: score discounted"]
        return RankedProduct(product=p, rank=0, score=score, label=label, retrieval_score=c.retrieval_score,
                             components=components, requirements=hard + soft, penalties=penalties,
                             reasons=reasons, tradeoffs=tradeoffs, over_budget_by=over)

    def _eval(self, tag: str, p: ProcessedProduct, kind: str) -> RequirementResult:
        r = evaluate_requirement(tag, p, self.stats)
        r.kind = kind  # type: ignore[assignment]
        aspect = TAG_TO_ASPECT.get(tag)
        if aspect and p.review_evidence:
            mode = {"waterproof": "waterproof", "water_resistant": "resistant"}.get(tag)
            r = _apply_reviews(r, summarize_aspect(p.review_evidence, aspect, mode, len(p.review_snippets)))
        return r

    @staticmethod
    def _label(type_rel: float, hard: list[RequirementResult], fit: Optional[float],
               penalties: list[str]) -> str:
        if type_rel < 0.5:
            return "poor"
        if any(r.state == "unmet" for r in hard):
            return "poor"
        if any(r.state == "conflict" for r in hard):
            return "conflict"
        all_met = all(r.state == "met" for r in hard)
        if all_met and type_rel >= 0.9 and not penalties and (fit is None or fit >= 0.3):
            return "strong"
        return "partial"

    @staticmethod
    def _explain(q, p, reqs, fit, over, comps, penalties, sr) -> tuple[list[str], list[str]]:
        reasons, tradeoffs = [], []
        for r in reqs:
            line = f"{_human(r.requirement)}: {r.note}"
            (reasons if r.state == "met" else tradeoffs).append(line)
            if r.review_note and r.state == "met":        # soft warning from a lone dissenting review
                tradeoffs.append(f"{_human(r.requirement)}: {r.review_note}")
        if q.budget.max is not None:
            if over:
                tradeoffs.append(f"price ${p.price_usd:.2f} is ${over:.2f} over your ${q.budget.max:.0f} budget")
            elif p.price_usd is not None:
                reasons.append(f"within your ${q.budget.max:.0f} budget (${p.price_usd:.2f})")
        elif "cheap" in q.tone_signals and comps.get("budget", 0) >= 0.8:
            reasons.append(f"one of the lowest-priced options (${p.price_usd:.2f})")
        if sr is not None and sr >= 4.4 and (p.num_reviews or 0) >= 100:
            reasons.append(f"highly rated ({p.average_rating:.1f} stars from {p.num_reviews} reviews)")
        tradeoffs.extend(penalties)
        return reasons, tradeoffs


# ------------------------------------------------------------------------ CLI
def main() -> None:
    import argparse

    from app.data.loader import load_catalog, load_processed
    from app.services.attribute_extractor import rules_extract
    from app.services.query_understanding import understand_query
    from app.services.retriever import Retriever

    ap = argparse.ArgumentParser(description="Rank products for a shopper message")
    ap.add_argument("message")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("-v", "--verbose", action="store_true", help="show components, reasons, tradeoffs")
    args = ap.parse_args()

    products = ensure_review_evidence(load_processed() or [rules_extract(r) for r in load_catalog()])
    q = understand_query(args.message, known_brands=sorted({p.brand for p in products}))
    cands = Retriever(products).retrieve(q).candidates
    ranked = Ranker(products).rank(q, cands)[: args.k]

    print(f"query: {q.product_type!r} hard={q.hard_requirements} soft={q.soft_preferences} "
          f"budget<= {q.budget.max} ({q.budget.firmness}) tone={q.tone_signals}")
    print(f"{'#':>2} {'id':5} {'price':>7} {'score':>6} {'label':8} title")
    for r in ranked:
        print(f"{r.rank:2} {r.product.product_id:5} {r.product.price_usd or 0:7.2f} {r.score:6.3f} {r.label:8} {r.product.title}")
        if args.verbose:
            print("     components:", {c.name: c.value for c in r.components})
            for line in r.reasons:
                print("     + ", line)
            for line in r.tradeoffs:
                print("     - ", line)


if __name__ == "__main__":
    main()