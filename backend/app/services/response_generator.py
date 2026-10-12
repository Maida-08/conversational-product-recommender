"""
Response generation: structured decisions -> what the shopper reads.

Layers (the LLM can change WORDING, never FACTS):

  1. build_*_facts()   deterministic. Titles, prices and short natural-language clauses
                       ("is $9.50 over your $80 budget", "2 of 3 reviewers report it let water in").
                       Tradeoffs are written by CODE, so wording can never hide them.
  2. render()          deterministic template -> a natural reply. Always works (no API key,
                       rate limits, malformed output) and is the fallback.
  3. generate()        optional LLM rewrite from the facts alone, as JSON {"message": ...}.
  4. check_grounding() the guard. The LLM text is used ONLY if it passes: every $ amount comes
                       from the facts, no products/brands outside the facts, the primary and every
                       caution are named, the over-budget amount and review caveats appear, the
                       upsell comes AFTER the primary, a clarification asks exactly one question
                       and shows no prices, and there are no IDs, markdown, internal jargon or
                       runaway length. If anything fails, the template is used and the reasons
                       are recorded in guard_failures.

Try it (from backend/):
    python -m app.services.response_generator 'Looking for cheap wireless earbuds, just need something basic'
    python -m app.services.response_generator --llm 'Something nice for my mom'
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from app.models.schemas import ProcessedProduct, StructuredQuery
from app.services.ambiguity import GateDecision, _pronoun
from app.services.llm_client import LLMError
from app.services.ranker import RankedProduct, RequirementResult
from app.services.recommender import Recommendation
from app.services.upsell import UpsellDecision

MAX_WORDS = 130
MAX_WORDS_CLARIFY = 60
MAX_PLUSES, MAX_MINUSES = 3, 3

# ------------------------------------------------------------------ formatting
_NAMES = {
    "waterproof": "waterproof", "water_resistant": "water resistant", "lightweight": "lightweight",
    "wireless": "wireless", "noise_cancelling": "noise cancelling", "gps": "GPS", "quiet": "quiet",
    "non_slip": "non-slip", "durable": "durable", "long_battery": "long battery life",
    "large_room": "large-room coverage", "fragrance_free": "fragrance-free", "cordless": "cordless",
    "comfortable": "comfortable", "breathable": "breathable", "warm": "warm", "cozy": "cozy",
    "compact": "compact", "cushioned": "cushioned", "fast_charging": "fast charging",
}
_ADJECTIVES = {"waterproof", "water resistant", "lightweight", "wireless", "noise cancelling", "quiet",
               "non-slip", "durable", "fragrance-free", "cordless", "comfortable", "breathable", "warm",
               "cozy", "compact", "cushioned"}


def money(x: float) -> str:
    return f"${x:,.0f}" if float(x).is_integer() else f"${x:,.2f}"


def _name(tag: str) -> str:
    return _NAMES.get(tag, tag.replace("_", " "))


def _join(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _with_article(option: str) -> str:
    if option.endswith("s") and not option.endswith("ss"):
        return option                                    # dumbbells, resistance bands, trekking poles
    return ("an " if option[:1].lower() in "aeiou" else "a ") + option


def _options_text(options: Sequence[str]) -> str:
    items = [_with_article(o) for o in options]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + ", or " + items[-1]


# ------------------------------------------------------------ requirement -> clause
_STOP = {"which", "their", "there", "about", "doesn", "because", "though", "listing", "stated", "reviewers"}


def requirement_clause(r: RequirementResult) -> tuple[str, str, int, list[str]]:
    """-> (polarity 'plus'|'minus', predicate phrase after 'it', priority, tokens a reply must keep).
    The tokens are alternatives (any one satisfies the guard). They always include distinctive words
    from the clause itself, so the guard can never demand a word the honest wording does not contain."""
    polarity, text, prio, tokens = _clause(r)
    own = [w for w in re.findall(r"[a-z]{5,}", text.lower()) if w not in _STOP][:3]
    return polarity, text, prio, (tokens + own) if tokens else []


def _clause(r: RequirementResult) -> tuple[str, str, int, list[str]]:
    name = _name(r.requirement)
    note = r.note
    adj = name in _ADJECTIVES
    itis = f"it's {name}" if adj else name                  # "doesn't confirm it's waterproof"
    key = [name.split()[0].lower()]
    after = lambda sep: note.split(sep, 1)[1] if sep in note else note

    if r.state == "conflict":
        return "minus", f"is listed as {name}, yet {after(', but ')}", 0, ["review"]
    if r.state == "unmet":
        if r.source == "reviews":
            return "minus", f"draws complaints because {after(', and ')}", 0, ["review"]
        if r.requirement == "lightweight":
            weight, _, _ = note.partition(": ")
            return "minus", f"is on the heavy side ({weight})", 1, key
        if r.requirement == "wireless":
            return "minus", "is wired, not wireless", 1, key
        if r.requirement == "noise_cancelling":
            return "minus", "has no noise cancellation", 1, ["noise"]
        if r.requirement == "large_room":
            return "minus", note, 1, ["room", "sq ft"]
        return "minus", (f"isn't {name}" if adj else f"doesn't offer {name}"), 1, key
    if r.state == "partial":
        if r.source == "reviews":
            rest = after("; ").replace(" (unverified)", "")
            if "(unverified)" in note:
                return "minus", f"doesn't confirm {itis}, and {rest}", 2, ["review"]
            return "minus", f"doesn't say {itis} in the listing, though {rest}", 2, ["review"]
        if r.requirement == "lightweight":
            weight, _, verdict = note.partition(": ")
            return "minus", f"isn't the lightest ({weight})", 2, key
        if r.requirement == "waterproof":
            return "minus", "is water resistant but not fully waterproof", 2, key
        if r.requirement == "water_resistant":
            return "minus", "is only splash resistant", 2, key
        return "minus", note, 2, key
    if r.state == "unknown":
        if "mixed" in note:
            return "minus", f"has mixed reviews on {name}", 2, ["review"]
        return "minus", (f"doesn't say whether it's {name}" if adj else f"doesn't state its {name}"), 3, key
    # met
    if r.source == "reviews":
        return "plus", f"is {name}, according to reviewers (the listing doesn't say)", 1, []
    if r.requirement == "lightweight":
        if "no weight is given" in note:
            return "plus", "is described as lightweight", 1, []
        return "plus", f"is light ({note.partition(': ')[0]})", 0, []
    if r.requirement == "waterproof":
        return "plus", "is fully waterproof", 0, []
    if r.requirement in ("noise_cancelling",):
        return "plus", "has active noise cancellation", 0, []
    if r.requirement == "gps":
        return "plus", "has built-in GPS", 0, []
    if r.requirement == "large_room":                      # note: "covers rooms up to 700 sq ft"
        return "plus", note, 0, []
    if r.requirement == "long_battery":                    # note: "up to 30 hours of battery"
        return "plus", f"has {note}", 0, []
    return "plus", (f"is {name}" if adj else f"has {name}"), 0, []


def _penalty_clause(p: str) -> Optional[str]:
    m = re.match(r"low rating: ([\d.]+) stars from (\d+) reviews", p)
    if m:
        return f"is rated only {m.group(1)} stars ({m.group(2)} reviews)"
    if p.startswith("marketed for kids"):
        return "is marketed for kids"
    return None                                            # internal penalties are not shown to the shopper


def product_fact(r: RankedProduct, q: StructuredQuery, role: str) -> dict[str, Any]:
    plus: list[tuple[int, str]] = []
    minus: list[tuple[int, str]] = []
    critical: list[list[str]] = []
    for req in r.requirements:
        polarity, text, prio, tokens = requirement_clause(req)
        (plus if polarity == "plus" else minus).append((prio, text))
        if polarity == "minus" and tokens:
            critical.append(tokens)
        if req.review_note and req.state == "met":
            minus.append((2, f"has one caveat: {req.review_note}"))
            critical.append(["review"])
    cap = q.budget.max
    if r.over_budget_by:
        minus.insert(0, (1, f"is {money(r.over_budget_by)} over your {money(cap)} budget"))
    elif cap is not None and r.product.price_usd is not None:
        plus.append((2, f"fits your {money(cap)} budget"))
    for reason in r.reasons:
        if reason.startswith("highly rated"):
            plus.append((3, f"is {reason}"))
        elif reason.startswith("one of the lowest-priced"):
            plus.append((3, f"is {reason}"))
    for pen in r.penalties:
        text = _penalty_clause(pen)
        if text:
            minus.append((4, text))
    return {
        "role": role, "title": r.product.title, "price": r.product.price_usd,
        "pluses": [t for _, t in sorted(plus, key=lambda x: x[0])],
        "minuses": [t for _, t in sorted(minus, key=lambda x: x[0])],
        "over_budget_by": r.over_budget_by, "critical": critical, "label": r.label,
        "exact_type": next((c.value for c in r.components if c.name == "type"), 0.0) >= 0.9,
    }


# ------------------------------------------------------------------ facts builders
def _benefit_phrase(b: str) -> str:
    if b.startswith("adds "):
        return b[5:]
    if ": " in b:
        key, value = b.split(": ", 1)
        return f"better {key} ({value})"
    return b


def build_recommendation_facts(query: StructuredQuery, rec: Recommendation, upsell: UpsellDecision) -> dict[str, Any]:
    up = None
    if upsell.suggestion:
        s = upsell.suggestion
        up = {**product_fact(s.product, query, "upsell"), "price_delta": s.price_delta, "style": s.style,
              "benefits": [_benefit_phrase(b) for b in s.benefits]}
    b = query.budget
    return {
        "type": "recommendation", "shopper_message": query.raw_message, "mode": rec.mode,
        "requested_type": query.product_type,
        "budget": {"max": b.max, "min": b.min, "firmness": b.firmness} if (b.max or b.min) else None,
        "primary": product_fact(rec.primary, query, "primary"),
        "secondary": [product_fact(r, query, "secondary") for r in rec.secondary],
        "cautions": [product_fact(r, query, "caution") for r in rec.cautions],
        "upsell": up,
    }


def build_clarification_facts(query: StructuredQuery, gate: GateDecision) -> dict[str, Any]:
    if query.is_gift or (query.recipient and not query.category and not query.intended_use):
        kind = "gift"
    elif query.category:
        kind = "category"
    elif query.intended_use:
        kind = "use"
    else:
        kind = "generic"
    return {"type": "clarification", "kind": kind, "options": gate.options, "shopper_message": query.raw_message,
            "pronoun": _pronoun(query.recipient), "category": query.category,
            "use": query.intended_use[0] if query.intended_use else None}


def build_no_match_facts(
    query: StructuredQuery,
    nearest_titles: Sequence[str],
    reason: str = "missing_type",
) -> dict[str, Any]:
    return {
        "type": "no_match",
        "shopper_message": query.raw_message,
        "product_type": query.product_type,
        "nearest": list(nearest_titles)[:2],
        "reason": reason,
    }


# ------------------------------------------------------------------ template renderer
def _pred_join(preds: Sequence[str]) -> str:
    return _join(list(preds))


def render(facts: dict[str, Any]) -> str:
    t = facts["type"]
    if t == "clarification":
        opts = _options_text(facts["options"])
        kind = facts["kind"]
        if kind == "gift":
            return f"Happy to help! What kind of thing is {facts['pronoun']} into? I'm thinking {opts}."
        if kind == "category":
            return f"Happy to help with {facts['category']}! Are you after something specific, like {opts}?"
        if kind == "use":
            return f"For {facts['use']}, what are you looking for: {opts}?"
        return f"Happy to help! What kind of product are you looking for? For example {opts}."

    if t == "no_match":
        what = (
               f"a {facts['product_type']}"
               if facts.get("product_type")
               else "a product"
        )

        if facts.get("reason") == "constraints":
             return (
                f"I couldn't find {what} that meets all your requirements. "
                "Would you like to adjust your budget or requirements?"
             )

        first = (
            f"I don't have {what} in the catalog."
            if facts.get("product_type")
            else "I couldn't find anything that matches that."
        )

        near = facts.get("nearest") or []
        if near:
            names = " and ".join(f"the {n}" for n in near)
            return (
                f"{first} The closest things I have are {names}, "
                "but they're not really the same thing. "
                "Would you like me to look at something else?"
            )

        return (
            f"{first} Tell me a bit more about what you need "
            "and I'll look for the closest fit."
        )

    p = facts["primary"]
    parts: list[str] = []
    cautions = list(facts["cautions"])
    # The shopper asked for X, X has a review problem, and our pick is a different kind of product:
    # say so, instead of silently swapping one product type for another.
    literal = next((c for c in cautions if c.get("exact_type")), None) \
        if (facts["mode"] != "closest_available" and not p.get("exact_type") and facts.get("requested_type")) else None

    if literal and literal["minuses"]:
        what = _with_article(facts["requested_type"])
        parts.append(f"The {literal['title']} ({money(literal['price'])}) is the closest match for {what}, "
                     f"but it {literal['minuses'][0]}.")
        parts.append(f"If that's a dealbreaker, I'd look at the {p['title']} ({money(p['price'])}) instead.")
        cautions.remove(literal)
    else:
        lead = {"confident": f"I'd go with the {p['title']} ({money(p['price'])}).",
                "tradeoff": f"My top pick is the {p['title']} ({money(p['price'])}).",
                "closest_available": (f"I couldn't find a great match for that. The closest I have is the "
                                      f"{p['title']} ({money(p['price'])}).")}[facts["mode"]]
        parts.append(lead)
    if p["pluses"] and facts["mode"] != "closest_available":
        parts.append(f"It {_pred_join(p['pluses'][:MAX_PLUSES])}.")
    if p["minuses"]:
        opener = "Keep in mind that it" if facts["mode"] != "closest_available" else "The catch: it"
        parts.append(f"{opener} {_pred_join(p['minuses'][:MAX_MINUSES])}.")

    for s in facts["secondary"][:1]:                       # only if it gives the shopper a real reason to look
        cheaper = s["price"] is not None and p["price"] is not None and s["price"] <= p["price"] * 0.8
        fixes_budget = bool(p.get("over_budget_by")) and not s.get("over_budget_by")
        distinct = [x for x in s["pluses"] if x not in p["pluses"]]
        if not (fixes_budget or cheaper or distinct):
            continue
        angle = "stay within your budget" if fixes_budget else "spend less" if cheaper else "see another option"
        reasons = distinct[:1]
        bits = f"the {s['title']} ({money(s['price'])})"
        if reasons and s["minuses"]:
            bits += f" {reasons[0]}, though it {s['minuses'][0]}"
        elif reasons:
            bits += f" {reasons[0]}"
        elif s["minuses"]:
            bits += f", though it {s['minuses'][0]}"
        parts.append(f"If you'd rather {angle}, there's {bits}." if not reasons and not s["minuses"]
                     else f"If you'd rather {angle}, {bits}.")

    for c in cautions[:2]:
        if c["minuses"]:
            parts.append(f"Heads-up on the {c['title']} ({money(c['price'])}): it {c['minuses'][0]}.")

    u = facts.get("upsell")
    if u:
        delta, benefits = money(u["price_delta"]), _join(u["benefits"][:2])
        if u["style"] == "mention":
            parts.append(f"If you ever want {benefits}, the {u['title']} has that for {delta} more.")
        else:
            parts.append(f"You might also like the {u['title']} ({money(u['price'])}): it adds {benefits} "
                         f"for {delta} more.")
    return " ".join(parts)


# ------------------------------------------------------------------ the grounding guard
_ID = re.compile(r"\bP\d{3}\b")
_INTERNAL = re.compile(r"\b(score|label|ranking|retrieval|confidence|json|algorithm|hard requirement)\b", re.I)
_PRICE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)")
_MARKDOWN = re.compile(r"\*\*|^#{1,6}\s|^\s*[-*]\s", re.M)
_STYLE = re.compile(r"\b(explicitly|the facts)\b", re.I)          # talking about its own sources


def normalize_typography(text: str) -> str:
    """Models like typographic dashes and quotes; chat boxes (and our guard) prefer plain ASCII."""
    table = {"\u2011": "-", "\u2010": "-", "\u2013": " - ", "\u2014": " - ", "\u2018": "'", "\u2019": "'",
             "\u201c": '"', "\u201d": '"', "\u00a0": " "}
    for k, v in table.items():
        text = text.replace(k, v)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _allowed_amounts(facts: dict[str, Any]) -> set[float]:
    out: set[float] = set()
    prods = [facts.get("primary")] + facts.get("secondary", []) + facts.get("cautions", []) + [facts.get("upsell")]
    for p in prods:
        if p:
            if p.get("price") is not None:
                out.add(float(p["price"]))
            if p.get("over_budget_by"):
                out.add(float(p["over_budget_by"]))
    if facts.get("upsell"):
        out.add(float(facts["upsell"]["price_delta"]))
    b = facts.get("budget") or {}
    out |= {float(v) for v in (b.get("max"), b.get("min")) if v is not None}
    return out


def _has_amount(text: str, value: float) -> bool:
    return any(abs(float(m.replace(",", "")) - value) < 0.005 for m in _PRICE.findall(text))


def check_grounding(text: str, facts: dict[str, Any], catalog: Sequence[ProcessedProduct]) -> list[str]:
    """Return a list of violations. Empty means the text is safe to show."""
    v: list[str] = []
    low = text.lower()
    t = facts["type"]
    limit = MAX_WORDS_CLARIFY if t == "clarification" else MAX_WORDS
    if not text.strip():
        return ["empty reply"]
    if len(text.split()) > limit:
        v.append(f"too long ({len(text.split())} words, limit {limit})")
    if _ID.search(text):
        v.append("leaks an internal product id")
    if _INTERNAL.search(text):
        v.append("uses internal jargon")
    if _MARKDOWN.search(text):
        v.append("uses markdown / bullet formatting")
    if _STYLE.search(text):
        v.append("comments on its own sources instead of answering")

    shown_titles: list[str] = []
    if t == "recommendation":
        shown_titles = [facts["primary"]["title"]] + [x["title"] for x in facts["secondary"]] \
            + [x["title"] for x in facts["cautions"]] + ([facts["upsell"]["title"]] if facts["upsell"] else [])
    elif t == "no_match":
        shown_titles = facts.get("nearest", [])

    allowed = _allowed_amounts(facts)
    if t in ("clarification", "no_match"):
        if "$" in text:
            v.append("shows a price where none belongs")
    else:
        for m in _PRICE.findall(text):
            if not any(abs(float(m.replace(",", "")) - a) < 0.005 for a in allowed):
                v.append(f"price ${m} is not in the facts")

    brands_ok = {c.brand for c in catalog if any(c.title == s for s in shown_titles)}
    for c in catalog:
        if c.title.lower() in low and c.title not in shown_titles:
            v.append(f"mentions a product that is not part of the answer ({c.title})")
        if c.brand not in brands_ok and re.search(rf"\b{re.escape(c.brand)}\b", text):
            v.append(f"mentions a brand that is not part of the answer ({c.brand})")

    if t == "clarification":
        if text.count("?") != 1:
            v.append(f"must ask exactly one question (found {text.count('?')})")
        opts = facts["options"]
        need = min(2, len(opts))
        if sum(1 for o in opts if o.lower() in low) < need:
            v.append("does not offer the catalog options")
    elif t == "no_match":
        if facts.get("product_type") and not any(w in low for w in facts["product_type"].lower().split()):
            v.append("does not say which product is missing")
    else:
        p = facts["primary"]
        if p["title"].lower() not in low:
            v.append("does not name the primary pick")
        for c in facts["cautions"]:
            if c["title"].lower() not in low:
                v.append(f"drops the caution about {c['title']}")
        if p.get("over_budget_by") and not _has_amount(text, p["over_budget_by"]):
            v.append("hides how far over budget the pick is")
        body = low                                          # caveat words must appear OUTSIDE product titles,
        for title in shown_titles:                          # e.g. "Waterproof" in a title is not a caveat
            body = body.replace(title.lower(), " ")
        for prod in [p] + facts["cautions"]:
            scope = body
            if prod["role"] == "caution" and prod["title"].lower() in low:
                # a caution's caveat must sit NEXT TO that caution, not be borrowed from another sentence
                at = low.index(prod["title"].lower())
                scope = low[max(0, at - 150): at + len(prod["title"]) + 250]
                for title in shown_titles:
                    scope = scope.replace(title.lower(), " ")
            for tokens in prod.get("critical", []):
                if not any(tok in scope for tok in tokens):
                    v.append(f"omits a required caveat ({tokens[0]}) for {prod['title']}")
        u = facts.get("upsell")
        if u:
            if u["title"].lower() not in low:
                v.append("drops the upsell that was decided")
            elif low.index(u["title"].lower()) < low.index(p["title"].lower()):
                v.append("mentions the upsell before the primary pick")
            if not (_has_amount(text, u["price_delta"]) or _has_amount(text, u["price"])):
                v.append("upsell has no price")
    return v


# ------------------------------------------------------------------ LLM wording
SYSTEM_PROMPT = (
    "You are a friendly, concise shopping assistant. You are given FACTS about the answer. Write the reply "
    "shown to the shopper using ONLY those facts. Never invent products, prices, features or reviews. "
    "Plain text only, with plain ASCII punctuation: no markdown, no bullet points, no product ids, no internal "
    "terms such as score or label. Write like a helpful person: open with a natural sentence such as "
    "\"I'd go with the <title> (<price>).\", and never comment on the facts, the listing, or what is missing. "
    "Return ONLY a JSON object: {\"message\": \"<the reply>\"}."
)

_BRIEFS = {
    "recommendation": (
        "Write a warm reply of at most 110 words. Lead with the primary pick: use its EXACT title and price. "
        "Say plainly everything in primary.minuses (never hide a price overshoot or a review concern). "
        "If there are secondary options, mention the first briefly (what it fixes). "
        "For EVERY caution, name it exactly and give its warning in one sentence. "
        "If upsell is present, mention it in ONE brief sentence AFTER the main recommendation as an optional "
        "extra tied to its benefits and price_delta: never as a replacement, never before the primary pick. "
        "If mode is closest_available, say honestly that nothing matches well."),
    "clarification": (
        "Ask ONE short, friendly clarifying question (exactly one question mark) and offer these options "
        "(use their exact wording). Do not mention any product, price or recommendation."),
         "no_match": (
        "Explain honestly that no product satisfies the request. If reason is "
        "'constraints', explain that none meet all requirements and do not claim "
        "the product type is absent. If reason is 'missing_type', say the catalog "
        "has no such product. Nearest titles may be mentioned as loosely related, "
        "not recommendations. No prices. End with at most one question."),
       }



def llm_view(facts: dict[str, Any]) -> dict[str, Any]:
    """The facts the LLM may see: no internal labels, ids or guard metadata."""
    def clean(p):
        return {k: v for k, v in p.items() if k not in ("critical", "label", "over_budget_by")}
    out = {k: v for k, v in facts.items() if k not in ("primary", "secondary", "cautions", "upsell")}
    if facts["type"] == "recommendation":
        out["primary"] = clean(facts["primary"])
        out["secondary"] = [clean(x) for x in facts["secondary"]]
        out["cautions"] = [clean(x) for x in facts["cautions"]]
        out["upsell"] = clean(facts["upsell"]) if facts["upsell"] else None
    return out


class GeneratedResponse(BaseModel):
    text: str
    method: Literal["llm", "template"]
    guard_failures: list[str] = Field(default_factory=list, description="Why LLM wording was rejected")
    facts: dict[str, Any]


def generate(facts: dict[str, Any], client: Any = None, catalog: Sequence[ProcessedProduct] = ()) -> GeneratedResponse:
    template = render(facts)
    if client is None:
        return GeneratedResponse(text=template, method="template", facts=facts)
    try:
        user = (f"{_BRIEFS[facts['type']]}\n\nFACTS:\n{json.dumps(llm_view(facts), ensure_ascii=False)}")
        data, _meta = client.complete_json(SYSTEM_PROMPT, user, max_tokens=1500)
        text = normalize_typography(str(data.get("message", "")).strip().strip('"'))
    except (LLMError, AttributeError, TypeError, ValueError) as e:
        return GeneratedResponse(text=template, method="template", facts=facts,
                                 guard_failures=[f"LLM unavailable: {e}"])
    problems = check_grounding(text, facts, catalog)
    if problems:
        return GeneratedResponse(text=template, method="template", facts=facts, guard_failures=problems)
    return GeneratedResponse(text=text, method="llm", facts=facts)


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
    from app.services.upsell import plan_recommendation

    ap = argparse.ArgumentParser(description="Produce the shopper-facing reply for one message")
    ap.add_argument("message")
    ap.add_argument("--llm", action="store_true", help="use Groq for query understanding AND wording")
    ap.add_argument("--facts", action="store_true", help="also print the facts")
    args = ap.parse_args()

    settings = get_settings()
    products = ensure_review_evidence(load_processed() or [rules_extract(r) for r in load_catalog()])
    client = LLMClient(settings) if (args.llm and settings.llm_enabled) else None
    q = understand_query(args.message, client=client, known_brands=sorted({p.brand for p in products}))

    g = gate(q, products)
    if g.decision == "clarify":
        facts = build_clarification_facts(q, g)
    else:
        q = g.query
        retrieval = Retriever(products).retrieve(q)
        if match_check(q, retrieval).decision == "no_match":
            facts = build_no_match_facts(q, [c.product.title for c in retrieval.candidates[:2]])
        else:
            rec, up = plan_recommendation(q, Ranker(products).rank(q, retrieval.candidates))
            facts = build_recommendation_facts(q, rec, up)

    out = generate(facts, client=client, catalog=products)
    print(f"[{facts['type']} | wording: {out.method}]\n")
    print(out.text)
    if out.guard_failures:
        print("\nguard:", out.guard_failures)
    if args.facts:
        print("\n" + json.dumps(llm_view(facts), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()