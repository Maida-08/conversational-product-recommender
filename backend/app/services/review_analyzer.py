"""
Review evidence: review snippets -> per-aspect evidence -> aggregated, query-aware summaries.

Reviews are EVIDENCE, NOT TRUTH:
  * Each snippet is classified per aspect as 'supports' or 'negative' (neutral / irrelevant
    snippets produce no record). Hedged wording ("a bit", "slightly") counts half.
  * Evidence is aggregated by agreement: one review is a soft signal, two or more agreeing
    reviews are a firm one. The ranker (not this module) decides what that means for a
    requirement, because it depends on what the description claims.
  * Aggregation is query-aware: a heavy-downpour complaint does not count against a shopper
    who only asked for light-rain resistance.

Classification is deterministic (rules with negation handling) and works with no LLM. An
optional LLM pass can refine polarity / hedging for nuance the rules miss, but it is guarded:
the aspect must be in the vocabulary AND the snippet must contain an aspect keyword, so the
LLM cannot invent evidence.

Evidence record (stored in ProcessedProduct.review_evidence):
  {"aspect", "polarity", "hedged", "scope" (water only), "quote" (verbatim snippet), "source"}

Run (from backend/):
    python -m app.services.review_analyzer              # rules, plus LLM if a key is set
    python -m app.services.review_analyzer --rules-only
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Any, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import get_settings
from app.models.schemas import ProcessedProduct
from app.services.llm_client import LLMClient, LLMError

ASPECTS = ("water", "grip", "durability", "battery", "weight", "fragrance")

# Query requirement tag -> the review aspect that can confirm or contradict it.
TAG_TO_ASPECT = {
    "waterproof": "water", "water_resistant": "water", "non_slip": "grip", "durable": "durability",
    "long_battery": "battery", "lightweight": "weight", "fragrance_free": "fragrance",
}

# Short phrases used in notes: "2 of 3 reviewers <phrase>".
ASPECT_PHRASES = {
    "water": ("say it stayed dry", "report it let water in"),
    "grip": ("say it grips well", "report it is slippery"),
    "durability": ("say it is sturdy", "report it wore out or broke"),
    "battery": ("say the battery lasts", "report the battery falling short"),
    "weight": ("say it is light", "find it heavy"),
    "fragrance": ("say it has no scent", "report a noticeable smell"),
}

_HEDGE = re.compile(r"\b(?:a bit|a little|kind of|sort of|somewhat|slightly|mostly|fairly|mildly|a touch)\b", re.I)
_CLAUSE_SPLIT = re.compile(r"\s*(?:[;,]|\bbut\b|\bthough\b|\balthough\b|\bwhile\b)\s*", re.I)
_HEAVY = re.compile(r"heavy|downpour|storm|pouring|torrential|thunder", re.I)
_LIGHT = re.compile(r"light (?:rain|drizzle|shower)|drizzl|sprinkl|\bdamp\b|\bmist", re.I)


def _c(p: str) -> re.Pattern:
    return re.compile(p, re.I)


# aspect -> (gate, ordered rules). Gate = the clause must mention the topic at all. Rules are tried
# in order and the FIRST match wins, so negations ("without leaking") are listed before the
# words they contain ("leak"). A polarity of None means "mentioned, but no evidence".
_RULES: dict[str, tuple[re.Pattern, list[tuple[re.Pattern, Optional[str]]]]] = {
    "water": (
        _c(r"\bwater|\brain|drizzle|\bdry\b|\bwet\b|soak|damp|leak|puddle|\bstream|downpour|\bstorm|\bswim|\bpool|sweat|splash|thunder"),
        [
            (_c(r"\b(?:without|never|didn'?t|did not|no|zero)\b[^,.;]{0,20}\bleak\w*"), "supports"),
            (_c(r"\bnot\s+(?:great|good|ideal|waterproof|water[\s-]?resistant)\b"), "negative"),
            (_c(r"would(?:n'?t| not) wear[^,.;]*(?:\bwet|\brain)"), "negative"),
            (_c(r"waterproof claim is wrong|isn'?t (?:really )?waterproof|not waterproof"), "negative"),
            (_c(r"soak\w*|\bleak\w*|wet socks|feet (?:got |were |stayed |felt )?wet|got wet|gets? (?:wet|damp)|"
                r"lets? (?:in )?water|water (?:got|came|seeped) in"), "negative"),
            (_c(r"stayed (?:bone )?dry|kept (?:me|my \w+|your \w+)?\s*dry|dry feet|\band dry\b|bone dry|\bsurvived\b|"
                r"handled[^,.;]*\brain|(?:fine|worked fine|great|good|no (?:issues|problems?))[^,.;]*"
                r"(?:\brain|drizzle|downpour|\bstorm|\bswim|\bpool|\bwater)|swim\w*[^,.;]*(?:no (?:issues|problems?)|fine)|"
                r"through (?:a |the )?(?:full )?(?:downpour|storm|\brain|thunderstorm)|still works|held up[^,.;]*swim|"
                r"without any problem|rain cover works|no leaks?"), "supports"),
        ]),
    "grip": (
        _c(r"grip|slip|slick|traction|stays? put|slid"),
        [
            (_c(r"slippery|slips\b|\bslid\b|slick|no grip|poor (?:grip|traction)|slides around"), "negative"),
            (_c(r"grippy|good grip|great grip|excellent traction|\bgrips\b|stays? put|doesn'?t slip|non[\s-]?slip|"
                r"tires grip|solid grip"), "supports"),
        ]),
    "durability": (
        _c(r"durab|sturdy|\blast|wore out|wear|broke|snap|peel|fray|stopped working|\bdied\b|fell apart|fell off|"
           r"flimsy|tough|solid|crack|chip|\brip|tear|worn|build|shape|hold"),
        [
            (_c(r"stopped working|\bdied\b|wore out|wears? out|peel(?:ing|ed)?|snap(?:ped)?|frays?|broke\b|broken|"
                r"fell apart|flimsy|not durable|cracked|came apart|chipped|ripped|fell off"), "negative"),
            (_c(r"lasts? forever|very durable|\bdurable\b|sturdy|well[\s-]?made|tough|holds? (?:its )?shape|"
                r"solid build|built like|long[\s-]?lasting|holds up"), "supports"),
        ]),
    "battery": (
        _c(r"battery|runtime|runs? closer|\bcharges? (?:for|twice)|\d+\s*(?:hours?|minutes?|days?)"),
        [
            (_c(r"battery (?:only|barely|dies|drains|is closer)|runs? closer|"
                r"closer to[^,.;]{0,25}(?:hours?|minutes?|days?)|(?:only|barely)\b[^,.;]{0,25}(?:hours?|minutes?|days?)|"
                r"(?:doesn'?t|does not|won'?t) last|short battery|poor battery"), "negative"),
            (_c(r"battery (?:easily )?lasts?[^,.;]{0,15}week|great battery|excellent battery|long battery|"
                r"battery (?:life )?is (?:great|excellent|amazing)|lasts all day|days of battery"), "supports"),
        ]),
    "weight": (
        _c(r"heav|\blight\b|bulk|clunky|weigh|feather|barely feel|lugging"),
        [
            (_c(r"(?:too|a little|a bit|slightly|kind of)\s+light\b"), None),
            (_c(r"\bheavy\b(?!\s+(?:rain|downpour|storm|shower|duty|traffic))|too heavy|heavier|bulky|clunky|weighs a ton"), "negative"),
            (_c(r"light as|super light|\blight\b(?!\s+(?:rain|drizzle|up|blue))|lightweight|feather|barely feel|"
                r"light enough|easy to carry"), "supports"),
        ]),
    "fragrance": (
        _c(r"scent|smell|fragrance|odou?r|perfum|aroma"),
        [
            (_c(r"not overpowering"), None),
            (_c(r"strong (?:smell|scent|fragrance|odou?r)|smells? (?:odd|bad|strange|chemical|funny|strong)|"
                r"overpowering|chemical smell"), "negative"),
            (_c(r"no (?:scent|smell|fragrance|odou?r)|unscented|fragrance[\s-]?free|scent[\s-]?free|"
                r"doesn'?t smell|odou?rless"), "supports"),
        ]),
}


# ============================================================ snippet classification (rules)
def _water_scope(text: str) -> Optional[str]:
    heavy, light = bool(_HEAVY.search(text)), bool(_LIGHT.search(text))
    if heavy and not light:
        return "heavy"
    if light and not heavy:
        return "light"
    return None


def classify_snippet(snippet: str) -> list[dict[str, Any]]:
    """Classify one review snippet into zero or more per-aspect evidence records."""
    found: dict[str, list[tuple[str, bool]]] = {}
    for clause in (c for c in _CLAUSE_SPLIT.split(snippet) if c and c.strip()):
        hedged = bool(_HEDGE.search(clause))
        for aspect, (gate, rules) in _RULES.items():
            if not gate.search(clause):
                continue
            for pattern, polarity in rules:
                if pattern.search(clause):
                    if polarity:
                        found.setdefault(aspect, []).append((polarity, hedged))
                    break
    records = []
    for aspect, items in found.items():
        polarities = {p for p, _ in items}
        if len(polarities) > 1:       # contradicts itself inside one snippet: no usable evidence
            continue
        records.append({
            "aspect": aspect, "polarity": polarities.pop(), "hedged": all(h for _, h in items),
            "scope": _water_scope(snippet) if aspect == "water" else None,
            "quote": snippet, "source": "rules",
        })
    return records


def analyze_product_rules(p: ProcessedProduct) -> list[dict[str, Any]]:
    return [rec for snippet in p.review_snippets for rec in classify_snippet(snippet)]


# ============================================================ aggregation (query-aware)
class AspectSummary(BaseModel):
    aspect: str
    support: float = 0.0         # weighted: hedged = 0.5
    negative: float = 0.0
    n_support: int = 0
    n_negative: int = 0
    n_reviews: int = 0
    quotes_support: list[str] = Field(default_factory=list)
    quotes_negative: list[str] = Field(default_factory=list)


def summarize_aspect(evidence: list[dict[str, Any]], aspect: str, water_mode: Optional[str] = None,
                     n_reviews: int = 0) -> AspectSummary:
    """
    Aggregate evidence for one aspect.
    water_mode "waterproof": every negative counts; light-rain positives count half (weaker proof).
    water_mode "resistant":  heavy-downpour negatives are ignored (the shopper only needs light rain).
    """
    s = AspectSummary(aspect=aspect, n_reviews=n_reviews)
    for rec in evidence:
        if rec.get("aspect") != aspect:
            continue
        w = 0.5 if rec.get("hedged") else 1.0
        if rec.get("source") == "llm":                  # LLM-only evidence counts half until a rule agrees
            w *= LLM_WEIGHT
        scope = rec.get("scope")
        if aspect == "water":
            if rec["polarity"] == "negative" and water_mode == "resistant" and scope == "heavy":
                continue
            if rec["polarity"] == "supports" and water_mode == "waterproof" and scope == "light":
                w *= 0.5
        if rec["polarity"] == "supports":
            s.support += w
            s.n_support += 1
            s.quotes_support.append(rec["quote"])
        elif rec["polarity"] == "negative":
            s.negative += w
            s.n_negative += 1
            s.quotes_negative.append(rec["quote"])
    return s


def product_warnings(p: ProcessedProduct) -> list[str]:
    """Query-independent warnings, e.g. '2 of 3 reviewers report it let water in'."""
    out = []
    m = len(p.review_snippets)
    for aspect in ASPECTS:
        s = summarize_aspect(p.review_evidence, aspect, n_reviews=m)
        if s.n_negative:
            out.append(f"{s.n_negative} of {m} reviewers {ASPECT_PHRASES[aspect][1]}")
    return out


def ensure_review_evidence(products: list[ProcessedProduct], force: bool = False) -> list[ProcessedProduct]:
    """Fill review_evidence with rule-based evidence for products that have none."""
    for p in products:
        if force or not p.review_evidence:
            p.review_evidence = analyze_product_rules(p)
            p.review_warnings = product_warnings(p)
            if force:     # a forced rebuild starts from rules only, so the LLM pass may run again
                p.extraction.notes = [n for n in p.extraction.notes if n != _LLM_DONE]
    return products


# ============================================================ optional LLM refinement
# POLICY (learned from checking real Groq output against the rules):
#   * Rules WIN on any disagreement: they have explicit negation handling and our exact directions.
#   * The LLM only FILLS GAPS (snippets the rules found nothing in), and only for aspects whose
#     direction is unambiguous. weight ("light" vs "heavy") and fragrance (we mean "no scent") are
#     direction-sensitive: an LLM that reads "heavy but calming" as a plus, or "lovely scent" as
#     support, flips the meaning for a shopper who asked for lightweight or fragrance-free.
#   * LLM-only evidence counts HALF until a rule agrees (LLM_WEIGHT).
LLM_ASPECTS = ("water", "grip", "durability", "battery")
LLM_WEIGHT = 0.5
_LLM_DONE = "review evidence refined by LLM"


class _LLMAspect(BaseModel):
    model_config = ConfigDict(extra="ignore")
    aspect: str
    polarity: str
    hedged: bool = False
    scope: Optional[str] = None


class _LLMReview(BaseModel):
    model_config = ConfigDict(extra="ignore")
    i: int
    aspects: list[_LLMAspect] = Field(default_factory=list)


class _Client(Protocol):
    def complete_json(self, system: str, user: str, max_tokens: int = 3500) -> tuple[dict, dict]: ...


SYSTEM_PROMPT = ("You classify short product reviews. Use ONLY what each review actually says about the "
                 "product itself. Return ONLY one JSON object.")
USER_PROMPT_TEMPLATE = """For each numbered review, list which of these aspects it clearly comments on:
- water: "supports" = it kept water out / stayed dry; "negative" = it let water in / got soaked.
- grip: "supports" = it grips / does not slip; "negative" = it slips.
- durability: "supports" = it is sturdy / lasts; "negative" = it broke, wore out or peeled.
- battery: "supports" = battery life is good or as advertised; "negative" = battery life fell short.
Ignore remarks about something else: a cleanser that does not remove waterproof mascara is NOT about water
resistance, and a pan that eggs slide off is NOT about grip. Skip aspects a review does not mention.
Also give hedged true/false (hedged wording such as "a bit", "kind of") and, for water only, scope "light"
(drizzle / light rain), "heavy" (downpour / storm) or null.
Output: {{"reviews": [{{"i": <number>, "aspects": [{{"aspect": "...", "polarity": "...", "hedged": false, "scope": null}}]}}]}}

Reviews:
{reviews}"""


def refine_with_llm(products: list[ProcessedProduct], client: _Client, batch_size: int = 5,
                    log=print) -> int:
    """
    Gap-fill rule-based evidence with the LLM (see POLICY above). Guarded: the aspect must be in
    LLM_ASPECTS and the snippet must contain an aspect keyword. Returns the number of products processed.
    Never raises on LLM failure.
    """
    refined = 0
    todo = [p for p in products if p.review_snippets and _LLM_DONE not in p.extraction.notes]
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        index: list[tuple[ProcessedProduct, str]] = [(p, s) for p in batch for s in p.review_snippets]
        listing = "\n".join(f"{i + 1}. {s}" for i, (_, s) in enumerate(index))
        try:
            data, meta = client.complete_json(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(reviews=listing))
        except LLMError as e:
            log(f"  review LLM failed -> keeping rule-based evidence: {e}")
            return refined
        by_index: dict[int, _LLMReview] = {}
        for item in data.get("reviews", []):
            try:
                r = _LLMReview.model_validate(item)
                by_index[r.i] = r
            except ValidationError:
                continue

        added = ignored = 0
        for p in batch:
            existing = {(r["quote"], r["aspect"]): r for r in p.review_evidence}
            for i, (owner, snippet) in enumerate(index, 1):
                if owner is not p or i not in by_index:
                    continue
                for a in by_index[i].aspects:
                    if a.aspect not in LLM_ASPECTS or a.polarity not in ("supports", "negative"):
                        continue
                    if not _RULES[a.aspect][0].search(snippet):      # keyword gate: no invented evidence
                        continue
                    if (snippet, a.aspect) in existing:              # rules already decided: rules win
                        ignored += existing[(snippet, a.aspect)]["polarity"] != a.polarity
                        continue
                    scope = a.scope if a.aspect == "water" and a.scope in ("light", "heavy") else None
                    p.review_evidence.append({
                        "aspect": a.aspect, "polarity": a.polarity, "hedged": bool(a.hedged),
                        "scope": scope, "quote": snippet, "source": "llm"})
                    added += 1
            p.review_warnings = product_warnings(p)
            p.extraction.notes.append(_LLM_DONE)
            refined += 1
        log(f"  reviews [{', '.join(p.product_id for p in batch)}] via {meta.get('model')}: "
            f"{added} gap(s) filled, {ignored} disagreement(s) ignored (rules win)")
    return refined


# ================================================================================= CLI
SAMPLE_IDS = ["P003", "P006", "P007", "P024", "P080", "P049"]


def main() -> None:
    from app.data.loader import load_processed, save_processed

    ap = argparse.ArgumentParser(description="Add review evidence to processed_products.json")
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute rule-based evidence for every product")
    args = ap.parse_args()

    settings = get_settings()
    products = load_processed()
    if not products:
        raise SystemExit("processed_products.json is empty. Run: python -m app.services.attribute_extractor")

    ensure_review_evidence(products, force=args.force)
    if not args.rules_only and settings.llm_enabled:
        print(f"LLM refinement: {settings.llm_model}")
        n = refine_with_llm(products, LLMClient(settings))
        print(f"refined {n} products")
    else:
        print("rules-only evidence")

    save_processed(products)
    counts: dict[str, dict[str, int]] = {}
    for p in products:
        for r in p.review_evidence:
            counts.setdefault(r["aspect"], {"supports": 0, "negative": 0})[r["polarity"]] += 1
    print("evidence records by aspect:", counts)
    print(f"products with review warnings: {sum(1 for p in products if p.review_warnings)} / {len(products)}")

    samples = [{"product_id": p.product_id, "title": p.title, "review_snippets": p.review_snippets,
                "review_evidence": p.review_evidence, "review_warnings": p.review_warnings}
               for p in products if p.product_id in SAMPLE_IDS]
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    out = settings.outputs_dir / "review_evidence_samples.json"
    out.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()