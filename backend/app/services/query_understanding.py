"""
Query understanding: free-text shopper message -> StructuredQuery.

Design: rules own what is reliable; the LLM covers phrasing the rules do not know,
under guardrails (same pattern as attribute extraction).

  Rules ............. budget, hard vs soft (by clause), product type/category (synonym
                      table), requirement tags, intended use, tone, gift/recipient, brand
  LLM (optional) .... fills gaps only. Every LLM claim must be supported by the message
                      (words present / number present) or it is dropped and noted.

The query parser also records lexical `vagueness_signals`. Deciding whether to ASK a
clarifying question happens later, combining these with retrieval strength.

Multi-turn: `history` is accepted and reserved; merging earlier constraints is added later.

Try it (from backend/):
    python -m app.services.query_understanding 'I need a lightweight waterproof running shoe, ideally under $80'
    python -m app.services.query_understanding --rules-only 'something nice for my mom'
(Use SINGLE quotes in PowerShell when the message contains a $ sign.)
"""
from __future__ import annotations

import argparse
import re
from typing import Any, Optional, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import get_settings
from app.models.schemas import Budget, StructuredQuery
# Reuse the product-side vocabulary so query tags and product tags line up.
from app.services.attribute_extractor import _USES, extract_lexicon
from app.services.llm_client import LLMClient, LLMError

KNOWN_CATEGORIES = ("footwear", "electronics", "home", "beauty", "sports",
                    "kitchen", "outdoor", "toys", "pets", "office")

# ================================================================= BUDGET
_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
# A number followed by one of these is a measurement/count, not a price.
_NOT_MONEY = (r"(?!\s*(?:lbs?|pounds?|oz|ounces?|kg|g\b|grams?|inch(?:es)?|in\b|hours?|hrs?|"
              r"minutes?|mins?|mm|cm|l\b|liters?|qt|quarts?|gb|tb|mah|w\b|watts?|%|years?|yrs?|"
              r"ft|feet|pieces?|pcs?|pairs?|reviews?|stars?|ratings?|options?|items?|results?|people|kids))")
_CAP_WORDS = (r"(?:under|below|less than|cheaper than|max(?:imum)?(?:\s+of)?|up to|no more than|"
              r"not more than|at most|within|no higher than|not (?:over|above|exceeding)|nothing over|"
              r"(?:can'?t|cannot|won'?t|don'?t want to|do not want to)\s+(?:spend|pay|go)\s+(?:more than|over|above))")
_MIN_WORDS = r"(?:over|above|more than|at least|starting at|minimum(?:\s+of)?)"

_RANGE_BETWEEN = re.compile(rf"\bbetween\s+\$?\s*(?P<a>{_NUM})\s*(?:and|-|–|to)\s*\$?\s*(?P<b>{_NUM}){_NOT_MONEY}", re.I)
_RANGE_DOLLAR = re.compile(rf"\$\s*(?P<a>{_NUM})\s*(?:-|–|to)\s*\$?\s*(?P<b>{_NUM})", re.I)
_CAP = re.compile(rf"\b{_CAP_WORDS}\s*(?:about\s+|around\s+|roughly\s+)?(?P<dollar>\$\s*)?(?P<num>{_NUM}){_NOT_MONEY}", re.I)
_MIN = re.compile(rf"\b{_MIN_WORDS}\s*(?:\$\s*(?P<num>{_NUM})|(?P<num2>{_NUM})\s*(?:dollars|bucks|usd)\b)", re.I)
_BARE_DOLLAR = re.compile(
    rf"(?P<approx>(?:around|about|roughly|approximately|~)\s*)?(?:\$\s*(?P<num>{_NUM})|(?P<num2>{_NUM})\s*(?:dollars|bucks|usd)\b)", re.I)

_SOFT_MARKERS = re.compile(
    r"\b(ideally|preferably|prefer|if possible|would be nice|nice to have|hopefully|maybe|optionally|bonus)\b", re.I)
_BUDGET_APPROX = re.compile(r"\b(around|about|roughly|approximately|ish|or so)\b|~", re.I)
_BUDGET_FIRM = re.compile(
    r"\b(must|can'?t|cannot|no more than|not more than|at most|max(?:imum)?|strict|hard limit|"
    r"don'?t want to spend more|not over|nothing over|no higher than)\b", re.I)

_STRICT_BUDGET_LANGUAGE = re.compile(
    r"\b(?:do\s+not|don't|dont|never)\s+"
    r"(?:recommend|suggest|show|include|offer)\b"
    r".{0,100}\b(?:above|over|beyond|exceed(?:ing)?)\b"
    r".{0,40}\b(?:my\s+)?budget\b"
    r"|\b(?:nothing|anything)\s+(?:above|over|beyond)\s+(?:my\s+)?budget\b",
    re.I,
)
def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def _clause_spans(text: str) -> list[tuple[int, int]]:
    """Split on punctuation / 'but' so a marker like 'ideally' only affects its own clause."""
    splitter = re.compile(r"[;!?]|,(?!\d)|\.(?=\s|$)|\b(?:but|though|although|however)\b", re.I)
    spans, start = [], 0
    for m in splitter.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def _clause_text(text: str, pos: int) -> str:
    for s, e in _clause_spans(text):
        if s <= pos <= e:
            return text[s:e]
    return text


def _normalize(text: str) -> str:
    """Curly apostrophes/quotes (phones, Word) -> straight, so "can’t" matches "can't"."""
    return text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def parse_budget(text: str) -> Budget:
    text = _normalize(text)
    b = Budget()
    pos = None

    m = _RANGE_BETWEEN.search(text) or _RANGE_DOLLAR.search(text)
    if m:
        lo, hi = sorted((_to_float(m.group("a")), _to_float(m.group("b"))))
        b.min, b.max, pos = lo, hi, m.start()
    else:
        cap = _CAP.search(text)
        if cap:
            val = _to_float(cap.group("num"))
            if cap.group("dollar") or val >= 5:      # bare small numbers ("under 2") are not prices
                b.max, pos = val, cap.start()
        m = _MIN.search(text)
        # "can't spend MORE THAN $60" is a cap, not a minimum: skip a min match inside the cap phrase
        inside_cap = bool(cap and b.max is not None and cap.start() <= m.start() < cap.end()) if m else False
        if m and not inside_cap:
            b.min = _to_float(m.group("num") or m.group("num2"))
            pos = pos if pos is not None else m.start()
        if b.max is None and b.min is None:
            m = _BARE_DOLLAR.search(text)
            if m:
                b.max = _to_float(m.group("num") or m.group("num2"))
                pos = m.start()

    
    if pos is not None:
       clause = _clause_text(text, pos)
       if _SOFT_MARKERS.search(clause) or _BUDGET_APPROX.search(clause):
           b.firmness = "soft"
       elif _BUDGET_FIRM.search(clause):
            b.firmness = "firm"

# An explicit prohibition can be in a different sentence from the price.
    if b.max is not None and _STRICT_BUDGET_LANGUAGE.search(text):
       b.firmness = "firm"

    return b


# ===================================================== PRODUCT TYPE / CATEGORY
# Order matters: specific patterns first. This table is tuned to the development catalog;
# on a real catalog it would be derived from catalog product types (see README limitations).

("over-ear headphones", "electronics",
 r"\bover[\s-]?ear\b.*\bheadphones?\b|\bheadphones?\b.*\bover[\s-]?ear\b"),
_PRODUCT_TYPES: list[tuple[str, str, str]] = [
    # footwear
    ("trail running shoe", "footwear", r"trail[\s-]running\s+shoes?|trail\s+runners?"),
    ("running shoe", "footwear", r"running\s+(?:shoes?|sneakers?)|\brunners?\b|\btrainers?\b|jogging\s+shoes?"),
    ("snow boot", "footwear", r"(?:snow|winter)\s+boots?"),
    ("hiking shoe", "footwear", r"hiking\s+(?:shoes?|boots?)"),
    ("sandal", "footwear", r"\bsandals?\b"),
    ("sneaker", "footwear", r"\bsneakers?\b"),
    ("boot", "footwear", r"\bboots?\b"),
    ("walking shoe", "footwear", r"walking\s+shoes?|\bloafers?\b|slip[\s-]on"),
    ("shoe", "footwear", r"\bshoes?\b|\bfootwear\b"),
    # electronics
    ("earbuds", "electronics", r"\bear\s?buds?\b|\bearphones?\b|\bairpods?\b|in[\s-]ear"),
    ("over-ear headphones", "electronics",
    r"\bover[\s-]?ear\b.*\bheadphones?\b|\bheadphones?\b.*\bover[\s-]?ear\b"),
    ("power bank", "electronics", r"power\s?banks?|portable\s+charger|battery\s+pack"),
    ("speaker", "electronics", r"\bspeakers?\b"),
    ("fitness tracker", "electronics", r"fitness\s+(?:tracker|band)|\bsmart\s?watch\b|activity\s+tracker"),
    ("tablet", "electronics", r"\btablets?\b"),
    ("monitor", "electronics", r"\bmonitors?\b"),
    # home
    ("blanket", "home", r"\bblankets?\b|\bthrows?\b"),
    ("pillow", "home", r"\bpillows?\b"),
    ("diffuser", "home", r"\bdiffusers?\b"),
    ("lamp", "home", r"\blamps?\b"),
    ("candle", "home", r"\bcandles?\b"),
    ("air purifier", "home", r"air\s+purifiers?|\bpurifier\b"),
    ("vacuum", "home", r"\bvacuums?\b"),
    ("storage basket", "home", r"storage\s+baskets?|\bbaskets?\b"),
    ("towel set", "home", r"\btowels?\b"),
    # kitchen
    ("kettle", "kitchen", r"\bkettles?\b"),
    ("blender", "kitchen", r"\bblenders?\b"),
    ("air fryer", "kitchen", r"air[\s-]?fryers?"),
    ("skillet", "kitchen", r"\bskillets?\b|frying\s+pan|cast[\s-]iron"),
    ("pan set", "kitchen", r"\bpans?\b|cookware|nonstick"),
    ("knife", "kitchen", r"\bknife\b|\bknives\b"),
    ("food containers", "kitchen", r"food\s+(?:storage|containers?)|tupperware|meal[\s-]prep\s+containers?"),
    ("pour-over coffee set", "kitchen", r"pour[\s-]over|coffee\s+(?:set|maker|dripper)"),
    ("silicone bags", "kitchen", r"silicone\s+bags?|reusable\s+bags?"),
    # beauty
    ("night cream", "beauty", r"night\s+cream"),
    ("moisturizer", "beauty", r"moisturi[sz]ers?|face\s+cream"),
    ("sunscreen", "beauty", r"sunscreen|\bspf\b|sun\s?block"),
    ("serum", "beauty", r"\bserum\b"),
    ("shampoo", "beauty", r"\bshampoo\b"),
    ("hair dryer", "beauty", r"hair\s?dryers?|blow\s?dryers?"),
    ("lip balm", "beauty", r"lip\s+balms?"),
    ("lipstick", "beauty", r"\blipsticks?\b"),
    ("cleanser", "beauty", r"cleansers?|face\s+wash"),
    ("perfume", "beauty", r"\bperfumes?\b|eau de parfum|\bcologne\b"),
    ("gift set", "beauty", r"(?:gift|spa)\s+sets?"),
    # sports
    ("yoga mat", "sports", r"yoga\s+mats?"),
    ("dumbbells", "sports", r"\bdumbbells?\b"),
    ("resistance bands", "sports", r"resistance\s+bands?"),
    ("swim goggles", "sports", r"\bgoggles\b"),
    ("basketball", "sports", r"\bbasketballs?\b"),
    ("foam roller", "sports", r"foam\s+rollers?"),
    ("helmet", "sports", r"\bhelmets?\b"),
    ("water bottle", "sports", r"water\s+bottles?|\bbottles?\b|\bflask\b|\btumbler\b"),
    ("running belt", "sports", r"running\s+belt"),
    ("jump rope", "sports", r"jump\s+rope|skipping\s+rope"),
    ("running shorts", "sports", r"\bshorts\b"),
    # outdoor
    ("backpack", "outdoor", r"\bbackpacks?\b|\brucksack\b|\bdaypack\b|\bpack\b"),
    ("tent", "outdoor", r"\btents?\b"),
    ("sleeping bag", "outdoor", r"sleeping\s+bags?"),
    ("rain jacket", "outdoor", r"rain\s+(?:jacket|coat)|\braincoat\b|\bjackets?\b"),
    ("trekking poles", "outdoor", r"(?:trekking|hiking)\s+poles?"),
    ("water filter", "outdoor", r"water\s+(?:filter|purifier)"),
    ("camp stove", "outdoor", r"\bstoves?\b"),
    ("headlamp", "outdoor", r"\bheadlamps?\b|head\s?torch|\bflashlight\b"),
    ("lantern", "outdoor", r"\blanterns?\b"),
    ("sun hat", "outdoor", r"sun\s+hat|\bhats?\b"),
    ("cooler", "outdoor", r"\bcoolers?\b"),
    # toys
    ("building set", "toys", r"building\s+(?:set|bricks?|blocks?)|\blego\b|\bbricks\b"),
    ("plush toy", "toys", r"\bplush\b|stuffed\s+animal|\bteddy\b"),
    ("puzzle", "toys", r"\bpuzzles?\b"),
    ("stem kit", "toys", r"stem\s+(?:kit|toy)|robot\s+kit|\brobotics\b"),
    ("water blaster", "toys", r"water\s+(?:blaster|gun)|super\s?soaker"),
    ("stacking blocks", "toys", r"stacking\s+blocks|wooden\s+blocks"),
    ("rc car", "toys", r"rc\s+car|remote[\s-]control\s+car|stunt\s+car"),
    ("board game", "toys", r"board\s+games?|party\s+game"),
    ("painting set", "toys", r"(?:painting|paint)\s+set|art\s+(?:set|supplies)"),
    ("night light", "toys", r"night\s?light|\bprojector\b"),
    ("play kitchen", "toys", r"play\s+kitchen"),
    ("toy", "toys", r"\btoys?\b"),
    # one-offs
    ("dog bed", "pets", r"dog\s+bed|pet\s+bed"),
    ("laptop stand", "office", r"laptop\s+stands?"),
]
_PRODUCT_TYPES_C = [(pt, cat, re.compile(p, re.I)) for pt, cat, p in _PRODUCT_TYPES]

# Category-only mentions ("skincare for mom") when no product type matched.
_CATEGORY_WORDS = {
    "beauty": r"skin\s?care|makeup|\bbeauty\b|cosmetics",
    "kitchen": r"\bkitchen\b|cookware|cooking",
    "outdoor": r"\boutdoors?\b|camping\s+gear|hiking\s+gear",
    "toys": r"\btoys\b",
    "electronics": r"electronics?|gadgets?",
    "sports": r"\bsports?\b|fitness|workout\s+gear|gym\s+gear",
    "home": r"home\s+decor|home\s+goods|household|\bdecor\b",
    "pets": r"\bpets?\b",
}


def detect_product_type(text: str) -> tuple[Optional[str], Optional[str]]:
    for ptype, cat, pat in _PRODUCT_TYPES_C:
        if pat.search(text):
            return ptype, cat
    return None, None


def detect_category_word(text: str) -> Optional[str]:
    for cat, pat in _CATEGORY_WORDS.items():
        if re.search(pat, text, re.I):
            return cat
    return None


# ============================================================ REQUIREMENT TAGS
_REQUIREMENTS: list[tuple[str, str]] = [
    ("waterproof", r"water[\s-]?proof|rain[\s-]?proof|keeps?\s+(?:my\s+|your\s+)?feet\s+dry"),
    ("water_resistant", r"water[\s-]?resist\w*|splash[\s-]?proof|sweat[\s-]?(?:proof|resist\w*)|\bsweat\b|"
                        r"\b(?:light\s+|heavy\s+)?rain\b(?!\s+(?:jacket|coat|cover|fly|gear))|drizzle|wet\s+weather|puddles?"),
    ("lightweight", r"light[\s-]?weight|ultra[\s-]?light|feather[\s-]?light|"
                    r"\blight\b(?!\s+(?:rain|drizzle|up|blue|green|pink|gr[ae]y))"),
    ("wireless", r"wireless|bluetooth"),
    ("cordless", r"\bcordless\b"),
    ("noise_cancelling", r"noise[\s-]?cancel\w*|\banc\b"),
    ("gps", r"\bgps\b"),
    ("non_slip", r"non[\s-]?slip|anti[\s-]?slip|slip[\s-]?resistant"),
    ("quiet", r"\bquiet\b|\bsilent\b|low[\s-]noise"),
    ("long_battery", r"long[\s-]?(?:lasting\s+)?battery|battery\s+life|all[\s-]day\s+battery"),
    ("fast_charging", r"fast[\s-]?charg\w*"),
    ("cushioned", r"cushion\w*"),
    ("comfortable", r"comfort\w*|\bcomfy\b"),
    ("durable", r"durable|long[\s-]?lasting|won'?t\s+fall\s+apart|built\s+to\s+last|\bsturdy\b"),
    ("breathable", r"breathab\w*"),
    ("warm", r"\bwarm\b|insulated"),
    ("cozy", r"\bcoz[yi]\w*|\bcosy\b|snuggl\w*"),
    ("fragrance_free", r"fragrance[\s-]?free|unscented|no\s+scent"),
    ("sensitive_skin", r"sensitive\s+skin"),
    ("large_room", r"large\s+(?:open[\s-]plan\s+)?(?:room|space|living\s+room)|open[\s-]plan"),
    ("compact", r"compact|portable|travel[\s-]size|pocket[\s-]?size"),
]
_REQUIREMENTS_C = [(tag, re.compile(p, re.I)) for tag, p in _REQUIREMENTS]


def detect_requirements(text: str) -> tuple[list[str], list[str]]:
    """Return (hard, soft). A soft marker ('ideally', 'prefer'...) softens its own clause only."""
    hard, soft = [], []
    for tag, pat in _REQUIREMENTS_C:
        m = pat.search(text)
        if not m:
            continue
        (soft if _SOFT_MARKERS.search(_clause_text(text, m.start())) else hard).append(tag)
    return hard, soft


# ===================================================================== TONE ETC.
_TONE = {
    "cheap": r"\bcheap(?:est)?\b|inexpensive|budget[\s-]friendly|\baffordable\b|low[\s-]cost",
    "basic": r"\bbasic\b|\bsimple\b|no[\s-]frills|nothing\s+fancy",
    "premium": r"\bpremium\b|high[\s-]end|top[\s-]of[\s-]the[\s-]line|\bluxury\b|\bprofessional\b",
    "best": r"\bbest\b|top[\s-]rated|highest[\s-]rated",
}
_RECIPIENT = re.compile(
    r"\bfor\s+(?:my|our|a|an|the)?\s*(mom|mother|mum|dad|father|wife|husband|girlfriend|boyfriend|partner|"
    r"sister|brother|friend|boss|colleague|coworker|teacher|grandma|grandmother|grandpa|grandfather|"
    r"son|daughter|kid|child|niece|nephew|aunt|uncle|neighbou?r|dog|cat)\b", re.I)
_GIFT = re.compile(r"\bgift\w*|\bpresent\b|birthday|anniversary|christmas|mother'?s day|father'?s day|"
                   r"valentine|wedding|housewarming|surprise", re.I)
_VAGUE_WORDS = re.compile(
    r"\b(nice|good|great|cool|something|anything|stuff|things?|special|cute|awesome|decent|interesting|perfect)\b", re.I)


def _vagueness(q: StructuredQuery) -> list[str]:
    signals = [f"vague_word:{w}" for w in sorted({m.group(1).lower() for m in _VAGUE_WORDS.finditer(q.raw_message)})]
    has_type = bool(q.product_type or q.category)
    has_attrs = bool(q.hard_requirements or q.soft_preferences or q.intended_use or q.brand)
    has_budget = q.budget.max is not None or q.budget.min is not None
    if not has_type:
        signals.append("no_product_type")
    if not (has_attrs or has_budget):
        signals.append("no_concrete_attributes")
    if q.recipient and not has_type and not has_attrs:
        signals.append("recipient_without_interest")
    return signals


def rules_parse(message: str, known_brands: Sequence[str] = (),
                categories: Sequence[str] = KNOWN_CATEGORIES) -> StructuredQuery:
    text = _normalize(message.strip())
    ptype, cat = detect_product_type(text)
    if cat is None:
        cat = detect_category_word(text)
    hard, soft = detect_requirements(text)
    uses = [u for u in extract_lexicon(text, _USES) if u != "gift"]
    rec = _RECIPIENT.search(text)
    brand = next((b for b in sorted(known_brands, key=len, reverse=True)
                  if re.search(rf"\b{re.escape(b)}\b", text, re.I)), None)

    q = StructuredQuery(
        raw_message=message,
        product_type=ptype,
        category=cat if cat in categories else None,
        budget=parse_budget(text),
        hard_requirements=hard,
        soft_preferences=soft,
        intended_use=uses,
        brand=brand,
        tone_signals=[t for t, p in _TONE.items() if re.search(p, text, re.I)],
        is_gift=bool(_GIFT.search(text)),
        recipient=rec.group(1).lower() if rec else None,
    )
    q.vagueness_signals = _vagueness(q)
    return q


# ====================================================================== LLM layer
class _LLMBudget(BaseModel):
    model_config = ConfigDict(extra="ignore")
    max: Optional[float] = None
    min: Optional[float] = None


class _LLMQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")
    product_type: Optional[str] = None
    category: Optional[str] = None
    budget: Optional[_LLMBudget] = None
    hard_requirements: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)
    intended_use: list[str] = Field(default_factory=list)
    brand: Optional[str] = None
    is_gift: Optional[bool] = None
    recipient: Optional[str] = None


class _Client(Protocol):
    def complete_json(self, system: str, user: str, max_tokens: int = 3500) -> tuple[dict, dict]: ...


SYSTEM_PROMPT = (
    "You convert a shopper's message into structured search requirements. Use ONLY what the "
    "shopper actually said; never add requirements they did not mention. Return ONLY one JSON object."
)
USER_PROMPT_TEMPLATE = """Shopper message: {message}

Return JSON with these fields:
- product_type: what they want to buy, as a short noun phrase in their own words, or null
- category: one of {categories}, or null
- budget: {{"max": number or null, "min": number or null}} (only if a price is stated)
- hard_requirements: product attributes they insist on (NOT the price or budget, NOT "cheap" or "premium", and NOT the product type itself)
- soft_preferences: product attributes marked optional ("ideally", "prefer", "if possible"); a price or budget is NOT an attribute
- intended_use: short lowercase tags for what it is for
- brand: brand name if one is mentioned, else null
- is_gift: true/false
- recipient: who it is for (e.g. "mom") or null"""


def _tokens_supported(term: str, message: str) -> bool:
    """At least half of the term's content words (by 4-letter prefix) appear in the message."""
    toks = [t for t in re.findall(r"[a-z0-9]+", term.lower()) if len(t) >= 3]
    if not toks:
        return False
    msg = message.lower()
    hits = sum(1 for t in toks if re.search(rf"\b{re.escape(t[:4])}", msg))
    return hits / len(toks) >= 0.5


def _number_in_message(value: float, message: str) -> bool:
    body = message.replace(",", "")
    return bool(re.search(rf"(?<![\d.]){re.escape(f'{value:g}')}(?![\d])", body))


def _canonical(term: str) -> Optional[str]:
    for tag, pat in _REQUIREMENTS_C:
        if pat.search(term):
            return tag
    return None


# Words about HOW MUCH the shopper will spend or the style of the ask. They have their own fields
# (budget, tone_signals) and are never product requirements.
_TONE_WORDS = {"cheap", "cheapest", "inexpensive", "affordable", "budget", "basic", "simple", "premium",
               "luxury", "best", "top", "quality", "decent", "good", "nice", "great", "price", "priced"}
_FILLER = {"for", "and", "the", "that", "with", "something", "looking", "need", "want", "ideally",
           "preferably", "under", "over", "below", "above", "than", "less", "more", "about", "around",
           "just", "really", "very", "quite", "some", "also", "maybe"}
_PRICE_PHRASE = re.compile(r"\$|\b(budget|dollars?|bucks?|price[ds]?)\b|\b(under|below|less than|over|above)\s+\d", re.I)


def _is_non_requirement(term: str, q: StructuredQuery) -> bool:
    """Price phrases, tone words and restatements of the product type are not requirements."""
    t = term.lower()
    if _PRICE_PHRASE.search(t):
        return True
    words = [w for w in re.findall(r"[a-z]+", t) if len(w) >= 3 and w not in _FILLER]
    if not words:
        return True
    known = _TONE_WORDS | set(re.findall(r"[a-z]+", f"{q.product_type or ''} {q.category or ''}".lower()))
    return all(w in known or w.rstrip("s") in known for w in words)


def _merge(rq: StructuredQuery, llm: _LLMQuery, categories: Sequence[str],
           known_brands: Sequence[str]) -> StructuredQuery:
    q = rq.model_copy(deep=True)
    msg, notes = q.raw_message, q.notes

    # product type / category: rules win; the LLM only fills a gap
    if q.product_type is None and llm.product_type:
        pt = llm.product_type.strip().lower()
        if _tokens_supported(pt, msg):
            q.product_type = pt
        else:
            notes.append(f"dropped LLM product_type '{pt}' (not supported by the message)")
    if q.category is None and llm.category:
        if llm.category.lower() in categories:
            q.category = llm.category.lower()
        else:
            notes.append(f"dropped LLM category '{llm.category}' (not a catalog category)")

    # budget: rules win; LLM numbers must literally appear in the message
    if q.budget.max is None and q.budget.min is None and llm.budget:
        for field in ("max", "min"):
            v = getattr(llm.budget, field)
            if v is None:
                continue
            if _number_in_message(v, msg):
                setattr(q.budget, field, v)
            else:
                notes.append(f"dropped LLM budget {field}={v:g} (number not in the message)")

    # requirements: each must be supported by the message; rules' hard/soft split wins on conflicts
    known = set(q.hard_requirements) | set(q.soft_preferences)
    for bucket, terms in (("hard", llm.hard_requirements), ("soft", llm.soft_preferences)):
        for term in terms:
            t = term.strip().lower()
            if not t:
                continue
            if not _tokens_supported(t, msg):
                notes.append(f"dropped LLM requirement '{t}' (not supported by the message)")
                continue
            if _is_non_requirement(t, q):
                notes.append(f"dropped LLM requirement '{t}' (price, tone or product type: not a requirement)")
                continue
            tag = _canonical(t) or re.sub(r"\s+", "_", t)
            if tag in known:
                continue
            known.add(tag)
            (q.hard_requirements if bucket == "hard" else q.soft_preferences).append(tag)

    pt_words = set(re.findall(r"[a-z]+", (q.product_type or "").lower()))
    for u in llm.intended_use:
        tag = u.strip().lower()
        if not tag or tag == "gift" or tag in q.intended_use or not _tokens_supported(tag, msg):
            continue
        if set(re.findall(r"[a-z]+", tag)) <= pt_words:     # merely restates the product type
            continue
        q.intended_use.append(tag)

    if q.brand is None and llm.brand:
        b = llm.brand.strip()
        ok = (b.lower() in {x.lower() for x in known_brands}) if known_brands else _tokens_supported(b, msg)
        if ok and _tokens_supported(b, msg):
            q.brand = b
        else:
            notes.append(f"dropped LLM brand '{b}'")

    if llm.recipient and q.recipient is None and _tokens_supported(llm.recipient, msg):
        q.recipient = llm.recipient.strip().lower()
    q.is_gift = q.is_gift or bool(llm.is_gift and _GIFT.search(msg))

    q.parser = "rules+llm"
    q.vagueness_signals = _vagueness(q)
    return q


def understand_query(
    message: str,
    history: Optional[list[dict]] = None,       # reserved for multi-turn memory (later phase)
    client: Optional[_Client] = None,
    known_brands: Sequence[str] = (),
    categories: Sequence[str] = KNOWN_CATEGORIES,
) -> StructuredQuery:
    """Rules first; if a client is given, let the LLM fill gaps. Never raises on LLM failure."""
    rq = rules_parse(message, known_brands, categories)
    if client is None:
        return rq
    try:
        user = USER_PROMPT_TEMPLATE.format(message=message, categories=list(categories))
        data, _meta = client.complete_json(SYSTEM_PROMPT, user, max_tokens=1500)
        llm = _LLMQuery.model_validate(data)
    except (LLMError, ValidationError) as e:
        rq.notes.append(f"LLM unavailable, rules-only: {e}")
        return rq
    return _merge(rq, llm, categories, known_brands)


# ======================================================================== CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Parse a shopper message into a StructuredQuery")
    ap.add_argument("message")
    ap.add_argument("--rules-only", action="store_true")
    args = ap.parse_args()
    settings = get_settings()
    client = LLMClient(settings) if (settings.llm_enabled and not args.rules_only) else None
    print(understand_query(args.message, client=client).model_dump_json(indent=2))


if __name__ == "__main__":
    main()