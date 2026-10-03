"""
Attribute extraction: raw catalog row -> structured attributes.

Design: rules own the facts where rules are reliable; the LLM handles free-form text
under guardrails.

  Rules only ........ weight, ANC/wireless/GPS flags (incl. negations)
  Rules first ....... water resistance (LLM only if rules found nothing AND it quotes
                      the description verbatim)
  LLM + lexicon ..... product_type, materials, intended_use, features, numeric specs

Every LLM claim is checked against the description. Anything that fails the check is
dropped and recorded in extraction.notes, so a reviewer can see what the guard caught.
If the LLM is unavailable, the result is rules-only and still valid.

Run (from backend/):
    python -m app.services.attribute_extractor                # LLM if key set, else rules
    python -m app.services.attribute_extractor --rules-only
    python -m app.services.attribute_extractor --force        # ignore cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from typing import Any, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import get_settings
from app.data.loader import load_catalog, load_processed, save_processed
from app.models.schemas import (
    Attributes, ExtractionInfo, ProcessedProduct, RawProduct, WaterResistance, Weight,
)
from app.services.llm_client import LLMClient, LLMError

# Bump when prompts or rules change so cached LLM results are re-extracted.
EXTRACTOR_VERSION = "1"

# =============================================================== WEIGHT (rules)
_OZ_G, _LB_G = 28.3495, 453.592
_UNIT_G = {"kg": 1000.0, "g": 1.0, "gram": 1.0, "grams": 1.0, "oz": _OZ_G, "ounce": _OZ_G,
           "ounces": _OZ_G, "lb": _LB_G, "lbs": _LB_G, "pound": _LB_G, "pounds": _LB_G}
_QTY = r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>kg|grams?|g|oz|ounces?|lbs?|pounds?)\b"
_LBOZ = r"(?P<lb>\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)\s*(?P<oz>\d+(?:\.\d+)?)\s*(?:oz|ounces?)\b"
_Q = rf"(?:{_LBOZ}|{_QTY})"
# Only accept numbers that are clearly the PRODUCT's weight. This avoids false hits such as
# "fits dogs up to 50 lb" or "resistance from 10 to 50 lb".
_W_PATTERNS = [
    re.compile(rf"\bweigh(?:s|ing|ed)?\s+(?:about\s+|approximately\s+|only\s+|just\s+|around\s+)?{_Q}", re.I),
    re.compile(rf"{_Q}\s*(?:\([^)]*\))?\s*per\s+(?:shoe|pair)", re.I),
    re.compile(rf"\bat\s+just\s+{_Q}", re.I),
]
_BASIS = re.compile(r"per\s+(shoe|pair)\b", re.I)


def _grams(m: re.Match) -> float:
    gd = m.groupdict()
    if gd.get("lb") is not None:
        return float(gd["lb"]) * _LB_G + float(gd["oz"]) * _OZ_G
    return float(gd["num"]) * _UNIT_G[gd["unit"].lower()]


def extract_weight(text: str) -> Weight:
    for pat in _W_PATTERNS:
        m = pat.search(text)
        if m:
            tail = text[m.end(): m.end() + 25]
            b = _BASIS.search(m.group(0)) or _BASIS.search(tail)
            basis = {"shoe": "per_shoe", "pair": "per_pair"}[b.group(1).lower()] if b else "item"
            raw = re.sub(r"^weigh\w*\s+", "", m.group(0).strip(), flags=re.I)
            return Weight(grams=round(_grams(m), 1), raw=raw, basis=basis)
    return Weight()   # unknown, never guessed


# ================================================================ WATER (rules)
_NEG_WATER = re.compile(
    r"\b(?:not|isn't|is not|no)\s+(?:fully\s+)?water[\s-]?(?:proof|resist\w*|repel\w*)|\bno\s+water\s+resistance",
    re.I)
_WATERPROOF = re.compile(r"water[\s-]?proof", re.I)
_RESISTANT = re.compile(r"water[\s-]?(?:resist\w*|repel\w*)", re.I)
_SPLASH = re.compile(r"splash[\s-]?(?:proof|resist\w*)", re.I)
_IPX = re.compile(r"\bIPX([0-8])\b", re.I)
_ATM = re.compile(r"\b(\d+)\s*ATM\b", re.I)


def _sentence_around(text: str, start: int, end: int) -> str:
    s = text.rfind(". ", 0, start)
    s = 0 if s == -1 else s + 2
    e = text.find(". ", end)
    e = len(text) if e == -1 else e + 1
    return text[s:e].strip()   # always a verbatim substring of text


def extract_water_resistance(text: str) -> WaterResistance:
    m = _NEG_WATER.search(text)
    if m:
        return WaterResistance(status=False, evidence=_sentence_around(text, m.start(), m.end()),
                               source="rules")

    ipx, atm = _IPX.search(text), _ATM.search(text)
    rating = f"IPX{ipx.group(1)}" if ipx else (f"{atm.group(1)}ATM" if atm else None)

    m = _WATERPROOF.search(text)
    if m:
        return WaterResistance(status=True, level="waterproof", rating=rating,
                               evidence=_sentence_around(text, m.start(), m.end()), source="rules")
    if ipx:
        n = int(ipx.group(1))
        level = "waterproof" if n >= 7 else "water_resistant" if n >= 5 else "splash_resistant"
        return WaterResistance(status=True, level=level, rating=rating,
                               evidence=_sentence_around(text, ipx.start(), ipx.end()), source="rules")
    if atm:
        level = "water_resistant" if int(atm.group(1)) >= 5 else "splash_resistant"
        return WaterResistance(status=True, level=level, rating=rating,
                               evidence=_sentence_around(text, atm.start(), atm.end()), source="rules")
    for pat, level in ((_RESISTANT, "water_resistant"), (_SPLASH, "splash_resistant")):
        m = pat.search(text)
        if m:
            return WaterResistance(status=True, level=level,
                                   evidence=_sentence_around(text, m.start(), m.end()), source="rules")
    return WaterResistance()   # unknown: silence is NOT evidence of "not waterproof"


# ============================================================== FEATURES (rules)
# name -> (positive pattern, negative pattern). Negative is checked first
# so "No active noise cancellation" lands in `lacks`, not `features`.
_FEATURE_RULES: dict[str, tuple[str, Optional[str]]] = {
    "active_noise_cancellation": (r"noise[\s-]cancel\w*|\bANC\b", r"\bno\s+(?:active\s+)?noise[\s-]cancel\w*"),
    "wireless": (r"wireless|bluetooth", r"\bnot\s+wireless\b|\bwired\b"),
    "gps": (r"\bGPS\b", None),
}


def extract_feature_flags(text: str) -> tuple[list[str], list[str]]:
    features, lacks = [], []
    for name, (pos, neg) in _FEATURE_RULES.items():
        if neg and re.search(neg, text, re.I):
            lacks.append(name)
        elif re.search(pos, text, re.I):
            features.append(name)
    return features, lacks


# ============================================================ LEXICONS (rules)
_MATERIALS = [
    ("leather", r"\bleather\b"), ("canvas", r"\bcanvas\b"), ("cotton", r"\bcotton\b"),
    ("mesh", r"\bmesh\b"), ("knit", r"\bknit\b"), ("stainless steel", r"stainless[\s-]steel"),
    ("silicone", r"\bsilicone\b"), ("ceramic", r"\bceramic\b"), ("cast iron", r"cast[\s-]iron"),
    ("aluminum", r"\balumin(?:um|ium)\b"), ("carbon fiber", r"carbon[\s-]fib(?:er|re)"),
    ("wood", r"\bwood(?:en)?\b"), ("glass", r"\bglass\b"), ("acrylic", r"\bacrylic\b"),
    ("microfiber", r"\bmicrofib(?:er|re)\b"), ("fleece", r"\bfleece\b"),
    ("memory foam", r"memory[\s-]foam"), ("foam", r"\bfoam\b"), ("rubber", r"\brubber\b"),
    ("natural rubber", r"natural[\s-]rubber"), ("TPE", r"\bTPE\b"), ("bamboo", r"\bbamboo\b"),
    ("soy wax", r"\bsoy\b"), ("latex", r"\blatex\b"), ("polyester", r"\bpolyester\b"),
    ("nylon", r"\bnylon\b"), ("TPU", r"\bTPU\b"), ("EVA", r"\bEVA\b"),
    ("synthetic", r"\bsynthetic\b"),
]

_USES = [
    ("running", r"\brunn(?:ing|er|ers)\b|\bjogg\w*"), ("trail", r"\btrail\b"),
    ("hiking", r"\bhik(?:e|es|ing|er)\b"), ("walking", r"\bwalking\b"), ("casual", r"\bcasual\b"),
    ("everyday", r"\beveryday\b|\bdaily\b"), ("racing", r"\brac(?:e|ing)\b"),
    ("commuting", r"\bcommut\w*"), ("workouts", r"\bworkouts?\b|\bgym\b|\bexercise\b"),
    ("swimming", r"\bswim\w*"), ("water sports", r"kayak|water and trail|water use"),
    ("camping", r"\bcamp\w*"), ("backpacking", r"\bbackpacking\b"), ("travel", r"\btravel\b"),
    ("sleep", r"\bsleep\w*"), ("relaxation", r"\brelax\w*|\bcalming\b"),
    ("gift", r"\bgift(?:s|-boxed|ed)?\b|\bpresent\b"), ("yoga", r"\byoga\b"),
    ("cycling", r"\bcycl\w*|\bbike\b"), ("winter", r"\bsnow\b|\bwinter\b"),
    ("kids", r"\bkids?\b|\btoddlers?\b|\bbaby\b|\bages?\s+\d|\bfrom birth\b"),
]


def extract_lexicon(text: str, lexicon: list[tuple[str, str]]) -> list[str]:
    found = [name for name, pat in lexicon if re.search(pat, text, re.I)]
    for specific, generic in (("memory foam", "foam"), ("natural rubber", "rubber")):
        if specific in found and generic in found:
            found.remove(generic)
    return found


# ================================================================ rules pipeline
def _source_hash(raw: RawProduct) -> str:
    h = hashlib.sha256(f"{EXTRACTOR_VERSION}|{raw.title}|{raw.description}".encode()).hexdigest()
    return h[:12]


def _product_type_from_title(raw: RawProduct) -> str:
    title = raw.title
    if title.lower().startswith(raw.brand.lower()):
        title = title[len(raw.brand):]
    return title.strip().lower()


def rules_extract(raw: RawProduct) -> ProcessedProduct:
    text = raw.description
    features, lacks = extract_feature_flags(text)
    attrs = Attributes(
        product_type=_product_type_from_title(raw),
        water_resistance=extract_water_resistance(text),
        weight=extract_weight(text),
        materials=extract_lexicon(text, _MATERIALS),
        intended_use=extract_lexicon(text, _USES),
        features=features,
        lacks=lacks,
    )
    info = ExtractionInfo(method="rules", source_hash=_source_hash(raw),
                          notes=["product_type derived from title (no LLM)"])
    return ProcessedProduct(**raw.model_dump(), attributes=attrs, extraction=info)


# ====================================================================== LLM layer
class _LLMWater(BaseModel):
    model_config = ConfigDict(extra="ignore")
    status: Optional[bool] = None
    level: Optional[str] = None
    evidence: Optional[str] = None


class _LLMProduct(BaseModel):
    model_config = ConfigDict(extra="ignore")
    product_id: str
    product_type: Optional[str] = None
    materials: list[str] = Field(default_factory=list)
    intended_use: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    lacks: list[str] = Field(default_factory=list)
    specs: dict[str, Any] = Field(default_factory=dict)
    water_resistance: Optional[_LLMWater] = None


class _Client(Protocol):   # lets tests inject a fake client
    def complete_json(self, system: str, user: str, max_tokens: int = 3500) -> tuple[dict, dict]: ...


SYSTEM_PROMPT = (
    "You extract product attributes from catalog text. Use ONLY facts explicitly stated in the "
    "description. Never guess or infer beyond the text. If a fact is not stated, omit it. "
    "Return ONLY one JSON object, no prose."
)

USER_PROMPT_TEMPLATE = """For each product return an object with these fields:
- product_id (copy exactly)
- product_type: short noun phrase, e.g. "running shoe", "wireless earbuds"
- materials: materials named in the description
- intended_use: short lowercase tags for what it is meant for (e.g. "trail running", "gift")
- features: explicitly stated features, snake_case (e.g. "hepa_filter", "usb_c_power_delivery")
- lacks: features the description explicitly says are ABSENT, snake_case
- specs: numeric specs stated in the text, snake_case keys, number values exactly as written
  (e.g. battery_hours_per_charge, battery_hours_with_case, capacity_mah, coverage_sq_ft, power_w, lumens)
- water_resistance: {{"status": true|false|null, "level": "waterproof"|"water_resistant"|"splash_resistant"|null,
  "evidence": "EXACT quote copied from the description, or null"}}
  status=false ONLY if the description explicitly says it is not waterproof/water resistant. Silence means null.
Do NOT extract weight. Do not convert units.
Output format: {{"products": [ ... ]}}

Products:
{products_json}"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("-", " ").replace("_", " ")).strip()


def _in_text(term: str, text: str) -> bool:
    t, body = _norm(term), _norm(text)
    return bool(t) and (t in body or t.rstrip("s") in body)


def _snake(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _number_in_text(value: float, text: str) -> bool:
    body = text.replace(",", "")
    return bool(re.search(rf"(?<![\d.]){re.escape(f'{value:g}')}(?![\d])", body))


_NEGATION = re.compile(r"\b(no|not|without|lacks?|isn't|doesn't|non)\b", re.I)


def _feature_supported(tag: str, text: str) -> bool:
    tokens = [t for t in tag.split("_") if len(t) >= 2]
    if not tokens:
        return False
    body = _norm(text)
    hits = sum(1 for t in tokens if t in body)
    return hits / len(tokens) >= 0.5


def _apply_llm(p: ProcessedProduct, llm: _LLMProduct) -> None:
    """Merge LLM output into p, dropping anything the description does not support."""
    a, notes, desc = p.attributes, p.extraction.notes, p.description
    if "product_type derived from title (no LLM)" in notes:
        notes.remove("product_type derived from title (no LLM)")

    if llm.product_type and llm.product_type.strip():
        a.product_type = llm.product_type.strip().lower()[:60]

    for m in llm.materials:
        mn = m.strip()
        if mn and _in_text(mn, desc):
            if mn.lower() not in [x.lower() for x in a.materials]:
                a.materials.append(mn)
        elif mn:
            notes.append(f"dropped unsupported material '{mn}'")

    for u in llm.intended_use[:6]:
        tag = _norm(u)
        if tag and len(tag.split()) <= 3 and tag not in a.intended_use:
            a.intended_use.append(tag)

    for tag in map(_snake, llm.features):
        if not tag or tag in a.features:
            continue
        if _feature_supported(tag, desc):
            a.features.append(tag)
        else:
            notes.append(f"dropped unsupported feature '{tag}'")

    for tag in map(_snake, llm.lacks):
        if not tag or tag in a.lacks:
            continue
        if _feature_supported(tag, desc) and _NEGATION.search(desc):
            a.lacks.append(tag)
            if tag in a.features:
                a.features.remove(tag)
        else:
            notes.append(f"dropped unsupported 'lacks' claim '{tag}'")

    for key, raw_val in llm.specs.items():
        k = _snake(str(key))
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            notes.append(f"dropped non-numeric spec '{key}'")
            continue
        if k and _number_in_text(val, desc):
            a.specs[k] = val
        else:
            notes.append(f"dropped spec '{k}={raw_val}' (number not found in description)")

    # Water resistance: rules win. The LLM may only fill a gap, and only with a verbatim quote.
    w = llm.water_resistance
    if a.water_resistance.status is None and w and w.status is not None:
        quote = (w.evidence or "").strip()
        verbatim = bool(quote) and quote.lower() in desc.lower()
        if not verbatim:
            notes.append("dropped LLM water-resistance claim: evidence is not a verbatim quote")
        elif w.status is False and not _NEGATION.search(quote):
            notes.append("dropped LLM 'not waterproof' claim: quote contains no negation")
        else:
            level = w.level if w.level in ("waterproof", "water_resistant", "splash_resistant") else None
            a.water_resistance = WaterResistance(
                status=w.status, level=level if w.status else None, evidence=quote, source="llm")


def llm_extract_batch(batch: list[ProcessedProduct], client: _Client) -> tuple[dict[str, _LLMProduct], dict]:
    payload = [{"product_id": p.product_id, "title": p.title, "category": p.category,
                "description": p.description} for p in batch]
    user = USER_PROMPT_TEMPLATE.format(products_json=json.dumps(payload, ensure_ascii=False))
    data, meta = client.complete_json(SYSTEM_PROMPT, user)
    out: dict[str, _LLMProduct] = {}
    for item in data.get("products", []):
        try:
            parsed = _LLMProduct.model_validate(item)
        except ValidationError:
            continue
        out[parsed.product_id] = parsed
    return out, meta


# ============================================================ batch orchestration
def build_processed_products(
    raw_products: list[RawProduct],
    client: Optional[_Client] = None,
    existing: Optional[list[ProcessedProduct]] = None,
    batch_size: int = 5,
    on_batch_done=None,
    log=print,
) -> list[ProcessedProduct]:
    """Rules for everything, then LLM enrichment in small batches. Never raises on LLM failure."""
    cache = {p.product_id: p for p in (existing or [])}
    results: list[ProcessedProduct] = []
    pending: list[ProcessedProduct] = []

    for raw in raw_products:
        cached = cache.get(raw.product_id)
        if (cached and cached.extraction.source_hash == _source_hash(raw)
                and (cached.extraction.method == "rules+llm" or client is None)):
            results.append(cached)          # reuse: nothing changed
        else:
            p = rules_extract(raw)
            results.append(p)
            if client is not None:
                pending.append(p)

    llm_alive = client is not None
    for i in range(0, len(pending), batch_size):
        if not llm_alive:
            break
        batch = pending[i:i + batch_size]
        ids = ", ".join(p.product_id for p in batch)
        try:
            parsed, meta = llm_extract_batch(batch, client)
        except LLMError as e:
            log(f"  batch [{ids}] LLM failed -> rules-only for the rest of this run: {e}")
            for p in pending[i:]:
                p.extraction.notes.append(f"LLM unavailable, rules-only: {e}")
            llm_alive = False
            break
        for p in batch:
            llm = parsed.get(p.product_id)
            if llm is None:
                p.extraction.notes.append("LLM returned nothing for this product; rules-only")
                continue
            _apply_llm(p, llm)
            p.extraction.method = "rules+llm"
            p.extraction.model = meta.get("model")
        log(f"  batch [{ids}] ok via {meta.get('model')} ({meta.get('total_tokens')} tokens)")
        if on_batch_done:
            on_batch_done(results)
    return results


# ===================================================================== CLI
SAMPLE_IDS = ["P001", "P002", "P003", "P005", "P011", "P014", "P015", "P024"]


def _summarize(products: list[ProcessedProduct]) -> None:
    n = len(products)
    ws = {"true": 0, "false": 0, "unknown": 0}
    for p in products:
        s = p.attributes.water_resistance.status
        ws["unknown" if s is None else "true" if s else "false"] += 1
    no_weight = sum(1 for p in products if p.attributes.weight.grams is None)
    llm_n = sum(1 for p in products if p.extraction.method == "rules+llm")
    dropped = sum(1 for p in products for note in p.extraction.notes if note.startswith("dropped"))
    print(f"\n{n} products | LLM-enriched: {llm_n} | guard drops: {dropped}")
    print(f"water_resistance: {ws} | weight unknown: {no_weight}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build processed_products.json")
    ap.add_argument("--rules-only", action="store_true", help="skip the LLM entirely")
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    args = ap.parse_args()

    settings = get_settings()
    raw = load_catalog()
    print(f"Loaded {len(raw)} products from {settings.catalog_csv.name}")

    client = None
    if not args.rules_only:
        if settings.llm_enabled:
            client = LLMClient(settings)
            print(f"LLM: {settings.llm_model} (fallback {settings.llm_fallback_model})")
        else:
            print("GROQ_API_KEY not set -> running rules-only")

    existing = [] if args.force else load_processed()
    products = build_processed_products(
        raw, client=client, existing=existing,
        on_batch_done=lambda partial: save_processed(partial),   # checkpoint after each batch
    )
    path = save_processed(products)
    print(f"Saved {path}")

    samples = [p.model_dump() for p in products if p.product_id in SAMPLE_IDS]
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    out = settings.outputs_dir / "attribute_extraction_samples.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {out} ({len(samples)} sample products)")
    _summarize(products)


if __name__ == "__main__":
    main()