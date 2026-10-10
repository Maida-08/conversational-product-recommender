"""
The recommendation pipeline: ONE entry point for everything above the services.

    message (+ history)
      -> query understanding   rules first, optional guarded LLM
      -> ambiguity gate        too vague? ask ONE question (options from real catalog data)
      -> retrieval             type relevance + TF-IDF; budget never filters
      -> match check           specific product with no real match? honest NO MATCH
      -> ranking               explicit score, review evidence, labels, tradeoffs
      -> recommendation + upsell
      -> response generation   template, or guarded LLM wording
      -> ChatResponse

The FastAPI route (and later Twilio) only call Engine.respond(): no logic is duplicated anywhere.
Each stage is decorated with @traced for observability (a no-op unless tracing is configured).

An Engine owns the loaded catalog, retriever, ranker and LLM client. Build one with get_engine() (cached) or
construct one directly, as the tests do with client=None so no test can ever call an LLM.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional, Sequence

from app.config import get_settings
from app.data.loader import load_catalog, load_processed
from app.models.schemas import (ChatDetails, ChatMessage, ChatResponse, ClarificationInfo, ProcessedProduct,
                                ProductCard, RankingEntry, StructuredQuery)
from app.services.ambiguity import GateDecision, gate, match_check
from app.services.attribute_extractor import rules_extract
from app.services.llm_client import LLMClient
from app.services.query_understanding import understand_query
from app.services.ranker import RankedProduct, Ranker
from app.services.response_generator import (GeneratedResponse, build_clarification_facts, build_no_match_facts,
                                             build_recommendation_facts, generate)
from app.services.retriever import RetrievalResult, Retriever
from app.services.review_analyzer import ensure_review_evidence
from app.services.tracing import traced
from app.services.upsell import UpsellDecision, plan_recommendation

log = logging.getLogger(__name__)

EXAMPLE_QUERIES = [
    "I need a lightweight waterproof running shoe, ideally under $80",
    "Something nice for my mom",
    "I want a casual sneaker that won't fall apart in light rain",
    "Looking for cheap wireless earbuds, just need something basic",
    "I want the best air purifier for a large open-plan living room",
    "I need a standing desk for my home office",
]


def _bullet(clause: str) -> str:
    """'is fully waterproof' -> 'Fully waterproof'; 'has active noise cancellation' -> 'Active noise cancellation'."""
    for lead in ("is ", "has "):
        if clause.startswith(lead):
            clause = clause[len(lead):]
            break
    return clause[:1].upper() + clause[1:]


def _card(r: RankedProduct, fact: dict[str, Any], role: str, **extra: Any) -> ProductCard:
    p = r.product
    return ProductCard(
        product_id=p.product_id, title=p.title, brand=p.brand, category=p.category, price_usd=p.price_usd,
        average_rating=p.average_rating, num_reviews=p.num_reviews, role=role, match_label=r.label, score=r.score,
        reasons=[_bullet(x) for x in fact["pluses"]], warnings=[_bullet(x) for x in fact["minuses"]], **extra)


class Engine:
    def __init__(self, products: list[ProcessedProduct], client: Any = None,
                 data_source: str = "processed_products.json"):
        self.products = ensure_review_evidence(products)
        self.retriever = Retriever(self.products)
        self.ranker = Ranker(self.products)
        self.brands = sorted({p.brand for p in self.products})
        self.categories = sorted({p.category for p in self.products})
        self.client = client
        self.data_source = data_source

    # ------------------------------------------------------------------ stages
    @traced("query_understanding")
    def _understand(self, message: str) -> StructuredQuery:
        return understand_query(message, client=self.client, known_brands=self.brands, categories=self.categories)

    @traced("ambiguity_gate")
    def _gate(self, q: StructuredQuery) -> GateDecision:
        return gate(q, self.products)

    @traced("retrieval")
    def _retrieve(self, q: StructuredQuery) -> RetrievalResult:
        return self.retriever.retrieve(q)

    @traced("ranking_and_review_evidence")
    def _rank(self, q: StructuredQuery, retrieval: RetrievalResult) -> list[RankedProduct]:
        return self.ranker.rank(q, retrieval.candidates)

    @traced("recommendation_and_upsell")
    def _recommend(self, q: StructuredQuery, ranked: list[RankedProduct]):
        return plan_recommendation(q, ranked)

    @traced("response_generation")
    def _write(self, facts: dict[str, Any]) -> GeneratedResponse:
        return generate(facts, client=self.client, catalog=self.products)

    # --------------------------------------------------------------------- run
    @traced("chat_pipeline")
    def respond(self, message: str, history: Optional[Sequence[ChatMessage]] = None) -> ChatResponse:
        message = message.strip()
        turns = len(history or [])
        q = self._understand(message)
        g = self._gate(q)

        def details(**kw: Any) -> ChatDetails:
            return ChatDetails(parsed_query=q, specificity=g.specificity, gate_reasons=g.reasons,
                               data_source=self.data_source, history_turns=turns, **kw)

        if g.decision == "clarify":
            out = self._write(build_clarification_facts(q, g))
            return ChatResponse(
                type="clarification", assistant_message=out.text,
                clarification=ClarificationInfo(question=out.text, options=g.options),
                details=details(wording=out.method, guard_failures=out.guard_failures))

        q = g.query                                           # may carry an inferred product type
        retrieval = self._retrieve(q)
        m = match_check(q, retrieval)
        if m.decision == "no_match":
            out = self._write(build_no_match_facts(q, [c.product.title for c in retrieval.candidates[:2]]))
            return ChatResponse(type="no_match", assistant_message=out.text,
                                details=details(match_reasons=m.reasons, wording=out.method,
                                                guard_failures=out.guard_failures))

        ranked = self._rank(q, retrieval)
        rec, upsell = self._recommend(q, ranked)
        facts = build_recommendation_facts(q, rec, upsell)
        out = self._write(facts)

        cards = [_card(rec.primary, facts["primary"], "primary")]
        cards += [_card(r, f, "secondary") for r, f in zip(rec.secondary, facts["secondary"])]
        cards += [_card(r, f, "caution") for r, f in zip(rec.cautions, facts["cautions"])]
        up_card = None
        if upsell.suggestion:
            s = upsell.suggestion
            up_card = _card(s.product, facts["upsell"], "upsell", price_delta=s.price_delta, style=s.style,
                            benefits=[b[:1].upper() + b[1:] for b in facts["upsell"]["benefits"]])
        return ChatResponse(
            type="recommendation", assistant_message=out.text, recommendations=cards, upsell=up_card,
            details=details(
                match_reasons=m.reasons, mode=rec.mode, upsell_skipped_reason=upsell.skipped_reason,
                upsell_rejected=upsell.rejected, wording=out.method, guard_failures=out.guard_failures,
                ranking=[RankingEntry(product_id=r.product.product_id, title=r.product.title,
                                      price_usd=r.product.price_usd, rank=r.rank, score=r.score, label=r.label,
                                      components={c.name: c.value for c in r.components}) for r in ranked[:8]]))


@lru_cache
def get_engine() -> Engine:
    """Build the production engine once: processed data if present (else rules over the CSV), and Groq if a key is set."""
    settings = get_settings()
    products = load_processed()
    source = settings.processed_products.name
    if not products:
        products = [rules_extract(r) for r in load_catalog()]          # raises FileNotFoundError if no CSV either
        source = f"{settings.catalog_csv.name} (rules only: run the attribute extractor for LLM enrichment)"
    client = LLMClient(settings) if settings.llm_enabled else None
    log.info("engine ready: %d products from %s, llm=%s", len(products), source, bool(client))
    return Engine(products, client=client, data_source=source)


def run_chat_pipeline(message: str, history: Optional[Sequence[ChatMessage]] = None) -> ChatResponse:
    """The single function every channel (web API, SMS/WhatsApp) calls."""
    return get_engine().respond(message, history)