"""
The recommendation pipeline: ONE entry point for everything above the services.

    message (+ history)
      -> query understanding
      -> ambiguity gate
      -> retrieval
      -> match check
      -> ranking
      -> recommendation + upsell
      -> response generation
      -> ChatResponse

The FastAPI route (and later Twilio) only call Engine.respond().
Each stage is decorated with @traced for observability.

An Engine owns the loaded catalog, retriever, ranker and LLM client.
Use get_engine() for the cached production engine, or construct Engine
directly with client=None in tests to prevent LLM calls.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional, Sequence

from app.config import get_settings
from app.data.loader import load_catalog, load_processed
from app.models.schemas import (
    ChatDetails,
    ChatMessage,
    ChatResponse,
    ClarificationInfo,
    ProcessedProduct,
    ProductCard,
    RankingEntry,
    StructuredQuery,
)
from app.services.ambiguity import gate, match_check
from app.services.attribute_extractor import rules_extract
from app.services.llm_client import LLMClient
from app.services.query_understanding import understand_query
from app.services.ranker import RankedProduct, Ranker
from app.services.response_generator import (
    GeneratedResponse,
    build_clarification_facts,
    build_no_match_facts,
    build_recommendation_facts,
    generate,
)
from app.services.retriever import RetrievalResult, Retriever
from app.services.review_analyzer import ensure_review_evidence
from app.services.tracing import traced
from app.services.upsell import plan_recommendation

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
    """Convert evidence clauses into readable product-card bullets."""
    for lead in ("is ", "has "):
        if clause.startswith(lead):
            clause = clause[len(lead):]
            break

    return clause[:1].upper() + clause[1:]


def _card(
    r: RankedProduct,
    fact: dict[str, Any],
    role: str,
    **extra: Any,
) -> ProductCard:
    """Convert a ranked product and its evidence into an API product card."""
    p = r.product

    return ProductCard(
        product_id=p.product_id,
        title=p.title,
        brand=p.brand,
        category=p.category,
        price_usd=p.price_usd,
        average_rating=p.average_rating,
        num_reviews=p.num_reviews,
        role=role,
        match_label=r.label,
        score=r.score,
        reasons=[_bullet(x) for x in fact["pluses"]],
        warnings=[_bullet(x) for x in fact["minuses"]],
        **extra,
    )


class Engine:
    """Own the catalog and execute the complete recommendation pipeline."""

    def __init__(
        self,
        products: list[ProcessedProduct],
        client: Any = None,
        data_source: str = "processed_products.json",
    ):
        self.products = ensure_review_evidence(products)
        self.retriever = Retriever(self.products)
        self.ranker = Ranker(self.products)
        self.brands = sorted({p.brand for p in self.products})
        self.categories = sorted({p.category for p in self.products})
        self.client = client
        self.data_source = data_source

    # ---------------------------------------------------------------
    # Pipeline stages
    # ---------------------------------------------------------------

    @traced("query_understanding")
    def _understand(self, message: str) -> StructuredQuery:
        return understand_query(
            message,
            client=self.client,
            known_brands=self.brands,
            categories=self.categories,
        )

    @traced("ambiguity_gate")
    def _gate(self, q: StructuredQuery):
        return gate(q, self.products)

    @traced("retrieval")
    def _retrieve(self, q: StructuredQuery) -> RetrievalResult:
        return self.retriever.retrieve(q)

    @traced("ranking_and_review_evidence")
    def _rank(
        self,
        q: StructuredQuery,
        retrieval: RetrievalResult,
    ) -> list[RankedProduct]:
        return self.ranker.rank(q, retrieval.candidates)

    @traced("recommendation_and_upsell")
    def _recommend(self, q: StructuredQuery, ranked: list[RankedProduct]):
        return plan_recommendation(q, ranked)

    @traced("response_generation")
    def _write(self, facts: dict[str, Any]) -> GeneratedResponse:
        return generate(
            facts,
            client=self.client,
            catalog=self.products,
        )

    # ---------------------------------------------------------------
    # Strict budget validation
    # ---------------------------------------------------------------

    @staticmethod
    def _filter_strict_budget(
        q: StructuredQuery,
        ranked: list[RankedProduct],
    ) -> list[RankedProduct]:
        """
        For firm maximum budgets, retain products that satisfy:
        - the requested product type,
        - the budget boundary,
        - every hard requirement.

        Budget filtering is deliberately applied after ranking.
        """
        if q.budget.firmness not in ("standard", "firm") or q.budget.max is None:
            return ranked

        message_lower = q.raw_message.lower()

        strictly_under = any(
            phrase in message_lower
            for phrase in (
                "strictly under",
                "strictly less than",
                "less than",
            )
        )

        def is_valid_strict_match(r: RankedProduct) -> bool:
            type_score = next(
                (
                    component.value
                    for component in r.components
                    if component.name == "type"
                ),
                0.0,
            )

            price = r.product.price_usd

            # Product type must be a strong match, and price must be known.
            if type_score < 0.9 or price is None:
                return False

            # "Less than $X" excludes a product priced exactly at $X.
            if strictly_under and price >= q.budget.max:
                return False

            # A regular firm maximum permits a product priced exactly at $X.
            if not strictly_under and price > q.budget.max:
                return False

            hard_requirements = [
                requirement
                for requirement in r.requirements
                if requirement.kind == "hard"
            ]

            return all(
                requirement.state == "met"
                for requirement in hard_requirements
            )

        return [product for product in ranked if is_valid_strict_match(product)]

    # ---------------------------------------------------------------
    # Main pipeline
    # ---------------------------------------------------------------

    @traced("chat_pipeline")
    def respond(
        self,
        message: str,
        history: Optional[Sequence[ChatMessage]] = None,
    ) -> ChatResponse:
        message = message.strip()
        turns = len(history or [])

        q = self._understand(message)
        g = self._gate(q)

        def details(**kw: Any) -> ChatDetails:
            return ChatDetails(
                parsed_query=q,
                specificity=g.specificity,
                gate_reasons=g.reasons,
                data_source=self.data_source,
                history_turns=turns,
                **kw,
            )

        # 1. Ask one clarification question when the request is too vague.
        if g.decision == "clarify":
            out = self._write(build_clarification_facts(q, g))

            return ChatResponse(
                type="clarification",
                assistant_message=out.text,
                clarification=ClarificationInfo(
                    question=out.text,
                    options=g.options,
                ),
                details=details(
                    wording=out.method,
                    guard_failures=out.guard_failures,
                ),
            )

        # The ambiguity gate may infer a product type.
        q = g.query

        # 2. Retrieve products relevant to the query.
        retrieval = self._retrieve(q)

        # 3. Check whether a specific requested product has a real match.
        match = match_check(q, retrieval)

        if match.decision == "no_match":
            out = self._write(
                build_no_match_facts(
                    q,
                    [
                        candidate.product.title
                        for candidate in retrieval.candidates[:2]
                    ],
                )
            )

            return ChatResponse(
                type="no_match",
                assistant_message=out.text,
                details=details(
                    match_reasons=match.reasons,
                    wording=out.method,
                    guard_failures=out.guard_failures,
                ),
            )

        # 4. Rank the retrieved products.
        ranked = self._rank(q, retrieval)

        # 5. Enforce firm budget and hard requirements.
        ranked = self._filter_strict_budget(q, ranked)

        if not ranked:
            out = self._write(
                build_no_match_facts(q, [], reason="constraints")
            )
            return ChatResponse(
                type="no_match",
                assistant_message=out.text,
                details=details(
                    match_reasons=[
                        "No catalog product satisfies the strict budget, "
                        "product type, and all hard requirements."
                    ],
                    wording=out.method,
                    guard_failures=out.guard_failures,
                ),
            )

        # 6. Choose recommendations and evaluate a possible upsell.
        rec, upsell = self._recommend(q, ranked)

        # 7. Generate the assistant's response from structured facts.
        facts = build_recommendation_facts(q, rec, upsell)
        out = self._write(facts)

        # 8. Build product cards.
        cards = [
            _card(rec.primary, facts["primary"], "primary")
        ]

        cards += [
            _card(product, fact, "secondary")
            for product, fact in zip(
                rec.secondary,
                facts["secondary"],
            )
        ]

        cards += [
            _card(product, fact, "caution")
            for product, fact in zip(
                rec.cautions,
                facts["cautions"],
            )
        ]

        # 9. Add an upsell card only when the planner selected one.
        up_card = None

        if upsell.suggestion:
            suggestion = upsell.suggestion

            up_card = _card(
                suggestion.product,
                facts["upsell"],
                "upsell",
                price_delta=suggestion.price_delta,
                style=suggestion.style,
                benefits=[
                    benefit[:1].upper() + benefit[1:]
                    for benefit in facts["upsell"]["benefits"]
                ],
            )

        # 10. Return the final structured API response.
        return ChatResponse(
            type="recommendation",
            assistant_message=out.text,
            recommendations=cards,
            upsell=up_card,
            details=details(
                match_reasons=match.reasons,
                mode=rec.mode,
                upsell_skipped_reason=upsell.skipped_reason,
                upsell_rejected=upsell.rejected,
                wording=out.method,
                guard_failures=out.guard_failures,
                ranking=[
                    RankingEntry(
                        product_id=product.product.product_id,
                        title=product.product.title,
                        price_usd=product.product.price_usd,
                        rank=product.rank,
                        score=product.score,
                        label=product.label,
                        components={
                            component.name: component.value
                            for component in product.components
                        },
                    )
                    for product in ranked[:8]
                ],
            ),
        )


@lru_cache
def get_engine() -> Engine:
    """
    Build and cache the production engine.

    Prefer processed products when available. Otherwise, fall back to
    rule-based extraction from the catalog CSV.
    """
    settings = get_settings()

    products = load_processed()
    source = settings.processed_products.name

    if not products:
        products = [
            rules_extract(row)
            for row in load_catalog()
        ]

        source = (
            f"{settings.catalog_csv.name} "
            "(rules only: run the attribute extractor for LLM enrichment)"
        )

    client = LLMClient(settings) if settings.llm_enabled else None

    log.info(
        "engine ready: %d products from %s, llm=%s",
        len(products),
        source,
        bool(client),
    )

    return Engine(
        products,
        client=client,
        data_source=source,
    )


def run_chat_pipeline(
    message: str,
    history: Optional[Sequence[ChatMessage]] = None,
) -> ChatResponse:
    """The single pipeline entry point for every channel."""
    return get_engine().respond(message, history)
