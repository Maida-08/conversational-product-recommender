"""
Chat route. Deliberately THIN: validate (Pydantic does that), call the pipeline, map errors.
All business logic lives in app/services/pipeline.py.
"""
import logging

from fastapi import APIRouter, Body, Depends, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services.pipeline import EXAMPLE_QUERIES, Engine, get_engine

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])


def engine_dependency() -> Engine:
    """Resolve the engine. Tests override this dependency with a rules-only engine."""
    try:
        return get_engine()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Product catalog is not available: {e}")


_EXAMPLES = {
    "specific_with_budget": {
        "summary": "Specific query with a budget (price tradeoff is stated, a review-flagged product is a caution)",
        "value": {"message": "I need a lightweight waterproof running shoe, ideally under $80"}},
    "ambiguous": {
        "summary": "Too vague: asks ONE clarifying question instead of ranking",
        "value": {"message": "Something nice for my mom"}},
    "review_only_signal": {
        "summary": "A signal that exists only in reviews (the canvas sneaker soaks in light rain)",
        "value": {"message": "I want a casual sneaker that won't fall apart in light rain"}},
    "upsell": {
        "summary": "Cheap and basic: the primary stays the cheap pick; a step-up is a brief optional mention",
        "value": {"message": "Looking for cheap wireless earbuds, just need something basic"}},
    "no_match": {
        "summary": "Specific product that is not in the catalog (needs the LLM to name an unknown product type)",
        "value": {"message": "I need a standing desk for my home office"}},
}


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a shopper message and get the assistant's reply",
    description=(
        "Runs the full pipeline: query understanding, ambiguity gate, retrieval, ranking with review evidence, "
        "recommendation, upsell decision and response generation.\n\n"
        "`type` tells the client what to render: **recommendation** (product cards), **clarification** "
        "(one question, no products) or **no_match** (an honest message). `details` explains the decision "
        "(parsed query, specificity, ranking scores, why an upsell was or was not offered)."),
)
def chat(request: ChatRequest = Body(..., openapi_examples=_EXAMPLES),
         engine: Engine = Depends(engine_dependency)) -> ChatResponse:
    try:
        return engine.respond(request.message, request.history)
    except Exception:                                    # never leak internals to a shopper
        log.exception("chat pipeline failed")
        raise HTTPException(status_code=500, detail="Something went wrong while preparing recommendations.")


@router.get("/examples", response_model=list[str], summary="Example shopper messages for trying the assistant")
def examples() -> list[str]:
    return EXAMPLE_QUERIES