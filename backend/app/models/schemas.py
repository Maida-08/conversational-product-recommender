"""
Pydantic schemas.

Phase 3 (this file, for now): product data models.
Phase 10 will append the API request/response models (ChatRequest, ChatResponse, ...).

Tri-state convention used throughout:
    True  -> the data supports this
    False -> the data EXPLICITLY says this is not the case
    None  -> unknown. Silence in a description is NOT evidence of absence.
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class RawProduct(BaseModel):
    """One catalog row after loading and cleaning (nothing inferred yet)."""

    product_id: str
    category: str
    brand: str
    title: str
    description: str = ""
    price_usd: Optional[float] = None
    average_rating: Optional[float] = None
    num_reviews: Optional[int] = None
    review_snippets: list[str] = Field(default_factory=list)


class WaterResistance(BaseModel):
    status: Optional[bool] = Field(
        None,
        description="True = description claims some water protection; "
        "False = description explicitly says it has none; None = unknown.",
    )
    level: Optional[Literal["waterproof", "water_resistant", "splash_resistant"]] = None
    rating: Optional[str] = Field(None, description="e.g. 'IPX7' or '5ATM' if stated")
    evidence: Optional[str] = Field(None, description="Verbatim sentence from the description")
    source: Literal["rules", "llm", "none"] = "none"


class Weight(BaseModel):
    grams: Optional[float] = Field(None, description="Weight in grams, as stated (see basis)")
    raw: Optional[str] = Field(None, description="The weight text as written")
    basis: Optional[Literal["item", "per_shoe", "per_pair"]] = Field(
        None, description="What the weight refers to. Footwear may be per shoe or per pair."
    )


class Attributes(BaseModel):
    product_type: Optional[str] = Field(None, description="Short noun phrase, e.g. 'running shoe'")
    water_resistance: WaterResistance = Field(default_factory=WaterResistance)
    weight: Weight = Field(default_factory=Weight)
    materials: list[str] = Field(default_factory=list)
    intended_use: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list, description="Explicitly stated features (snake_case)")
    lacks: list[str] = Field(
        default_factory=list, description="Features the description explicitly says are absent"
    )
    specs: dict[str, float] = Field(
        default_factory=dict, description="Numeric specs stated in the text (battery_hours, capacity_mah, ...)"
    )


class ExtractionInfo(BaseModel):
    method: Literal["rules", "rules+llm"] = "rules"
    model: Optional[str] = None
    source_hash: str = ""
    notes: list[str] = Field(
        default_factory=list, description="What the guardrails dropped, and any fallbacks used"
    )


class ProcessedProduct(RawProduct):
    """A catalog row plus structured attributes. This is what processed_products.json stores."""

    attributes: Attributes = Field(default_factory=Attributes)
    # Filled by review_analyzer.py in Phase 7. Loosely typed until then.
    review_evidence: list[dict[str, Any]] = Field(default_factory=list)
    review_warnings: list[str] = Field(default_factory=list)
    extraction: ExtractionInfo = Field(default_factory=ExtractionInfo)


    

# ======================================================================
# Phase 4: structured query
# ======================================================================
class Budget(BaseModel):
    max: Optional[float] = None
    min: Optional[float] = None
    firmness: Literal["soft", "standard", "firm"] = Field(
        "standard",
        description="soft = 'ideally/around'; standard = 'under $X'; firm = 'no more than / can't exceed'",
    )


class StructuredQuery(BaseModel):
    """What the shopper wants, in a form the retriever and ranker can compute on."""

    raw_message: str
    product_type: Optional[str] = Field(None, description="e.g. 'running shoe'")
    category: Optional[str] = Field(None, description="A catalog category, e.g. 'footwear'")
    budget: Budget = Field(default_factory=Budget)
    hard_requirements: list[str] = Field(default_factory=list, description="Must-haves (canonical tags)")
    soft_preferences: list[str] = Field(default_factory=list, description="Nice-to-haves")
    intended_use: list[str] = Field(default_factory=list)
    brand: Optional[str] = None
    tone_signals: list[str] = Field(default_factory=list, description="cheap / basic / premium / best")
    vagueness_signals: list[str] = Field(default_factory=list)
    is_gift: bool = False
    recipient: Optional[str] = None
    parser: Literal["rules", "rules+llm"] = "rules"
    notes: list[str] = Field(default_factory=list, description="What the guards dropped, fallbacks used")