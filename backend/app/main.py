from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.tracing import is_enabled as tracing_enabled
from app.api.routes.chat import router as chat_router
from app.config import get_settings

settings = get_settings()

TAGS = [
    {"name": "system", "description": "Liveness and configuration status."},
    {"name": "chat", "description": "The conversational product search and recommendation pipeline."},
]

app = FastAPI(
    title="Conversational Product Search & Recommendation",
    description=(
        "Retrieval, ranking and recommendation layer behind an AI shopping agent. Runs on a **synthetic** "
        "development catalog (see data/SYNTHETIC_DATA.md).\n\n"
        "Try `POST /api/v1/chat` below: the request body has five ready-made examples (open the "
        "'Examples' dropdown)."),
    version="1.0.0",
    openapi_tags=TAGS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)


@app.get("/health", tags=["system"], summary="Liveness check and configuration status")
def health() -> dict:
    return {
        "status": "ok",
        "catalog_found": settings.catalog_csv.exists() or settings.processed_products.exists(),
        "processed_data_found": settings.processed_products.exists(),
        "llm_configured": settings.llm_enabled,
        "tracing_enabled": tracing_enabled(),
        "web_enrichment_enabled": settings.tavily_enabled,
    }