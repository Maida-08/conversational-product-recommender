from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Conversational Product Search & Recommendation",
    description="Retrieval, ranking and recommendation layer behind an AI shopping agent. "
                "Runs on a SYNTHETIC development catalog.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"], summary="Liveness check")
def health() -> dict:
    return {
        "status": "ok",
        "llm_configured": settings.llm_enabled,
        "tracing_enabled": settings.langsmith_tracing,
        "web_enrichment_enabled": settings.tavily_enabled,
    }