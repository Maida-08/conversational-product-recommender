"""Tests for the HTTP API: GET /health, POST /api/v1/chat, validation, errors, OpenAPI."""
import pytest
from fastapi.testclient import TestClient

from app.api.routes import chat as chat_module
from app.api.routes.chat import engine_dependency
from app.config import get_settings
from app.data.loader import load_catalog
from app.main import app
from app.services.attribute_extractor import rules_extract
from app.services.pipeline import Engine

needs_catalog = pytest.mark.skipif(not get_settings().catalog_csv.exists(), reason="synthetic catalog missing")

SCEN_D = "Looking for cheap wireless earbuds, just need something basic"


@pytest.fixture(scope="module")
def engine():
    return Engine([rules_extract(r) for r in load_catalog()], client=None, data_source="test")


@pytest.fixture()
def client(engine):
    app.dependency_overrides[engine_dependency] = lambda: engine          # rules-only: never calls an LLM
    yield TestClient(app)
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- health
def test_health_reports_configuration_without_leaking_secrets():
    r = TestClient(app).get("/health")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok"
    assert {"catalog_found", "processed_data_found", "llm_configured", "tracing_enabled"} <= set(body)
    assert all(isinstance(v, (bool, str)) for v in body.values()) and "key" not in str(body).lower().replace("llm_configured", "")


# ------------------------------------------------------------------------------ chat
@needs_catalog
def test_chat_returns_a_valid_recommendation_response(client):
    r = client.post("/api/v1/chat", json={"message": SCEN_D})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "recommendation" and body["recommendations"][0]["role"] == "primary"
    assert body["recommendations"][0]["product_id"] == "P014" and body["upsell"]["product_id"] == "P015"
    assert {"parsed_query", "specificity", "ranking", "wording"} <= set(body["details"])
    assert body["assistant_message"].startswith("I'd go with the SonicPulse Wireless Earbuds Lite")


@needs_catalog
def test_chat_returns_a_clarification_for_a_vague_message(client):
    body = client.post("/api/v1/chat", json={"message": "Something nice for my mom"}).json()
    assert body["type"] == "clarification" and body["recommendations"] == [] and body["clarification"]["options"]


@needs_catalog
def test_chat_accepts_history(client):
    r = client.post("/api/v1/chat", json={"message": SCEN_D, "history": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]})
    assert r.status_code == 200 and r.json()["details"]["history_turns"] == 2


@pytest.mark.parametrize("payload", [
    {}, {"message": ""}, {"message": "   "}, {"message": "x" * 1001}, {"message": 5},
    {"message": "hi", "history": [{"role": "robot", "content": "beep"}]},
    {"message": "hi", "history": "not a list"},
])
def test_invalid_requests_are_rejected_with_422(client, payload):
    assert client.post("/api/v1/chat", json=payload).status_code == 422


def test_unexpected_pipeline_failure_returns_a_generic_500():
    class Boom:
        def respond(self, *a, **k):
            raise RuntimeError("secret internal detail")
    app.dependency_overrides[engine_dependency] = lambda: Boom()
    try:
        r = TestClient(app, raise_server_exceptions=False).post("/api/v1/chat", json={"message": "hi"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 500 and "secret" not in r.text and "Something went wrong" in r.json()["detail"]


def test_missing_catalog_is_a_503_not_a_crash(monkeypatch):
    def missing():
        raise FileNotFoundError("catalog.csv")
    monkeypatch.setattr(chat_module, "get_engine", missing)
    r = TestClient(app).post("/api/v1/chat", json={"message": "hi"})
    assert r.status_code == 503 and "catalog" in r.json()["detail"].lower()


@needs_catalog
def test_examples_endpoint_lists_the_demo_queries(client):
    ex = client.get("/api/v1/examples").json()
    assert "Something nice for my mom" in ex and len(ex) >= 5


# --------------------------------------------------------------------------- OpenAPI / Swagger
def test_openapi_documents_both_endpoints_and_the_response_schema():
    spec = TestClient(app).get("/openapi.json").json()
    assert "/health" in spec["paths"] and "/api/v1/chat" in spec["paths"]
    assert {"ChatRequest", "ChatResponse", "ProductCard", "ChatDetails"} <= set(spec["components"]["schemas"])
    content = spec["paths"]["/api/v1/chat"]["post"]["requestBody"]["content"]["application/json"]
    assert {"specific_with_budget", "ambiguous", "review_only_signal", "upsell", "no_match"} <= set(content["examples"])


def test_swagger_ui_is_served():
    r = TestClient(app).get("/docs")
    assert r.status_code == 200 and "swagger" in r.text.lower()