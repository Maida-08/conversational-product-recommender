"""Tests for tracing: optional, failure-proof, readable, and one trace tree per chat message."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langsmith import Client
from pydantic import BaseModel

from app.data.loader import load_catalog
from app.main import app
from app.services import tracing
from app.services.attribute_extractor import rules_extract
from app.services.pipeline import Engine
from app.services.tracing import _brief, traced, wrap_llm_client

SCEN_D = "Looking for cheap wireless earbuds, just need something basic"
STAGES = ["query_understanding", "ambiguity_gate", "retrieval", "ranking_and_review_evidence",
          "recommendation_and_upsell", "response_generation"]


class Recorder:
    """A stand-in for the LangSmith client that records what the SDK would send."""
    def __init__(self, fail=False):
        self.created, self.updated = [], []
        self.client = MagicMock(spec=Client)
        self.client.tracing_queue = None

        def create(**kw):
            if fail:
                raise RuntimeError("LangSmith is down")
            self.created.append(kw)
        self.client.create_run.side_effect = create
        self.client.update_run.side_effect = lambda *a, **kw: self.updated.append(kw)

    @property
    def names(self):
        return [r["name"] for r in self.created]


@pytest.fixture()
def recorder():
    rec = Recorder()
    tracing.configure(enabled=True, client=rec.client)
    yield rec
    tracing.configure()                                   # always reset: tracing must not leak between tests


@pytest.fixture(scope="module")
def engine():
    return Engine([rules_extract(r) for r in load_catalog()], client=None, data_source="test")


# ------------------------------------------------------------------ optional by construction
def test_disabled_tracing_calls_the_function_directly():
    tracing.configure(enabled=False)
    try:
        @traced("x")
        def f(a, b=2):
            return a + b
        assert f(1, b=5) == 6 and tracing._runners == {}     # the SDK was never even touched
    finally:
        tracing.configure()


@pytest.mark.parametrize("flag,key,expected", [(True, "", False), (True, "k", True), (False, "k", False), (False, "", False)])
def test_tracing_needs_both_the_flag_and_a_key(monkeypatch, flag, key, expected):
    tracing.configure()
    monkeypatch.setattr(tracing, "get_settings", lambda: SimpleNamespace(langsmith_tracing=flag, langsmith_api_key=key))
    assert tracing.is_enabled() is expected


def test_health_reports_whether_tracing_is_effectively_on(monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    monkeypatch.setattr(main, "tracing_enabled", lambda: True)
    assert TestClient(app).get("/health").json()["tracing_enabled"] is True


# ------------------------------------------------------------------ one message = one trace tree
def test_a_recommendation_is_one_trace_tree_with_a_run_per_stage(engine, recorder):
    engine.respond(SCEN_D)
    assert recorder.names[0] == "chat_pipeline" and recorder.names[1:] == STAGES
    root_id = str(recorder.created[0]["id"])
    assert recorder.created[0].get("parent_run_id") is None
    assert all(str(r["parent_run_id"]) == root_id for r in recorder.created[1:])
    assert {r.get("session_name") for r in recorder.created} == {tracing.get_settings().langsmith_project}


def test_a_clarification_trace_shows_it_stopped_before_retrieval(engine, recorder):
    engine.respond("Something nice for my mom")
    assert recorder.names == ["chat_pipeline", "query_understanding", "ambiguity_gate", "response_generation"]


def test_tracing_does_not_change_the_answer(engine, recorder):
    traced_answer = engine.respond(SCEN_D).model_dump()
    tracing.configure(enabled=False)
    try:
        assert engine.respond(SCEN_D).model_dump() == traced_answer
    finally:
        tracing.configure()


# ------------------------------------------------------------------ readable, small traces
def test_traces_never_log_the_engine_or_the_catalog_and_stay_small(engine, recorder):
    engine.respond("I need a lightweight waterproof running shoe, ideally under $80")
    assert all("self" not in (r.get("inputs") or {}) for r in recorder.created)
    payload = json.dumps([r.get("inputs") for r in recorder.created] + [u.get("outputs") for u in recorder.updated],
                         default=str)
    assert len(payload) < 120_000 and "Aquaridge StormShield" in payload      # informative, but bounded


def test_the_retrieval_run_records_a_summary_not_the_whole_catalog(engine, recorder):
    engine.respond(SCEN_D)
    retrieval_inputs = json.dumps(next(r for r in recorder.created if r["name"] == "retrieval")["inputs"], default=str)
    assert len(retrieval_inputs) < 5_000


# ------------------------------------------------------------------ failures never break a request
def test_a_langsmith_outage_never_breaks_a_chat_request(engine):
    rec = Recorder(fail=True)
    tracing.configure(enabled=True, client=rec.client)
    try:
        r = engine.respond(SCEN_D)
    finally:
        tracing.configure()
    assert r.type == "recommendation" and r.recommendations[0].product_id == "P014"


def test_setup_problems_fall_back_to_the_plain_function(monkeypatch):
    tracing.configure(enabled=True, client=MagicMock())
    monkeypatch.setattr(tracing, "_runner", lambda *a: (_ for _ in ()).throw(ImportError("langsmith missing")))
    try:
        @traced("x")
        def f():
            return "ok"
        assert f() == "ok"
    finally:
        tracing.configure()


def test_a_failing_stage_still_raises_and_is_recorded_as_an_error(recorder):
    @traced("exploding_stage")
    def boom():
        raise ValueError("stage failed")
    with pytest.raises(ValueError, match="stage failed"):
        boom()
    assert recorder.names == ["exploding_stage"] and any(u.get("error") for u in recorder.updated)


# ------------------------------------------------------------------ summaries
def test_long_strings_are_truncated():
    assert _brief("x" * 1000).endswith("...") and len(_brief("x" * 1000)) < 400


def test_long_lists_and_dicts_are_clipped():
    assert _brief(list(range(50)))[-1] == "... +45 more"
    out = _brief({f"k{i}": i for i in range(30)})
    assert len(out) == 13 and "more keys" in out["..."]


def test_nesting_is_limited():
    assert _brief({"a": {"b": {"c": {"d": 1}}}}) == {"a": {"b": {"c": "<dict>"}}}


def test_pydantic_models_are_summarised():
    class M(BaseModel):
        name: str
        items: list[int]
    assert _brief(M(name="n", items=list(range(20)))) == {"name": "n", "items": [0, 1, 2, 3, 4, "... +15 more"]}


def test_summaries_never_raise_even_for_hostile_objects():
    class Hostile:
        def __repr__(self):
            raise RuntimeError("no repr for you")
    assert _brief(Hostile()) == "<unprintable>" and _brief([Hostile()]) == ["<unprintable>"]


# ------------------------------------------------------------------ LLM calls
def test_wrapping_is_a_no_op_when_tracing_is_off():
    tracing.configure(enabled=False)
    try:
        client = object()
        assert wrap_llm_client(client) is client
    finally:
        tracing.configure()


def test_a_wrapping_failure_returns_the_original_client(monkeypatch, recorder):
    import langsmith.wrappers as wrappers
    monkeypatch.setattr(wrappers, "wrap_openai", lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
    client = object()
    assert wrap_llm_client(client) is client


def test_llm_calls_appear_inside_the_stage_that_made_them(recorder):
    import openai
    from openai.types.chat import ChatCompletion
    reply = ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52}})
    client = openai.OpenAI(api_key="x", base_url="http://127.0.0.1:9", max_retries=0)
    client.chat.completions.create = lambda **kw: reply      # stands in for the network call
    wrapped = wrap_llm_client(client)

    @traced("response_generation")
    def stage():
        return wrapped.chat.completions.create(model="m", messages=[{"role": "user", "content": "hello there"}])

    stage()
    llm = next(r for r in recorder.created if r["run_type"] == "llm")
    parent = next(r for r in recorder.created if r["id"] == llm["parent_run_id"])
    assert parent["name"] == "response_generation" and "hello there" in json.dumps(llm["inputs"], default=str)
    assert any("total_tokens" in json.dumps(u, default=str) for u in recorder.updated)