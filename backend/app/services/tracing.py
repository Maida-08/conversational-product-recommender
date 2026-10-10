
"""Optional, failure-safe tracing for pipeline stages and LLM calls."""
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, TypeVar
from uuid import uuid4

F = TypeVar("F", bound=Callable[..., Any])

_enabled_override: bool | None = None
_client = None
_runners: dict[str, Any] = {}
_current_run: ContextVar[str | None] = ContextVar(
    "tracing_current_run", default=None
)

_MAX_STR = 350
_MAX_ITEMS = 5
_MAX_DICT_ITEMS = 12
_MAX_DEPTH = 3


def get_settings():
    from app.config import get_settings as load_settings
    return load_settings()


def configure(enabled=None, client=None):
    """Set a testing override, or reset to environment-based configuration."""
    global _enabled_override, _client
    _enabled_override = enabled
    _client = client
    _runners.clear()


def is_enabled():
    """Tracing requires the flag and key, unless explicitly overridden."""
    if _enabled_override is not None:
        return bool(_enabled_override)

    try:
        settings = get_settings()
        return bool(settings.langsmith_tracing and settings.langsmith_api_key)
    except Exception:
        return False


def _get_client():
    """Return the configured client, creating one only when needed."""
    global _client

    if _client is not None:
        return _client

    try:
        settings = get_settings()
        if not (settings.langsmith_tracing and settings.langsmith_api_key):
            return None

        from langsmith import Client
        _client = Client(api_key=settings.langsmith_api_key)
        return _client
    except Exception:
        return None


def _brief(value, depth=0):
    """Convert values to bounded, safe summaries for tracing."""
    try:
        if depth >= _MAX_DEPTH:
            if isinstance(value, dict):
                return "<dict>"
            if isinstance(value, (list, tuple)):
                return "<list>"
            if value is None or isinstance(value, (bool, int, float)):
                return value
            return f"<{type(value).__name__}>"

        if isinstance(value, str):
            return value if len(value) <= _MAX_STR else value[:_MAX_STR - 3] + "..."

        if value is None or isinstance(value, (bool, int, float)):
            return value

        if isinstance(value, dict):
            result = {}
            items = list(value.items())
            for key, item in items[:_MAX_DICT_ITEMS]:
                result[str(key)] = _brief(item, depth + 1)
            if len(items) > _MAX_DICT_ITEMS:
                result["..."] = f"{len(items) - _MAX_DICT_ITEMS} more keys"
            return result

        if isinstance(value, (list, tuple)):
            result = [_brief(item, depth + 1) for item in value[:_MAX_ITEMS]]
            if len(value) > _MAX_ITEMS:
                result.append(f"... +{len(value) - _MAX_ITEMS} more")
            return result

        if hasattr(value, "model_dump"):
            return _brief(value.model_dump(), depth + 1)

        # Catch objects whose repr implementation is broken.
        repr(value)
        return f"<{type(value).__name__}>"

    except Exception:
        return "<unprintable>"


def _process_inputs(inputs):
    if not isinstance(inputs, dict):
        return {"input": _brief(inputs)}

    return {
        str(key): _brief(value)
        for key, value in inputs.items()
        if key != "self"
    }


def _runner(name, run_type="chain"):
    """Return a recorder; tracing failures must never break the application."""

    def record(inputs, outputs=None, error=None, parent_run_id=None):
        if not is_enabled():
            return None

        client = _get_client()
        if client is None:
            return None

        run_id = uuid4()

        try:
            settings = get_settings()
            project = getattr(
                settings,
                "langsmith_project",
                "conversational-product-recommender",
            )

            client.create_run(
                id=run_id,
                name=name,
                run_type=run_type,
                inputs=_process_inputs(inputs),
                project_name=project,
                session_name=project,
                parent_run_id=parent_run_id,
                error=str(error) if error else None,
            )

            if outputs is not None or error is not None:
                client.update_run(
                    run_id=run_id,
                    outputs=_brief(outputs) if outputs is not None else None,
                    error=str(error) if error else None,
                )

            return run_id

        except Exception:
            return None

    return record


def traced(name: str, run_type: str = "chain") -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not is_enabled():
                return fn(*args, **kwargs)

            parent = _current_run.get()

            try:
                run_id = _runner(name, run_type)(
                    {"args": args, **kwargs},
                    parent_run_id=parent,
                )
            except Exception:
                # A tracing setup failure must not prevent the real function.
                return fn(*args, **kwargs)
            token = _current_run.set(run_id if run_id else parent)
            try:
                result = fn(*args, **kwargs)

                client = _get_client()
                if run_id and client is not None:
                    try:
                        client.update_run(
                            run_id=run_id,
                            outputs=_brief(result),
                        )
                    except Exception:
                        pass

                return result

            except Exception as exc:
                client = _get_client()
                if run_id and client is not None:
                    try:
                        client.update_run(run_id=run_id, error=str(exc))
                    except Exception:
                        pass
                raise

            finally:
                _current_run.reset(token)

        return wrapper  # type: ignore[return-value]

    return decorator


def wrap_llm_client(client):
    """Trace OpenAI-compatible chat completion calls as child LLM runs."""
    if not is_enabled():
        return client

    try:
        completions = client.chat.completions
        original_create = completions.create

        if getattr(original_create, "_tracing_wrapped", False):
            return client

       
        @wraps(original_create)
        def traced_create(*args, **kwargs):
            parent = _current_run.get()
            inputs = {"args": args, **kwargs}
            run_id = None

            try:
                run_id = _runner(
                    "openai.chat.completions.create", "llm"
                )(inputs, parent_run_id=parent)
            except Exception:
                pass

            try:
                result = original_create(*args, **kwargs)
                output = _brief(result)

                usage = getattr(result, "usage", None)
                if usage is not None:
                    output = {
                        "response": output,
                        "usage": _brief(
                            usage.model_dump()
                            if hasattr(usage, "model_dump")
                            else usage
                        ),
                    }

                active_client = _get_client()
                if run_id and active_client is not None:
                    try:
                        active_client.update_run(
                            run_id=run_id,
                            outputs=output,
                        )
                    except Exception:
                        pass

                return result

            except Exception as exc:
                active_client = _get_client()
                if run_id and active_client is not None:
                    try:
                        active_client.update_run(
                            run_id=run_id,
                            error=str(exc),
                        )
                    except Exception:
                        pass
                raise

        traced_create._tracing_wrapped = True
        completions.create = traced_create

    except Exception:
        # Wrapping is optional; preserve the original client on any failure.
        pass

    return client
