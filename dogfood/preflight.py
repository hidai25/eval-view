"""Bounded provider check before running the paid dogfood suite.

Failures here are infrastructure/configuration failures, not agent regressions.
Only structured error metadata is logged; credentials and raw responses are not.
"""

import json
import os
from typing import Any, Dict

from openai import OpenAI


def classify_provider_error(exc: Exception) -> Dict[str, Any]:
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    error_type = error.get("type") if isinstance(error, dict) else None
    if code in {"credit_balance_exhausted", "insufficient_quota"} or error_type == "insufficient_quota":
        category = "provider_quota"
    elif status in {401, 403}:
        category = "provider_auth"
    elif status == 429:
        category = "provider_rate_limit"
    elif status == 404:
        category = "provider_model_access"
    elif status is None or status >= 500:
        category = "provider_unavailable"
    else:
        category = "provider_request_error"
    return {"category": category, "status_code": status, "code": code, "type": error_type}


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print(json.dumps({"provider": "openai", "category": "provider_not_configured"}))
        return 1

    from evalview.core.llm_provider import LLMProvider, PROVIDER_CONFIGS

    # The chat agent/live evaluator default and dogfood/config.yaml judge must
    # both be usable. A successful /models listing does not validate credits.
    models = dict.fromkeys([
        os.getenv("EVAL_MODEL") or PROVIDER_CONFIGS[LLMProvider.OPENAI].default_model,
        "gpt-4o",
    ])
    with OpenAI(timeout=20.0, max_retries=0) as client:
        for model in models:
            try:
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Reply OK."}],
                    max_completion_tokens=16,
                )
            except Exception as exc:
                print(json.dumps({"provider": "openai", "model": model, **classify_provider_error(exc)}))
                return 1
    print(json.dumps({"provider": "openai", "models": list(models), "category": "ready"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
