"""Aggregate real step outcomes, including missing and provider-blocked checks."""

import json
import os
from typing import Any, Dict, List, Tuple


SCOPE_CHECKS = {
    "core": {
        "pytest": "pytest",
        "mypy": "mypy",
        "demo": "demo",
        "help": "help",
        "dogfood_e2e": "dogfood-e2e",
        "regression": "regression",
        "monitor": "monitor",
    },
    "live": {
        "provider": "provider",
        "pytest_llm": "pytest-llm",
        "dogfood": "dogfood",
    },
}


def summarize(steps: Dict[str, Any], scope: str = "core") -> Tuple[List[str], str]:
    if scope not in SCOPE_CHECKS:
        raise ValueError("DOGFOOD_SCOPE must be explicitly set to core or live")
    failed = []
    lines = [f"## {scope.title()} Dogfood Results", "", "| Check | Outcome |", "| --- | --- |"]
    for step_id, name in SCOPE_CHECKS[scope].items():
        # `conclusion` is success for continue-on-error steps; `outcome` is the
        # original command result. A skipped or missing check is not evidence.
        outcome = steps.get(step_id, {}).get("outcome", "not run")
        if outcome != "success":
            failed.append(name)
        if outcome == "skipped" and step_id in {"pytest_llm", "dogfood"} and steps.get("provider", {}).get("outcome") != "success":
            outcome = "blocked by provider preflight"
        lines.append(f"| {name} | {outcome} |")
    if "provider" in failed:
        lines.extend([
            "",
            "Provider readiness failed. Live results are unavailable, not passing. "
            "See provider-output.txt for the infrastructure failure category.",
        ])
    if scope == "core":
        lines.extend(["", "Core results cover deterministic checks only. Live provider coverage is reported separately."])
    return failed, "\n".join(lines) + "\n"


def main() -> None:
    failed, summary = summarize(
        json.loads(os.environ["CHECK_STEPS"]), scope=os.environ.get("DOGFOOD_SCOPE", "")
    )
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"has_failures={'true' if failed else 'false'}\n")
        output.write(f"failed={','.join(failed)}\n")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
        output.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
