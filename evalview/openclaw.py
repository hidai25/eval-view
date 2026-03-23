"""OpenClaw integration helpers for EvalView.

Provides high-level functions for autonomous agent loops that use EvalView
as a regression gate.  These are designed to be called from OpenClaw claws,
autonomous coding agents, or any loop that modifies agent code and needs
to verify the changes didn't break anything.

Quick start::

    from evalview.openclaw import gate_or_revert

    # After making a code change:
    ok = gate_or_revert(test_dir="tests/")
    # Returns True if change is safe, False if it was reverted.

For more control::

    from evalview.openclaw import check_and_decide, accept_change

    decision = check_and_decide(test_dir="tests/")
    if decision.action == "revert":
        # Change broke something — already reverted
        print(decision.reason)
    elif decision.action == "accept":
        # Looks intentional — call accept_change to snapshot
        accept_change(decision)
    else:
        # Clean pass — continue working
        pass
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from evalview.api import DiffStatus, GateResult, gate

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    """Result of check_and_decide() — what to do after a gate check.

    Attributes:
        action: One of "continue", "revert", "accept", "review".
        reason: Human-readable explanation of why this action was chosen.
        gate_result: The underlying GateResult for inspection.
        changed_tests: Names of tests that changed (non-passing).
        reverted: Whether the change was already reverted.
    """

    action: str  # "continue" | "revert" | "accept" | "review"
    reason: str
    gate_result: GateResult
    changed_tests: List[str]
    reverted: bool = False


def gate_or_revert(
    test_dir: str = "tests",
    revert_cmd: Optional[str] = None,
    timeout: float = 30.0,
    quick: bool = False,
) -> bool:
    """Run regression gate.  Revert automatically on regression.

    This is the simplest integration point — call it after every code change.

    Args:
        test_dir: Path to test directory.
        revert_cmd: Shell command to revert changes.  Default: ``git checkout -- .``
        timeout: Per-test timeout in seconds.
        quick: If True, skip LLM judge — deterministic checks only ($0, fast).

    Returns:
        True if the change is safe (no regressions), False if it was reverted.

    Example::

        # In an autonomous coding loop:
        make_code_change()
        if not gate_or_revert("tests/", quick=True):
            # Change was reverted — try a different approach
            try_alternative_approach()
    """
    result = gate(test_dir=test_dir, timeout=timeout, quick=quick)

    if result.passed:
        return True

    # Has regressions — revert
    cmd = revert_cmd or "git checkout -- ."
    logger.info(f"Regression detected, reverting with: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Revert command failed: {e}")

    return False


def check_and_decide(
    test_dir: str = "tests",
    strict: bool = False,
    auto_revert: bool = True,
    revert_cmd: Optional[str] = None,
    timeout: float = 30.0,
) -> GateDecision:
    """Run gate and return a decision with full context.

    More granular than gate_or_revert — distinguishes between regressions
    (revert), intentional improvements (accept), and minor changes (review).

    Args:
        test_dir: Path to test directory.
        strict: If True, treat any change as a failure (not just regressions).
        auto_revert: If True, automatically revert on regression.
        revert_cmd: Shell command to revert.  Default: ``git checkout -- .``
        timeout: Per-test timeout in seconds.

    Returns:
        GateDecision with action, reason, and full gate result.
    """
    fail_on = {DiffStatus.REGRESSION}
    if strict:
        fail_on = {DiffStatus.REGRESSION, DiffStatus.TOOLS_CHANGED, DiffStatus.OUTPUT_CHANGED}

    result = gate(test_dir=test_dir, fail_on=fail_on, timeout=timeout)
    changed = [d.test_name for d in result.diffs if not d.passed]

    # All clean
    if result.summary.regressions == 0 and result.summary.tools_changed == 0 and result.summary.output_changed == 0:
        return GateDecision(
            action="continue",
            reason=f"All {result.summary.total} tests passed.",
            gate_result=result,
            changed_tests=[],
        )

    # Regressions — must revert
    if result.summary.regressions > 0:
        reverted = False
        if auto_revert:
            cmd = revert_cmd or "git checkout -- ."
            try:
                subprocess.run(cmd, shell=True, check=True, capture_output=True)
                reverted = True
            except subprocess.CalledProcessError:
                pass

        regression_names = [
            d.test_name for d in result.diffs
            if d.status == DiffStatus.REGRESSION
        ]
        return GateDecision(
            action="revert",
            reason=f"Regression in {len(regression_names)} test(s): {', '.join(regression_names)}",
            gate_result=result,
            changed_tests=changed,
            reverted=reverted,
        )

    # Tools/output changed but scores improved or stable — suggest accepting
    improving = [
        d for d in result.diffs
        if not d.passed and d.score_delta >= 0
    ]
    if improving:
        names = [d.test_name for d in improving]
        return GateDecision(
            action="accept",
            reason=f"Changes look intentional (scores improved) in: {', '.join(names)}. Run accept_change() to snapshot.",
            gate_result=result,
            changed_tests=changed,
        )

    # Changes with score drops (but not regressions) — needs review
    return GateDecision(
        action="review",
        reason=f"{len(changed)} test(s) changed: {', '.join(changed)}. Scores declined slightly — review before accepting.",
        gate_result=result,
        changed_tests=changed,
    )


def accept_change(
    decision: GateDecision,
    test_dir: str = "tests",
) -> int:
    """Accept an intentional change by snapshotting new baselines.

    Call this after check_and_decide returns action="accept".

    Args:
        decision: The GateDecision from check_and_decide.
        test_dir: Path to test directory.

    Returns:
        Number of tests snapshotted.
    """
    count = 0
    for test_name in decision.changed_tests:
        try:
            subprocess.run(
                ["evalview", "snapshot", "--path", test_dir, "--test", test_name],
                check=True,
                capture_output=True,
            )
            count += 1
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to snapshot {test_name}: {e}")
    return count


def install_skill(target_dir: str = ".") -> str:
    """Copy the evalview-gate skill to a target directory.

    This makes the skill available to any OpenClaw claw working in that
    directory.

    Args:
        target_dir: Directory to install the skill into.
            Creates a ``skills/`` subdirectory if needed.

    Returns:
        Path to the installed skill file.
    """
    import shutil

    skill_source = Path(__file__).parent / "skills" / "builtin" / "evalview-gate.md"
    target = Path(target_dir) / "skills" / "evalview-gate.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(skill_source, target)
    return str(target)
