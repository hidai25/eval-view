"""EvalView - Testing framework for multi-step AI agents."""

__version__ = "0.7.0"

# Public API — importable as ``from evalview import gate``
from evalview.api import gate, gate_async, GateResult, DiffStatus  # noqa: F401

# Model comparison API — importable as ``import evalview; evalview.run_eval(...)``
from evalview.compare import (  # noqa: F401
    run_eval,
    score,
    compare_models,
    ModelResult,
    print_comparison_table,
)
