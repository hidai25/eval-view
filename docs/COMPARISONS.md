# Comparisons

Use these guides when you are deciding where EvalView fits in your stack.

## Comparison Guides

- [EvalView vs LangSmith](VS_LANGSMITH.md)
- [EvalView vs Langfuse](VS_LANGFUSE.md)
- [EvalView vs Braintrust](VS_BRAINTRUST.md)
- [EvalView vs DeepEval](VS_DEEPEVAL.md)
- [EvalView vs Promptfoo](VS_PROMPTFOO.md)

## Short Version

- Use **EvalView** when the core problem is **regression testing for agent behavior**
- Use observability platforms when the core problem is **trace collection and production debugging**
- Use broader eval platforms when the core problem is **scoring, datasets, and experimentation**

EvalView is strongest when you need:
- golden baseline testing
- tool-call and trajectory diffs
- agent regression gates in CI/CD
- fast draft suite generation from a live agent or logs
