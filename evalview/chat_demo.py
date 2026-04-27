"""Scripted chat-mode demos used for marketing videos and quick walkthroughs.

run_demo() renders pre-baked scenarios without any LLM calls so playback
is instant and reproducible. Extracted from chat.py to keep that module
focused on the live chat loop.
"""

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from evalview.chat_runtime import print_banner, print_separator


async def run_demo(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    style: int = 1,
) -> None:
    """Run a scripted demo for marketing videos.

    Uses pre-baked responses for instant, consistent playback.
    Perfect for recording demos - no LLM calls, no waiting.
    Fully self-contained - no subprocess calls.

    Styles:
        1: "3am panic" - Emotional, relatable crisis scenario
        2: "Instant action" - One-liner straight to demo
        3: "Cost explosion" - Money-focused shock and relief
        4: "Chat UI" - Showcase interactive chat features
    """
    import time
    from rich.live import Live

    console = Console()

    # Style-specific banners
    banner_subtitles = {
        1: "Demo: The 3am Panic",
        2: "Demo: LangGraph Agent",
        3: "Demo: Cost Explosion",
        4: "Interactive Chat",
    }
    print_banner(console, banner_subtitles.get(style, "Demo Mode"))

    time.sleep(0.5)

    def show_user_input(text: str) -> None:
        """Show simulated user input with typing effect."""
        console.print()
        console.print("[bold green]You[/bold green]", end=" ")
        for char in text:
            console.print(char, end="")
            time.sleep(0.02)
        console.print()

    def show_thinking(duration: float) -> None:
        """Show thinking animation."""
        with Live(console=console, refresh_per_second=10, transient=True) as live:
            for j in range(int(duration * 10)):
                dots = "." * ((j % 3) + 1)
                live.update(Text(f"  Thinking{dots}", style="dim"))
                time.sleep(0.1)

    def show_response(text: str, tokens: int, duration: float) -> None:
        """Show response with stats."""
        print_separator(console)
        console.print(f"[dim]  {duration:.1f}s  │  {tokens:,} tokens[/dim]")
        print_separator(console)
        console.print()
        console.print("[bold cyan]EvalView[/bold cyan]")
        console.print(Markdown(text))

    def show_regression_report(
        results: list[tuple[str, str, str]],
        cost_old: str,
        cost_new: str,
        cost_pct: str,
        latency_old: str,
        latency_new: str,
        latency_pct: str,
    ) -> None:
        """Show inline regression report."""
        console.print()
        console.print("[dim]Running regression check...[/dim]")
        time.sleep(0.3)
        for name, _, _ in results:
            console.print(f"[dim]  Analyzing {name}...[/dim]", end="")
            time.sleep(0.15)
            console.print("[dim] done[/dim]")
        console.print()

        # Report header
        console.print("━" * 68)
        console.print("[bold]                     Regression Report[/bold]")
        console.print("━" * 68)
        console.print()

        # Results
        for name, status, detail in results:
            if status == "PASSED":
                console.print(f"  [green]✓ PASSED[/green]         {name}")
            elif status == "TOOLS_CHANGED":
                console.print(f"  [yellow]⚠ TOOLS_CHANGED[/yellow]  {name:<16} {detail}")
            elif status == "OUTPUT_CHANGED":
                console.print(f"  [blue]~ OUTPUT_CHANGED[/blue] {name:<16} {detail}")
            elif status == "REGRESSION":
                console.print(f"  [red]✗ REGRESSION[/red]     {name:<16} {detail}")

        console.print()
        console.print(f"  Cost:    {cost_old} → {cost_new}  ({cost_pct})  [yellow]⚠[/yellow]")
        console.print(f"  Latency: {latency_old} → {latency_new}  ({latency_pct})  [yellow]⚠[/yellow]")
        console.print()
        console.print("━" * 68)
        console.print("  [red]❌ This would fail CI[/red]")
        console.print("━" * 68)

    # --- DEMO 1: "3am panic" - Agent broke, what changed? (verbose) ---
    if style == 1:
        show_user_input("My agent broke in production. Users are complaining. What changed since yesterday?")
        show_thinking(0.8)
        show_response(
            "Don't panic. Let me compare your current agent against yesterday's baseline.",
            523,
            0.8,
        )
        time.sleep(0.3)

        # Verbose test execution - focus on finding the regression
        tests: List[Dict[str, Any]] = [
            {
                "name": "auth-flow",
                "query": "Login with email test@example.com",
                "tools": ["validate_email", "create_session"],
                "answer": "Successfully logged in. Session token: sk-...",
                "status": "PASSED",
                "score": 95,
                "baseline_score": 95,
                "cost": 0.002,
                "latency": 0.8,
            },
            {
                "name": "search-query",
                "query": "Find products matching 'wireless headphones'",
                "tools": ["parse_query", "web_search", "db_search"],
                "answer": "Found 12 products matching your search...",
                "status": "TOOLS_CHANGED",
                "score": 89,
                "baseline_score": 91,
                "cost": 0.004,
                "latency": 1.2,
                "new_tool": "web_search",
            },
            {
                "name": "summarizer",
                "query": "Summarize customer feedback from last week",
                "tools": ["fetch_feedback", "analyze_sentiment"],
                "answer": "Customer feedback was mostly positive...",
                "status": "OUTPUT_CHANGED",
                "score": 82,
                "baseline_score": 88,
                "similarity": 72,
                "cost": 0.003,
                "latency": 0.9,
            },
            {
                "name": "checkout",
                "query": "Process order #12345 with payment",
                "tools": ["validate_cart", "process_payment"],
                "answer": "Error: Unable to process payment method...",
                "status": "REGRESSION",
                "score": 67,
                "baseline_score": 94,
                "cost": 0.005,
                "latency": 1.3,
            },
        ]

        console.print()
        for i, test in enumerate(tests):
            console.print(f"[bold]Test {i+1}/4:[/bold] {test['name']}")
            console.print(f"[dim]  Query:[/dim] \"{test['query']}\"")
            time.sleep(0.15)
            console.print(f"[dim]  Tools:[/dim] {' → '.join(test['tools'])}")
            console.print(f"[dim]  Answer:[/dim] \"{test['answer'][:45]}...\"")

            if test["status"] == "PASSED":
                console.print(f"  [green]✓ PASSED[/green]  score: {test['score']}  ${test['cost']:.3f}  {test['latency']}s")
            elif test["status"] == "TOOLS_CHANGED":
                console.print(f"  [yellow]⚠ TOOLS_CHANGED[/yellow]  +{test['new_tool']}  score: {test['score']}  ${test['cost']:.3f}  {test['latency']}s")
            elif test["status"] == "OUTPUT_CHANGED":
                console.print(f"  [blue]~ OUTPUT_CHANGED[/blue]  similarity: {test['similarity']}%  score: {test['score']}  ${test['cost']:.3f}  {test['latency']}s")
            elif test["status"] == "REGRESSION":
                drop = test['baseline_score'] - test['score']
                console.print(f"  [red]✗ REGRESSION[/red]  score: {test['baseline_score']} → {test['score']} [red](-{drop})[/red]  ${test['cost']:.3f}  {test['latency']}s")
            console.print()
            time.sleep(0.12)

        # Summary
        console.print("━" * 68)
        console.print("[bold]                        Summary[/bold]")
        console.print("━" * 68)
        console.print()
        console.print("  Tests:   [green]1 passed[/green]  [red]1 regression[/red]  [yellow]1 tools changed[/yellow]  [blue]1 output changed[/blue]")
        console.print("  Cost:    $0.014 total (was $0.008)")
        console.print("  Latency: 4.2s total (was 1.1s)")
        console.print()
        console.print("━" * 68)
        console.print("  [red]❌ checkout regressed: 94 → 67 (-27 points)[/red]")
        console.print("━" * 68)

        time.sleep(0.5)
        console.print()
        console.print("[bold green]Found it.[/bold green] The checkout flow broke. Fix it and run `evalview golden update checkout`")
        console.print("[dim]pip install evalview[/dim]")
        console.print()
        console.print("[dim]⭐ Star if this helped → github.com/hidai25/eval-view[/dim]\n")

    # --- DEMO 2: "LangGraph agent" - Real framework, verbose output ---
    elif style == 2:
        show_user_input("test my langgraph agent")
        show_thinking(0.5)
        show_response(
            "Running tests against your LangGraph agent on localhost:2024...",
            156,
            0.5,
        )
        time.sleep(0.3)

        # Verbose test execution output
        demo2_tests: List[Dict[str, Any]] = [
            {
                "name": "tavily-search",
                "query": "What is the weather in San Francisco?",
                "tools": ["tavily_search_results_json"],
                "answer": "The weather in San Francisco is currently 62°F...",
                "status": "PASSED",
                "score": 94,
                "cost": 0.003,
                "latency": 1.2,
            },
            {
                "name": "weather-query",
                "query": "Get the forecast for Tokyo this week",
                "tools": ["tavily_search_results_json"],
                "answer": "I don't have access to real-time weather...",
                "status": "FAILED",
                "score": 71,
                "expected_score": 88,
                "cost": 0.004,
                "latency": 0.9,
            },
            {
                "name": "rag-retrieval",
                "query": "Find documents about authentication",
                "tools": ["vector_search", "rerank_documents"],
                "answer": "Found 3 relevant documents about auth...",
                "status": "TOOLS_CHANGED",
                "score": 91,
                "cost": 0.002,
                "latency": 0.8,
                "new_tool": "rerank_documents",
            },
            {
                "name": "summarizer",
                "query": "Summarize the Q3 earnings report",
                "tools": ["tavily_search_results_json"],
                "answer": "Q3 earnings showed 15% revenue growth...",
                "status": "PASSED",
                "score": 96,
                "cost": 0.003,
                "latency": 1.1,
            },
        ]

        console.print()
        for i, test in enumerate(demo2_tests):
            console.print(f"[bold]Test {i+1}/4:[/bold] {test['name']}")
            console.print(f"[dim]  Query:[/dim] \"{test['query']}\"")
            time.sleep(0.2)
            console.print(f"[dim]  Tools:[/dim] {' → '.join(test['tools'])}")
            console.print(f"[dim]  Answer:[/dim] \"{test['answer'][:50]}...\"")

            if test["status"] == "PASSED":
                console.print(f"  [green]✓ PASSED[/green]  score: {test['score']}  ${test['cost']:.3f}  {test['latency']}s")
            elif test["status"] == "FAILED":
                console.print(f"  [red]✗ FAILED[/red]  score: {test['expected_score']} → {test['score']} (-{test['expected_score'] - test['score']})  ${test['cost']:.3f}  {test['latency']}s")
            elif test["status"] == "TOOLS_CHANGED":
                console.print(f"  [yellow]⚠ TOOLS_CHANGED[/yellow]  +{test['new_tool']}  score: {test['score']}  ${test['cost']:.3f}  {test['latency']}s")
            console.print()
            time.sleep(0.15)

        # Summary
        console.print("━" * 68)
        console.print("[bold]                        Summary[/bold]")
        console.print("━" * 68)
        console.print()
        console.print("  Tests:   [green]2 passed[/green]  [red]1 failed[/red]  [yellow]1 changed[/yellow]")
        console.print("  Cost:    $0.012 total")
        console.print("  Latency: 4.0s total (1.0s avg)")
        console.print()
        console.print("━" * 68)
        console.print("  [red]❌ 1 regression detected[/red]")
        console.print("━" * 68)

        time.sleep(0.5)
        console.print()
        console.print("[bold green]Done.[/bold green] Run `evalview golden update weather-query` after fixing.")
        console.print("[dim]pip install evalview[/dim]")
        console.print()
        console.print("[dim]⭐ Star if this helped → github.com/hidai25/eval-view[/dim]\n")

    # --- DEMO 3: "Cost explosion" - $847 bill shock (verbose, cost-focused) ---
    elif style == 3:
        show_user_input("My OpenAI bill is $847. Last month it was $12. What happened?")
        show_thinking(0.9)
        show_response(
            "$847 vs $12? That's a **70x spike**. Let me find which tests exploded.",
            634,
            0.9,
        )
        time.sleep(0.3)

        # Verbose test execution - focus on COSTS
        # Math: $12/month → $847/month at 30 runs/month
        # Old: $0.40/run, New: $28.23/run
        demo3_tests: List[Dict[str, Any]] = [
            {
                "name": "auth-flow",
                "query": "Authenticate user with OAuth",
                "tools": ["validate_token", "refresh_session"],
                "answer": "User authenticated successfully...",
                "status": "PASSED",
                "score": 96,
                "cost": 0.10,
                "baseline_cost": 0.10,
                "latency": 0.9,
            },
            {
                "name": "search-query",
                "query": "Search inventory for SKU-12345",
                "tools": ["query_parser", "db_lookup"],
                "answer": "Found 3 items matching SKU-12345...",
                "status": "PASSED",
                "score": 94,
                "cost": 0.10,
                "baseline_cost": 0.10,
                "latency": 1.1,
            },
            {
                "name": "doc-processor",
                "query": "Process and summarize the 50-page contract",
                "tools": ["pdf_extract", "chunk_text", "summarize"],
                "answer": "Contract summary: This agreement covers...",
                "status": "COST_SPIKE",
                "score": 91,
                "cost": 14.02,
                "baseline_cost": 0.10,
                "latency": 23.4,
            },
            {
                "name": "report-gen",
                "query": "Generate quarterly analytics report",
                "tools": ["fetch_metrics", "analyze_trends", "format_report"],
                "answer": "Q3 Report: Revenue up 12%, costs down...",
                "status": "COST_SPIKE",
                "score": 88,
                "cost": 14.01,
                "baseline_cost": 0.10,
                "latency": 19.8,
            },
        ]

        console.print()
        for i, test in enumerate(demo3_tests):
            console.print(f"[bold]Test {i+1}/4:[/bold] {test['name']}")
            console.print(f"[dim]  Query:[/dim] \"{test['query']}\"")
            time.sleep(0.15)
            console.print(f"[dim]  Tools:[/dim] {' → '.join(test['tools'])}")
            console.print(f"[dim]  Answer:[/dim] \"{test['answer'][:40]}...\"")

            if test["status"] == "PASSED":
                console.print(f"  [green]✓ PASSED[/green]  score: {test['score']}  [green]${test['cost']:.2f}[/green]  {test['latency']}s")
            elif test["status"] == "COST_SPIKE":
                cost_increase = test['cost'] / test['baseline_cost']
                console.print(f"  [red]💰 COST SPIKE[/red]  ${test['baseline_cost']:.2f} → [red]${test['cost']:.2f}[/red] ({cost_increase:.0f}x)  {test['latency']}s")
            console.print()
            time.sleep(0.12)

        # Summary - emphasize costs
        # New total: $0.10 + $0.10 + $14.02 + $14.01 = $28.23
        # Old total: $0.10 + $0.10 + $0.10 + $0.10 = $0.40
        console.print("━" * 68)
        console.print("[bold]                        Summary[/bold]")
        console.print("━" * 68)
        console.print()
        console.print("  Tests:   [green]2 passed[/green]  [red]2 cost spikes[/red]")
        console.print()
        console.print("  [bold]Cost breakdown:[/bold]")
        console.print("    auth-flow:     $0.10   [green](no change)[/green]")
        console.print("    search-query:  $0.10   [green](no change)[/green]")
        console.print("    doc-processor: [red]$14.02[/red]  [red](was $0.10 → 140x!)[/red]")
        console.print("    report-gen:    [red]$14.01[/red]  [red](was $0.10 → 140x!)[/red]")
        console.print()
        console.print("  [bold]Total:[/bold] $0.40 → [red]$28.23[/red] per run")
        console.print("  [bold]At 30 runs/month:[/bold] $12 → [red]$847[/red]")
        console.print()
        console.print("━" * 68)
        console.print("  [red]❌ 2 cost explosions detected[/red]")
        console.print("━" * 68)

        time.sleep(0.5)
        console.print()
        console.print("[bold green]Found it.[/bold green] Check doc-processor and report-gen for infinite loops or missing limits.")
        console.print("[dim]pip install evalview[/dim]")
        console.print()
        console.print("[dim]⭐ Star if this helped → github.com/hidai25/eval-view[/dim]\n")

    # --- DEMO 4: "Chat UI" - Showcase the interactive chat experience ---
    elif style == 4:
        term_width = console.width or 80

        def show_chat_box(text: str, typing: bool = True) -> None:
            """Show the beautiful chat box with fast typing effect."""
            # Top border
            title_text = "─ You "
            dashes = term_width - len(title_text) - 2
            console.print(f"[#22d3ee]╭{title_text}{'─' * dashes}╮[/#22d3ee]")
            console.print(f"[#22d3ee]│{' ' * (term_width - 2)}│[/#22d3ee]")

            # Type the text - FAST
            if typing:
                console.print("[#22d3ee]│[/#22d3ee] ", end="")
                for char in text:
                    console.print(char, end="", highlight=False)
                    time.sleep(0.012)  # Fast typing
                padding = term_width - len(text) - 4
                console.print(f"{' ' * padding}[#22d3ee]│[/#22d3ee]")
            else:
                padding = term_width - len(text) - 4
                console.print(f"[#22d3ee]│[/#22d3ee] {text}{' ' * padding}[#22d3ee]│[/#22d3ee]")

            console.print(f"[#22d3ee]│{' ' * (term_width - 2)}│[/#22d3ee]")
            console.print(f"[#22d3ee]╰{'─' * (term_width - 2)}╯[/#22d3ee]")
            console.print(f"[dim]  .../my-project{' ' * (term_width - 35)}llama3.2[/dim]")
            console.print(f"[dim]{' ' * (term_width - 8)}/model[/dim]")

        def show_slash_dropdown() -> None:
            """Show the slash command dropdown below the box."""
            # Show complete box with / inside
            console.print(Panel(
                "/",
                title="[bold #22d3ee]You[/bold #22d3ee]",
                title_align="left",
                border_style="#22d3ee",
                padding=(0, 1),
                expand=True
            ))
            time.sleep(0.15)

            # Dropdown appears BELOW the box
            commands = [
                ("/model", "Switch to a different model"),
                ("/docs", "Open EvalView documentation"),
                ("/cli", "Show CLI commands cheatsheet"),
                ("/help", "Show help and tips"),
            ]
            console.print("[dim]─── Slash Commands ───[/dim]")
            console.print("  [#22d3ee bold]▸ /model        [/#22d3ee bold] [dim]Switch to a different model[/dim]")
            for cmd, desc in commands[1:]:
                console.print(f"    [dim]{cmd:<14} {desc}[/dim]")

            time.sleep(0.8)

        def show_ai_response(text: str, tokens: int, duration: float) -> None:
            """Show AI response with fast streaming effect."""
            print_separator(console)
            console.print(f"[dim]  {duration:.1f}s  │  {tokens:,} tokens[/dim]")
            print_separator(console)
            console.print()

            # Stream the response word by word - FAST
            words = text.split()
            displayed = ""
            with Live(console=console, refresh_per_second=60, transient=False) as live:
                for i, word in enumerate(words):
                    displayed += word + " "
                    live.update(Markdown(displayed))
                    time.sleep(0.015)  # Super fast streaming

        # Scene 1: Show slash commands
        console.print()
        console.print("[dim]Type / to see available commands...[/dim]")
        time.sleep(0.4)
        show_slash_dropdown()

        # Clear and show actual question
        time.sleep(0.3)
        console.print()
        console.print()

        # Scene 2: Ask a question
        show_chat_box("How do I catch regressions before deploying?")
        time.sleep(0.15)
        show_thinking(0.3)

        show_ai_response(
            """Save a **golden baseline** from a working run, then compare future runs against it:

```bash
# 1. Save your current working state
evalview golden save .evalview/results/latest.json

# 2. Make changes to your agent

# 3. Run with --diff to catch regressions
evalview run --diff
```

This catches **tool changes**, **output drift**, **cost spikes**, and **latency issues** before they hit production.""",
            487,
            0.9,
        )

        time.sleep(0.4)

        # Scene 3: Follow-up
        console.print()
        show_chat_box("Run it now")
        time.sleep(0.15)

        # Show command execution
        console.print()
        console.print("[dim]Running:[/dim] evalview run --diff")
        time.sleep(0.15)

        # Quick test results
        quick_results = [
            ("auth-flow", "PASSED", "green"),
            ("search-query", "PASSED", "green"),
            ("checkout", "REGRESSION", "red"),
        ]
        console.print()
        for name, status, color in quick_results:
            time.sleep(0.1)
            icon = "✓" if status == "PASSED" else "✗"
            console.print(f"  [{color}]{icon} {status:<12}[/{color}] {name}")

        console.print()
        console.print("━" * 50)
        console.print("  [red]❌ 1 regression detected - blocked deploy[/red]")
        console.print("━" * 50)

        time.sleep(0.5)
        console.print()
        console.print("[bold #22d3ee]Ask anything. Get answers. Ship with confidence.[/bold #22d3ee]")
        console.print("[dim]pip install evalview && evalview chat[/dim]")
        console.print()
        console.print("[dim]⭐ Star if this helped → github.com/hidai25/eval-view[/dim]\n")
