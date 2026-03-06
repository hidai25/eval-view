"""MCP contract management commands."""
from __future__ import annotations

from datetime import datetime

import click

from evalview.commands.shared import console


@click.group()
def mcp():
    """Manage MCP contracts (detect external server interface drift).

    MCP contracts are snapshots of an external MCP server's tool definitions.
    Use them with `evalview run --contracts` to detect when servers change
    their interface before your tests break.

    Examples:
        evalview mcp snapshot "npx:@modelcontextprotocol/server-github" --name server-github
        evalview mcp check server-github
        evalview mcp list
    """
    pass


@mcp.command("snapshot")
@click.argument("endpoint")
@click.option("--name", "-n", required=True, help="Server name (used as contract identifier)")
@click.option("--notes", help="Notes about this snapshot")
@click.option("--timeout", default=30.0, type=float, help="Connection timeout in seconds")
def mcp_snapshot(endpoint: str, name: str, notes: str, timeout: float):
    """Snapshot an MCP server's tool definitions as a contract.

    ENDPOINT is the MCP server endpoint (e.g., "npx:@modelcontextprotocol/server-github").

    Examples:
        evalview mcp snapshot "npx:@modelcontextprotocol/server-filesystem /tmp" --name fs-server
        evalview mcp snapshot "http://localhost:8080" --name my-server --notes "v2.1 release"
    """
    import asyncio
    from evalview.adapters.mcp_adapter import MCPAdapter
    from evalview.core.mcp_contract import ContractStore

    console.print("\n[cyan]━━━ MCP Contract Snapshot ━━━[/cyan]\n")
    console.print(f"  Server: [bold]{name}[/bold]")
    console.print(f"  Endpoint: {endpoint}")
    console.print()

    adapter = MCPAdapter(endpoint=endpoint, timeout=timeout)

    try:
        tools = asyncio.run(adapter.discover_tools())
    except Exception as e:
        console.print(f"[red]Failed to connect to MCP server: {e}[/red]")
        console.print("[dim]Check that the server is running and the endpoint is correct.[/dim]\n")
        raise SystemExit(1)

    if not tools:
        console.print("[yellow]Server returned no tools.[/yellow]\n")
        raise SystemExit(1)

    store = ContractStore()

    if store.has_contract(name):
        if not click.confirm(
            f"Contract '{name}' already exists. Overwrite?",
            default=False,
        ):
            console.print("[dim]Cancelled[/dim]\n")
            return

    path = store.save_contract(
        server_name=name,
        endpoint=endpoint,
        tools=tools,
        notes=notes,
    )

    console.print(f"[green]Snapshot saved: {path}[/green]")
    console.print(f"  Tools discovered: [bold]{len(tools)}[/bold]")
    for tool in tools:
        desc = tool.get("description", "")
        if len(desc) > 60:
            desc = desc[:57] + "..."
        console.print(f"    [dim]- {tool['name']}[/dim]  {desc}")
    console.print()
    console.print("[dim]Check for drift: evalview mcp check " + name + "[/dim]")
    console.print("[dim]Use in CI: evalview run --contracts --fail-on CONTRACT_DRIFT[/dim]\n")


@mcp.command("check")
@click.argument("name")
@click.option("--endpoint", help="Override endpoint (default: use endpoint from snapshot)")
@click.option("--timeout", default=30.0, type=float, help="Connection timeout in seconds")
def mcp_check(name: str, endpoint: str, timeout: float):
    """Check an MCP server for contract drift.

    NAME is the contract name (from `evalview mcp snapshot --name`).

    Examples:
        evalview mcp check server-github
        evalview mcp check my-server --endpoint "http://new-host:8080"
    """
    import asyncio
    from evalview.adapters.mcp_adapter import MCPAdapter
    from evalview.core.mcp_contract import ContractStore
    from evalview.core.contract_diff import diff_contract, ContractDriftStatus

    store = ContractStore()
    contract = store.load_contract(name)

    if not contract:
        console.print(f"\n[red]No contract found: {name}[/red]")
        console.print("[dim]Create one with: evalview mcp snapshot <endpoint> --name " + name + "[/dim]\n")
        raise SystemExit(1)

    target_endpoint = endpoint or contract.metadata.endpoint
    adapter = MCPAdapter(endpoint=target_endpoint, timeout=timeout)

    console.print("\n[cyan]━━━ MCP Contract Check ━━━[/cyan]\n")
    console.print(f"  Contract: [bold]{name}[/bold]")
    console.print(f"  Endpoint: {target_endpoint}")

    # Show snapshot age
    age = datetime.now() - contract.metadata.snapshot_at
    age_days = age.days
    if age_days > 30:
        console.print(f"  Snapshot age: [yellow]{age_days} days (consider refreshing)[/yellow]")
    else:
        console.print(f"  Snapshot age: [dim]{age_days} day(s)[/dim]")
    console.print()

    try:
        current_tools = asyncio.run(adapter.discover_tools())
    except Exception as e:
        console.print(f"[red]Failed to connect to MCP server: {e}[/red]")
        console.print("[dim]The server may be down. Use --endpoint to try a different host.[/dim]\n")
        raise SystemExit(2)

    result = diff_contract(contract, current_tools)

    if result.status == ContractDriftStatus.PASSED:
        if result.informational_changes:
            console.print(f"[green]PASSED[/green] - No breaking changes ({result.summary()})")
            console.print()
            for change in result.informational_changes:
                console.print(f"  [dim]+ {change.tool_name}: {change.detail}[/dim]")
        else:
            console.print("[green]PASSED[/green] - Interface matches snapshot exactly")
        console.print()
    else:
        console.print(f"[red]CONTRACT_DRIFT[/red] - {result.summary()}\n")

        for change in result.breaking_changes:
            if change.kind.value == "removed":
                console.print(f"  [red]REMOVED: {change.tool_name}[/red] - {change.detail}")
            else:
                console.print(f"  [red]CHANGED: {change.tool_name}[/red] - {change.detail}")

        if result.informational_changes:
            console.print()
            for change in result.informational_changes:
                console.print(f"  [dim]INFO: {change.tool_name} - {change.detail}[/dim]")

        console.print()
        console.print("[dim]To accept the new interface: evalview mcp snapshot " + target_endpoint + " --name " + name + "[/dim]\n")
        raise SystemExit(1)


@mcp.command("list")
def mcp_list():
    """List all MCP contract snapshots.

    Shows all saved contracts with metadata.
    """
    from evalview.core.mcp_contract import ContractStore

    store = ContractStore()
    contracts = store.list_contracts()

    if not contracts:
        console.print("\n[yellow]No MCP contracts found.[/yellow]")
        console.print("[dim]Create one: evalview mcp snapshot <endpoint> --name <name>[/dim]\n")
        return

    console.print("\n[cyan]━━━ MCP Contracts ━━━[/cyan]\n")

    for c in sorted(contracts, key=lambda x: x.server_name):
        age = datetime.now() - c.snapshot_at
        age_str = f"{age.days}d ago" if age.days > 0 else "today"

        console.print(f"  [bold]{c.server_name}[/bold]")
        console.print(f"    [dim]Endpoint: {c.endpoint}[/dim]")
        console.print(f"    [dim]Tools: {c.tool_count} | Snapshot: {age_str}[/dim]")
        if c.notes:
            console.print(f"    [dim]Notes: {c.notes}[/dim]")
        console.print()

    console.print(f"[dim]Total: {len(contracts)} contract(s)[/dim]")
    console.print("[dim]Check for drift: evalview mcp check <name>[/dim]\n")


@mcp.command("delete")
@click.argument("name")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
def mcp_delete(name: str, force: bool):
    """Delete an MCP contract snapshot.

    NAME is the contract name to delete.
    """
    from evalview.core.mcp_contract import ContractStore

    store = ContractStore()

    if not store.has_contract(name):
        console.print(f"\n[yellow]No contract found: {name}[/yellow]\n")
        return

    if not force:
        if not click.confirm(f"Delete contract '{name}'?", default=False):
            console.print("[dim]Cancelled[/dim]")
            return

    store.delete_contract(name)
    console.print(f"\n[green]Deleted contract: {name}[/green]\n")


@mcp.command("show")
@click.argument("name")
def mcp_show(name: str):
    """Show details of an MCP contract snapshot.

    NAME is the contract name.
    """
    from evalview.core.mcp_contract import ContractStore

    store = ContractStore()
    contract = store.load_contract(name)

    if not contract:
        console.print(f"\n[yellow]No contract found: {name}[/yellow]")
        console.print("[dim]Create one: evalview mcp snapshot <endpoint> --name " + name + "[/dim]\n")
        return

    meta = contract.metadata
    age = datetime.now() - meta.snapshot_at

    console.print(f"\n[cyan]━━━ MCP Contract: {meta.server_name} ━━━[/cyan]\n")
    console.print(f"  Endpoint: {meta.endpoint}")
    console.print(f"  Snapshot: {meta.snapshot_at.strftime('%Y-%m-%d %H:%M')} ({age.days}d ago)")
    console.print(f"  Protocol: {meta.protocol_version}")
    console.print(f"  Schema hash: {meta.schema_hash}")
    if meta.notes:
        console.print(f"  Notes: {meta.notes}")
    console.print()

    console.print(f"[bold]Tools ({meta.tool_count}):[/bold]\n")

    for tool in contract.tools:
        console.print(f"  [bold]{tool.name}[/bold]")
        if tool.description:
            console.print(f"    {tool.description}")
        if tool.inputSchema.get("properties"):
            props = tool.inputSchema["properties"]
            required = set(tool.inputSchema.get("required", []))
            for pname, pdef in props.items():
                ptype = pdef.get("type", "any")
                req_marker = " [red]*[/red]" if pname in required else ""
                console.print(f"    [dim]- {pname}: {ptype}{req_marker}[/dim]")
        console.print()


@mcp.command("serve")
@click.option("--test-path", default="tests", help="Path to test directory")
def mcp_serve(test_path: str) -> None:
    """Start EvalView as an MCP server for Claude Code.

    Exposes run_check, run_snapshot, and list_tests as MCP tools so you can
    run regression checks inline without switching to a terminal.

    \b
    One-time setup:
        claude mcp add --transport stdio evalview -- evalview mcp serve

    \b
    Verify:
        claude mcp list

    \b
    Then ask Claude: "Did my refactor break the golden baseline?"
    """
    from evalview.mcp_server import MCPServer

    MCPServer(test_path=test_path).serve()
