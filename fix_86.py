# evalview/__init__.py

import click
from evalview.check_cmd import _execute_check_tests
from rich.console import Console
from rich.table import Table

@click.group()
def evalview():
    pass

@evalview.command()
@click.option('--a', required=True, help='Endpoint for Agent A')
@click.option('--b', required=True, help='Endpoint for Agent B')
@click.option('--test', default=None, help='Specific test to run')
@click.option('--json', is_flag=True, help='Output results in JSON format')
def compare(a, b, test, json):
    results_a = _execute_check_tests(a, test)
    results_b = _execute_check_tests(b, test)

    if json:
        import json
        combined_results = {
            "agent_a": results_a,
            "agent_b": results_b
        }
        click.echo(json.dumps(combined_results, indent=4))
        return

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test")
    table.add_column(f"Agent A ({a})")
    table.add_column(f"Agent B ({b})")

    for test_name in results_a:
        score_a = results_a[test_name]['score']
        score_b = results_b[test_name]['score']
        row = [
            test_name,
            f"✅ {score_a}/100",
            f"✅ {score_b}/100"
        ]
        table.add_row(*row)

    console.print(table)

    # Summary
    regressions = sum(1 for test_name in results_a if results_a[test_name]['score'] > results_b[test_name]['score'])
    console.print(f"Summary: Agent B regressed on {regressions}/{len(results_a)} tests")

if __name__ == "__main__":
    evalview()