# Recipe: Extend the HTML Report

## Goal

Add a new field, panel, badge, or summary to the visual report generated after `check` and related commands.

## Read These Files First

- `evalview/visualization/generators.py`
- `evalview/commands/check_cmd.py`
- `tests/test_visualization_generators.py`

## Requirements

- the report should consume typed data, not scrape terminal output
- report changes must not alter evaluation semantics
- if the data is part of `check`, pass it through explicitly from the command layer

## Steps

1. Add or extend data passed into `generate_visual_report(...)`.
2. Update `evalview/visualization/generators.py` to compute or render the new field.
3. Update the Jinja template inside `generators.py`.
4. Add or update tests in `tests/test_visualization_generators.py`.

## Done Criteria

- generated HTML contains the new data
- the report still renders without the new field when it is absent
- tests assert concrete strings or structure in the HTML

## Common Pitfalls

- adding user-visible state only to terminal output
- changing the report template without passing the required data
- assuming every report path comes from `check`; other commands also use the generator
