"""Contract diff engine for detecting MCP server interface drift.

Compares a saved contract snapshot against the current tool definitions
from an MCP server. Detects:
  - REMOVED tools (breaking)
  - ADDED tools (informational)
  - CHANGED schemas: new required params, renamed params, type changes (breaking)
  - Description-only changes (informational, ignored by default)

This is the mirror of diff.py: diff.py compares agent behavior traces,
contract_diff.py compares server interface schemas.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Set
import logging

from evalview.core.mcp_contract import MCPContract, ToolSchema

logger = logging.getLogger(__name__)


class ContractDriftStatus(Enum):
    """Result of comparing current tools against a contract snapshot.

    Two states:
    - PASSED: Interface matches snapshot (no breaking changes).
    - CONTRACT_DRIFT: Breaking changes detected (tools removed, schemas changed).
    """

    PASSED = "passed"
    CONTRACT_DRIFT = "contract_drift"


class ChangeKind(Enum):
    """Kind of change detected in a tool schema."""

    REMOVED = "removed"           # Tool no longer exists (breaking)
    ADDED = "added"               # New tool available (informational)
    PARAM_ADDED_REQ = "param_added_required"  # New required param (breaking)
    PARAM_REMOVED = "param_removed"           # Param removed (breaking)
    PARAM_TYPE_CHANGED = "param_type_changed"  # Param type changed (breaking)
    PARAM_ADDED_OPT = "param_added_optional"  # New optional param (safe)
    DESCRIPTION_CHANGED = "description_changed"  # Description changed (info)


# Changes that constitute a breaking contract drift
BREAKING_CHANGES: Set[ChangeKind] = {
    ChangeKind.REMOVED,
    ChangeKind.PARAM_ADDED_REQ,
    ChangeKind.PARAM_REMOVED,
    ChangeKind.PARAM_TYPE_CHANGED,
}


@dataclass
class ToolChange:
    """A single change detected in a tool's schema."""

    tool_name: str
    kind: ChangeKind
    detail: str

    @property
    def is_breaking(self) -> bool:
        return self.kind in BREAKING_CHANGES


@dataclass
class ContractDiff:
    """Complete diff between a contract snapshot and current server tools."""

    server_name: str
    changes: List[ToolChange] = field(default_factory=list)
    snapshot_tool_count: int = 0
    current_tool_count: int = 0

    @property
    def has_breaking_changes(self) -> bool:
        return any(c.is_breaking for c in self.changes)

    @property
    def status(self) -> ContractDriftStatus:
        if self.has_breaking_changes:
            return ContractDriftStatus.CONTRACT_DRIFT
        return ContractDriftStatus.PASSED

    @property
    def breaking_changes(self) -> List[ToolChange]:
        return [c for c in self.changes if c.is_breaking]

    @property
    def informational_changes(self) -> List[ToolChange]:
        return [c for c in self.changes if not c.is_breaking]

    def summary(self) -> str:
        if not self.changes:
            return "No changes"

        breaking = len(self.breaking_changes)
        info = len(self.informational_changes)
        parts = []
        if breaking:
            parts.append(f"{breaking} breaking change(s)")
        if info:
            parts.append(f"{info} informational change(s)")
        return ", ".join(parts)


def _get_required_params(schema: Dict[str, Any]) -> Set[str]:
    """Extract required parameter names from a JSON Schema."""
    return set(schema.get("required", []))


def _get_all_params(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract all parameter definitions from a JSON Schema."""
    return schema.get("properties", {})


def _diff_tool_schema(
    tool_name: str,
    snapshot: ToolSchema,
    current: ToolSchema,
) -> List[ToolChange]:
    """Compare two versions of the same tool's schema."""
    changes: List[ToolChange] = []

    # Description change (informational)
    if snapshot.description != current.description:
        changes.append(ToolChange(
            tool_name=tool_name,
            kind=ChangeKind.DESCRIPTION_CHANGED,
            detail=(
                f"description changed from "
                f"'{snapshot.description[:60]}' to '{current.description[:60]}'"
            ),
        ))

    snap_schema = snapshot.inputSchema
    curr_schema = current.inputSchema

    snap_params = _get_all_params(snap_schema)
    curr_params = _get_all_params(curr_schema)
    snap_required = _get_required_params(snap_schema)
    curr_required = _get_required_params(curr_schema)

    snap_names = set(snap_params.keys())
    curr_names = set(curr_params.keys())

    # Removed parameters (breaking)
    for name in snap_names - curr_names:
        changes.append(ToolChange(
            tool_name=tool_name,
            kind=ChangeKind.PARAM_REMOVED,
            detail=f"parameter '{name}' removed",
        ))

    # Added parameters
    for name in curr_names - snap_names:
        if name in curr_required:
            changes.append(ToolChange(
                tool_name=tool_name,
                kind=ChangeKind.PARAM_ADDED_REQ,
                detail=f"new required parameter '{name}'",
            ))
        else:
            changes.append(ToolChange(
                tool_name=tool_name,
                kind=ChangeKind.PARAM_ADDED_OPT,
                detail=f"new optional parameter '{name}'",
            ))

    # Changed parameters (type changes)
    for name in snap_names & curr_names:
        snap_type = snap_params[name].get("type")
        curr_type = curr_params[name].get("type")
        if snap_type != curr_type:
            changes.append(ToolChange(
                tool_name=tool_name,
                kind=ChangeKind.PARAM_TYPE_CHANGED,
                detail=f"parameter '{name}' type changed from '{snap_type}' to '{curr_type}'",
            ))

        # Parameter became required (breaking - it's a new constraint)
        if name not in snap_required and name in curr_required:
            changes.append(ToolChange(
                tool_name=tool_name,
                kind=ChangeKind.PARAM_ADDED_REQ,
                detail=f"parameter '{name}' became required",
            ))

    return changes


def diff_contract(
    contract: MCPContract,
    current_tools: List[Dict[str, Any]],
) -> ContractDiff:
    """Compare a saved contract against current tool definitions.

    Args:
        contract: The saved contract snapshot.
        current_tools: Current tool definitions from the MCP server.

    Returns:
        ContractDiff with all detected changes.
    """
    current_schemas = {
        t["name"]: ToolSchema.model_validate(t) for t in current_tools
    }
    snapshot_schemas = {t.name: t for t in contract.tools}

    changes: List[ToolChange] = []

    # Removed tools (in snapshot but not in current)
    for name in snapshot_schemas:
        if name not in current_schemas:
            changes.append(ToolChange(
                tool_name=name,
                kind=ChangeKind.REMOVED,
                detail=f"tool '{name}' no longer available",
            ))

    # Added tools (in current but not in snapshot)
    for name in current_schemas:
        if name not in snapshot_schemas:
            changes.append(ToolChange(
                tool_name=name,
                kind=ChangeKind.ADDED,
                detail=f"new tool '{name}' available",
            ))

    # Changed tools (in both)
    for name in snapshot_schemas.keys() & current_schemas.keys():
        tool_changes = _diff_tool_schema(
            name, snapshot_schemas[name], current_schemas[name]
        )
        changes.extend(tool_changes)

    return ContractDiff(
        server_name=contract.metadata.server_name,
        changes=changes,
        snapshot_tool_count=len(snapshot_schemas),
        current_tool_count=len(current_schemas),
    )
