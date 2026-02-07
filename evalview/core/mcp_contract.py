"""MCP contract storage and management.

MCP contracts are snapshots of an external MCP server's tool definitions.
When running tests with --contracts, current tool definitions are compared
against the snapshot to detect interface drift before tests execute.

This is the mirror of golden traces: golden traces detect when YOUR agent
drifts, contracts detect when an EXTERNAL server drifts underneath you.

Storage format:
  .evalview/contracts/
    <server-name>.contract.json    # The tool schema snapshot
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class ToolSchema(BaseModel):
    """Schema for a single MCP tool."""

    name: str
    description: str = ""
    inputSchema: Dict[str, Any] = Field(default_factory=dict)


class ContractMetadata(BaseModel):
    """Metadata about a contract snapshot."""

    server_name: str
    endpoint: str
    snapshot_at: datetime
    protocol_version: str = "2024-11-05"
    tool_count: int = 0
    notes: Optional[str] = None
    schema_hash: str = ""  # Hash of tool schemas for quick comparison


class MCPContract(BaseModel):
    """A contract snapshot from an MCP server."""

    metadata: ContractMetadata
    tools: List[ToolSchema] = Field(default_factory=list)

    @property
    def tool_names(self) -> List[str]:
        return [t.name for t in self.tools]


class ContractStore:
    """Manages MCP contract storage and retrieval."""

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path(".")
        self.contracts_dir = self.base_path / ".evalview" / "contracts"

    def _safe_name(self, server_name: str) -> str:
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in server_name)

    def _get_contract_path(self, server_name: str) -> Path:
        return self.contracts_dir / f"{self._safe_name(server_name)}.contract.json"

    def _hash_schemas(self, tools: List[Dict[str, Any]]) -> str:
        """Hash tool schemas for quick drift comparison."""
        canonical = json.dumps(tools, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def save_contract(
        self,
        server_name: str,
        endpoint: str,
        tools: List[Dict[str, Any]],
        notes: Optional[str] = None,
    ) -> Path:
        """Save a tool schema snapshot as a contract.

        Args:
            server_name: Human-readable server identifier.
            endpoint: The MCP server endpoint used for discovery.
            tools: Raw tool definitions from tools/list response.
            notes: Optional notes about this snapshot.

        Returns:
            Path to saved contract file.
        """
        self.contracts_dir.mkdir(parents=True, exist_ok=True)

        tool_schemas = [ToolSchema.model_validate(t) for t in tools]

        contract = MCPContract(
            metadata=ContractMetadata(
                server_name=server_name,
                endpoint=endpoint,
                snapshot_at=datetime.now(),
                tool_count=len(tool_schemas),
                notes=notes,
                schema_hash=self._hash_schemas(tools),
            ),
            tools=tool_schemas,
        )

        contract_path = self._get_contract_path(server_name)
        with open(contract_path, "w") as f:
            f.write(contract.model_dump_json(indent=2))

        logger.info(f"Saved contract: {contract_path}")
        return contract_path

    def load_contract(self, server_name: str) -> Optional[MCPContract]:
        """Load a contract by server name."""
        contract_path = self._get_contract_path(server_name)
        if not contract_path.exists():
            return None

        with open(contract_path) as f:
            data = json.load(f)

        return MCPContract.model_validate(data)

    def has_contract(self, server_name: str) -> bool:
        return self._get_contract_path(server_name).exists()

    def list_contracts(self) -> List[ContractMetadata]:
        """List all saved contracts."""
        if not self.contracts_dir.exists():
            return []

        results = []
        for path in self.contracts_dir.glob("*.contract.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                results.append(ContractMetadata.model_validate(data["metadata"]))
            except Exception as e:
                logger.warning(f"Failed to load contract {path}: {e}")

        return results

    def delete_contract(self, server_name: str) -> bool:
        """Delete a contract."""
        contract_path = self._get_contract_path(server_name)
        if contract_path.exists():
            contract_path.unlink()
            return True
        return False
