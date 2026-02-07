"""Tests for MCP contract testing: storage, diff engine, and integration."""

import json
import pytest
from datetime import datetime

from evalview.core.mcp_contract import (
    ContractStore,
    MCPContract,
    ContractMetadata,
    ToolSchema,
)
from evalview.core.contract_diff import (
    diff_contract,
    ContractDiff,
    ContractDriftStatus,
    ChangeKind,
    ToolChange,
    BREAKING_CHANGES,
    _diff_tool_schema,
)


# ============================================================================
# Test Data
# ============================================================================

SAMPLE_TOOLS = [
    {
        "name": "create_issue",
        "description": "Create a new issue in a GitHub repository",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["repo", "title"],
        },
    },
    {
        "name": "list_issues",
        "description": "List issues in a repository",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "state": {"type": "string"},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the filesystem",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
]


@pytest.fixture
def contract_store(tmp_path):
    """Create a ContractStore using a temp directory."""
    return ContractStore(base_path=tmp_path)


@pytest.fixture
def saved_contract(contract_store):
    """Create and save a sample contract."""
    contract_store.save_contract(
        server_name="test-server",
        endpoint="npx:@test/server",
        tools=SAMPLE_TOOLS,
        notes="Test snapshot",
    )
    return contract_store.load_contract("test-server")


# ============================================================================
# ContractStore Tests
# ============================================================================


class TestContractStore:
    """Tests for MCP contract storage."""

    def test_save_and_load(self, contract_store):
        """Save a contract and load it back."""
        path = contract_store.save_contract(
            server_name="my-server",
            endpoint="http://localhost:8080",
            tools=SAMPLE_TOOLS,
            notes="Initial snapshot",
        )

        assert path.exists()
        assert path.suffix == ".json"

        loaded = contract_store.load_contract("my-server")
        assert loaded is not None
        assert loaded.metadata.server_name == "my-server"
        assert loaded.metadata.endpoint == "http://localhost:8080"
        assert loaded.metadata.tool_count == 3
        assert loaded.metadata.notes == "Initial snapshot"
        assert len(loaded.tools) == 3

    def test_tool_names(self, saved_contract):
        """Contract exposes tool names."""
        assert saved_contract.tool_names == ["create_issue", "list_issues", "read_file"]

    def test_load_nonexistent(self, contract_store):
        """Loading a nonexistent contract returns None."""
        assert contract_store.load_contract("does-not-exist") is None

    def test_has_contract(self, contract_store):
        """has_contract returns correct boolean."""
        assert not contract_store.has_contract("my-server")

        contract_store.save_contract(
            server_name="my-server",
            endpoint="http://localhost:8080",
            tools=SAMPLE_TOOLS,
        )

        assert contract_store.has_contract("my-server")

    def test_list_contracts(self, contract_store):
        """List all saved contracts."""
        assert contract_store.list_contracts() == []

        contract_store.save_contract("server-a", "http://a", SAMPLE_TOOLS[:1])
        contract_store.save_contract("server-b", "http://b", SAMPLE_TOOLS[:2])

        contracts = contract_store.list_contracts()
        assert len(contracts) == 2
        names = {c.server_name for c in contracts}
        assert names == {"server-a", "server-b"}

    def test_delete_contract(self, contract_store):
        """Delete a contract."""
        contract_store.save_contract("my-server", "http://localhost", SAMPLE_TOOLS)
        assert contract_store.has_contract("my-server")

        result = contract_store.delete_contract("my-server")
        assert result is True
        assert not contract_store.has_contract("my-server")

    def test_delete_nonexistent(self, contract_store):
        """Deleting a nonexistent contract returns False."""
        assert contract_store.delete_contract("nope") is False

    def test_overwrite(self, contract_store):
        """Overwriting a contract replaces it."""
        contract_store.save_contract("my-server", "http://old", SAMPLE_TOOLS[:1])
        contract_store.save_contract("my-server", "http://new", SAMPLE_TOOLS[:2])

        loaded = contract_store.load_contract("my-server")
        assert loaded.metadata.endpoint == "http://new"
        assert loaded.metadata.tool_count == 2

    def test_schema_hash_changes_on_different_tools(self, contract_store):
        """Different tool sets produce different schema hashes."""
        contract_store.save_contract("server-a", "http://a", SAMPLE_TOOLS[:1])
        contract_store.save_contract("server-b", "http://b", SAMPLE_TOOLS[:2])

        a = contract_store.load_contract("server-a")
        b = contract_store.load_contract("server-b")
        assert a.metadata.schema_hash != b.metadata.schema_hash

    def test_safe_name_sanitization(self, contract_store):
        """Server names with special chars are sanitized for filesystem."""
        contract_store.save_contract(
            "my server/with:special chars!",
            "http://x",
            SAMPLE_TOOLS[:1],
        )
        assert contract_store.has_contract("my server/with:special chars!")

    def test_metadata_timestamp(self, contract_store):
        """Saved contract has a valid timestamp."""
        contract_store.save_contract("ts-test", "http://x", SAMPLE_TOOLS[:1])
        loaded = contract_store.load_contract("ts-test")
        assert isinstance(loaded.metadata.snapshot_at, datetime)


# ============================================================================
# Contract Diff Engine Tests
# ============================================================================


class TestContractDiff:
    """Tests for the schema diff engine."""

    def test_no_changes(self, saved_contract):
        """Identical tools produce no changes."""
        result = diff_contract(saved_contract, SAMPLE_TOOLS)

        assert result.status == ContractDriftStatus.PASSED
        assert result.changes == []
        assert not result.has_breaking_changes
        assert result.summary() == "No changes"

    def test_tool_removed(self, saved_contract):
        """Removing a tool is a breaking change."""
        current = [t for t in SAMPLE_TOOLS if t["name"] != "read_file"]
        result = diff_contract(saved_contract, current)

        assert result.status == ContractDriftStatus.CONTRACT_DRIFT
        assert result.has_breaking_changes

        removed = [c for c in result.changes if c.kind == ChangeKind.REMOVED]
        assert len(removed) == 1
        assert removed[0].tool_name == "read_file"
        assert removed[0].is_breaking

    def test_tool_added(self, saved_contract):
        """Adding a tool is informational (not breaking)."""
        current = SAMPLE_TOOLS + [{
            "name": "delete_file",
            "description": "Delete a file",
            "inputSchema": {"type": "object", "properties": {}},
        }]
        result = diff_contract(saved_contract, current)

        assert result.status == ContractDriftStatus.PASSED  # Not breaking
        assert len(result.changes) == 1
        assert result.changes[0].kind == ChangeKind.ADDED
        assert result.changes[0].tool_name == "delete_file"
        assert not result.changes[0].is_breaking

    def test_required_param_added(self, saved_contract):
        """Adding a new required parameter is breaking."""
        current = json.loads(json.dumps(SAMPLE_TOOLS))
        current[0]["inputSchema"]["properties"]["owner"] = {"type": "string"}
        current[0]["inputSchema"]["required"].append("owner")

        result = diff_contract(saved_contract, current)

        assert result.status == ContractDriftStatus.CONTRACT_DRIFT
        breaking = result.breaking_changes
        assert any(
            c.kind == ChangeKind.PARAM_ADDED_REQ and "owner" in c.detail
            for c in breaking
        )

    def test_optional_param_added(self, saved_contract):
        """Adding a new optional parameter is not breaking."""
        current = json.loads(json.dumps(SAMPLE_TOOLS))
        current[0]["inputSchema"]["properties"]["labels"] = {"type": "array"}
        # Not added to "required"

        result = diff_contract(saved_contract, current)

        assert result.status == ContractDriftStatus.PASSED
        info = result.informational_changes
        assert any(c.kind == ChangeKind.PARAM_ADDED_OPT for c in info)

    def test_param_removed(self, saved_contract):
        """Removing a parameter is breaking."""
        current = json.loads(json.dumps(SAMPLE_TOOLS))
        del current[0]["inputSchema"]["properties"]["body"]

        result = diff_contract(saved_contract, current)

        assert result.status == ContractDriftStatus.CONTRACT_DRIFT
        assert any(
            c.kind == ChangeKind.PARAM_REMOVED and "body" in c.detail
            for c in result.breaking_changes
        )

    def test_param_type_changed(self, saved_contract):
        """Changing a parameter type is breaking."""
        current = json.loads(json.dumps(SAMPLE_TOOLS))
        current[1]["inputSchema"]["properties"]["repo"]["type"] = "integer"

        result = diff_contract(saved_contract, current)

        assert result.status == ContractDriftStatus.CONTRACT_DRIFT
        assert any(
            c.kind == ChangeKind.PARAM_TYPE_CHANGED
            for c in result.breaking_changes
        )

    def test_description_changed(self, saved_contract):
        """Description change is informational."""
        current = json.loads(json.dumps(SAMPLE_TOOLS))
        current[0]["description"] = "Updated description for create_issue"

        result = diff_contract(saved_contract, current)

        assert result.status == ContractDriftStatus.PASSED
        assert any(c.kind == ChangeKind.DESCRIPTION_CHANGED for c in result.changes)

    def test_param_became_required(self, saved_contract):
        """A parameter becoming required is breaking."""
        current = json.loads(json.dumps(SAMPLE_TOOLS))
        # "body" was optional, make it required
        current[0]["inputSchema"]["required"].append("body")

        result = diff_contract(saved_contract, current)

        assert result.status == ContractDriftStatus.CONTRACT_DRIFT
        assert any(
            c.kind == ChangeKind.PARAM_ADDED_REQ and "body" in c.detail
            and "became required" in c.detail
            for c in result.breaking_changes
        )

    def test_multiple_changes(self, saved_contract):
        """Multiple changes across tools are all detected."""
        current = json.loads(json.dumps(SAMPLE_TOOLS))
        # Remove read_file
        current = [t for t in current if t["name"] != "read_file"]
        # Add required param to list_issues
        current[1]["inputSchema"]["properties"]["owner"] = {"type": "string"}
        current[1]["inputSchema"]["required"].append("owner")
        # Add new tool
        current.append({
            "name": "merge_pr",
            "description": "Merge a pull request",
            "inputSchema": {"type": "object", "properties": {}},
        })

        result = diff_contract(saved_contract, current)

        assert result.status == ContractDriftStatus.CONTRACT_DRIFT
        assert result.snapshot_tool_count == 3
        assert result.current_tool_count == 3  # removed 1, added 1

        kinds = {c.kind for c in result.changes}
        assert ChangeKind.REMOVED in kinds
        assert ChangeKind.ADDED in kinds
        assert ChangeKind.PARAM_ADDED_REQ in kinds

    def test_summary_no_changes(self, saved_contract):
        """Summary for no changes."""
        result = diff_contract(saved_contract, SAMPLE_TOOLS)
        assert result.summary() == "No changes"

    def test_summary_with_breaking_only(self, saved_contract):
        """Summary with breaking changes only."""
        current = [t for t in SAMPLE_TOOLS if t["name"] != "read_file"]
        result = diff_contract(saved_contract, current)
        assert result.summary() == "1 breaking change(s)"

    def test_summary_with_mixed_changes(self, saved_contract):
        """Summary with both breaking and informational changes."""
        current = json.loads(json.dumps(SAMPLE_TOOLS))
        # Remove a tool (breaking) and add one (informational)
        current = [t for t in current if t["name"] != "read_file"]
        current.append({
            "name": "new_tool",
            "description": "A new tool",
            "inputSchema": {"type": "object", "properties": {}},
        })
        result = diff_contract(saved_contract, current)
        summary = result.summary()
        assert "1 breaking change(s)" in summary
        assert "1 informational change(s)" in summary

    def test_duplicate_tool_names_in_current(self, saved_contract):
        """Duplicate tool names in current tools are deduplicated silently."""
        current = SAMPLE_TOOLS + [SAMPLE_TOOLS[0]]  # Duplicate create_issue
        result = diff_contract(saved_contract, current)
        # Dict comprehension deduplicates - should still pass
        assert result.status == ContractDriftStatus.PASSED

    def test_empty_snapshot_vs_tools(self):
        """Empty snapshot vs populated tools shows all as added."""
        contract = MCPContract(
            metadata=ContractMetadata(
                server_name="empty",
                endpoint="http://x",
                snapshot_at=datetime.now(),
            ),
            tools=[],
        )
        result = diff_contract(contract, SAMPLE_TOOLS)

        assert result.status == ContractDriftStatus.PASSED  # All added = not breaking
        assert all(c.kind == ChangeKind.ADDED for c in result.changes)
        assert len(result.changes) == 3

    def test_all_tools_removed(self, saved_contract):
        """All tools removed is breaking."""
        result = diff_contract(saved_contract, [])

        assert result.status == ContractDriftStatus.CONTRACT_DRIFT
        assert all(c.kind == ChangeKind.REMOVED for c in result.changes)
        assert len(result.changes) == 3


# ============================================================================
# Breaking Changes Classification Tests
# ============================================================================


class TestBreakingChanges:
    """Test that the right change kinds are classified as breaking."""

    def test_breaking_set(self):
        """Verify the breaking changes set."""
        assert ChangeKind.REMOVED in BREAKING_CHANGES
        assert ChangeKind.PARAM_ADDED_REQ in BREAKING_CHANGES
        assert ChangeKind.PARAM_REMOVED in BREAKING_CHANGES
        assert ChangeKind.PARAM_TYPE_CHANGED in BREAKING_CHANGES

    def test_non_breaking_set(self):
        """Verify non-breaking changes."""
        assert ChangeKind.ADDED not in BREAKING_CHANGES
        assert ChangeKind.PARAM_ADDED_OPT not in BREAKING_CHANGES
        assert ChangeKind.DESCRIPTION_CHANGED not in BREAKING_CHANGES

    def test_tool_change_is_breaking(self):
        """ToolChange.is_breaking reflects its kind."""
        breaking = ToolChange("t", ChangeKind.REMOVED, "gone")
        assert breaking.is_breaking

        safe = ToolChange("t", ChangeKind.ADDED, "new")
        assert not safe.is_breaking


# ============================================================================
# DiffStatus Integration Tests
# ============================================================================


class TestDiffStatusIntegration:
    """Test that CONTRACT_DRIFT integrates with the existing DiffStatus."""

    def test_contract_drift_in_diff_status(self):
        """CONTRACT_DRIFT is available in the main DiffStatus enum."""
        from evalview.core.diff import DiffStatus

        assert hasattr(DiffStatus, "CONTRACT_DRIFT")
        assert DiffStatus.CONTRACT_DRIFT.value == "contract_drift"

    def test_contract_drift_status_enum(self):
        """ContractDriftStatus has the expected values."""
        assert ContractDriftStatus.PASSED.value == "passed"
        assert ContractDriftStatus.CONTRACT_DRIFT.value == "contract_drift"


# ============================================================================
# Tool Schema Diff Tests (unit level)
# ============================================================================


class TestToolSchemaDiff:
    """Unit tests for _diff_tool_schema."""

    def test_identical_schemas(self):
        """No changes for identical schemas."""
        schema = ToolSchema(
            name="test",
            description="A test tool",
            inputSchema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        )
        changes = _diff_tool_schema("test", schema, schema)
        assert changes == []

    def test_empty_schemas(self):
        """No changes for two empty schemas."""
        schema = ToolSchema(name="test", description="", inputSchema={})
        changes = _diff_tool_schema("test", schema, schema)
        assert changes == []

    def test_schema_with_no_properties(self):
        """Handle schemas without properties gracefully."""
        snap = ToolSchema(
            name="test",
            description="old",
            inputSchema={"type": "object"},
        )
        curr = ToolSchema(
            name="test",
            description="new",
            inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        changes = _diff_tool_schema("test", snap, curr)
        # Should detect description change and new optional param
        kinds = {c.kind for c in changes}
        assert ChangeKind.DESCRIPTION_CHANGED in kinds
        assert ChangeKind.PARAM_ADDED_OPT in kinds
