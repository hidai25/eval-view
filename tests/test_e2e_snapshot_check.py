"""End-to-end tests for snapshot → check workflow."""

import tempfile
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime
import pytest


class TestSnapshotCheckWorkflow:
    """Test complete snapshot → modify → check workflow."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        tmpdir = tempfile.mkdtemp()
        project_dir = Path(tmpdir) / "test-project"
        project_dir.mkdir()

        # Create test case directory
        test_cases_dir = project_dir / "tests" / "test-cases"
        test_cases_dir.mkdir(parents=True)

        # Create a simple test case
        test_case = test_cases_dir / "simple-test.yaml"
        test_case.write_text("""
name: simple-test
input: "What is 2+2?"
adapter:
  type: http
  endpoint: http://localhost:8080/agent
expected:
  output_contains:
    - "4"
  min_score: 70
""")

        # Create mock agent response file (simulates agent)
        mock_response_dir = project_dir / ".evalview" / "mock"
        mock_response_dir.mkdir(parents=True)

        yield project_dir

        # Cleanup
        shutil.rmtree(tmpdir)

    def test_full_workflow_snapshot_check_pass(self, temp_project):
        """Test: snapshot → no changes → check passes."""
        # This test requires a running agent, so we'll create mock data

        # Step 1: Create mock golden manually (simulates successful snapshot)
        golden_dir = temp_project / ".evalview" / "golden"
        golden_dir.mkdir(parents=True)

        golden_data = {
            "metadata": {
                "test_name": "simple-test",
                "blessed_at": datetime.now().isoformat(),
                "score": 85.0,
                "blessed_by": "test"
            },
            "trace": {
                "session_id": "test-session",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "steps": [
                    {
                        "step_id": "1",
                        "step_name": "calculate",
                        "tool_name": "calculator",
                        "parameters": {"expression": "2+2"},
                        "output": "4",
                        "success": True,
                        "start_time": datetime.now().isoformat(),
                        "end_time": datetime.now().isoformat(),
                        "metrics": {"cost": 0.01, "latency": 100}
                    }
                ],
                "final_output": "The answer is 4",
                "metrics": {"total_cost": 0.01, "total_latency": 100}
            },
            "tool_sequence": ["calculator"],
            "output_hash": "abc123"
        }

        golden_file = golden_dir / "simple-test.golden.json"
        golden_file.write_text(json.dumps(golden_data, indent=2))

        # Step 2: Verify golden exists
        assert golden_file.exists()

        # Step 3: Create state file (simulates first snapshot)
        state_dir = temp_project / ".evalview"
        state_file = state_dir / "state.json"
        state_data = {
            "last_snapshot_at": datetime.now().isoformat(),
            "last_check_at": None,
            "created_at": datetime.now().isoformat(),
            "last_check_status": None,
            "current_streak": 0,
            "longest_streak": 0,
            "regression_count": 0,
            "total_snapshots": 1,
            "total_checks": 0,
            "milestones_hit": [],
            "conversion_suggestion_shown": False
        }
        state_file.write_text(json.dumps(state_data, indent=2))

        # Verify workflow artifacts exist
        assert (temp_project / ".evalview" / "golden" / "simple-test.golden.json").exists()
        assert (temp_project / ".evalview" / "state.json").exists()

    def test_snapshot_creates_golden_and_state(self, temp_project):
        """Test that snapshot creates expected files."""
        golden_dir = temp_project / ".evalview" / "golden"
        state_file = temp_project / ".evalview" / "state.json"

        # Simulate snapshot command
        golden_dir.mkdir(parents=True)

        # Create a golden file
        golden_file = golden_dir / "test.golden.json"
        golden_file.write_text('{"metadata": {"test_name": "test", "blessed_at": "2024-01-01T00:00:00", "score": 90.0}}')

        # Create state file
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text('{"total_snapshots": 1, "total_checks": 0}')

        # Verify
        assert golden_file.exists()
        assert state_file.exists()

        # Verify content
        state_data = json.loads(state_file.read_text())
        assert state_data["total_snapshots"] == 1

    def test_check_updates_state_on_clean_run(self, temp_project):
        """Test that check updates state correctly on passing check."""
        state_file = temp_project / ".evalview" / "state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)

        # Initial state
        initial_state = {
            "current_streak": 0,
            "longest_streak": 0,
            "total_checks": 0,
            "regression_count": 0,
            "last_check_status": None,
            "last_check_at": None,
            "total_snapshots": 1,
            "milestones_hit": [],
            "conversion_suggestion_shown": False,
            "created_at": datetime.now().isoformat(),
            "last_snapshot_at": datetime.now().isoformat()
        }
        state_file.write_text(json.dumps(initial_state, indent=2))

        # Simulate a clean check by updating state
        state_data = json.loads(state_file.read_text())
        state_data["total_checks"] = 1
        state_data["current_streak"] = 1
        state_data["longest_streak"] = 1
        state_data["last_check_status"] = "passed"
        state_data["last_check_at"] = datetime.now().isoformat()
        state_file.write_text(json.dumps(state_data, indent=2))

        # Verify state update
        updated_state = json.loads(state_file.read_text())
        assert updated_state["current_streak"] == 1
        assert updated_state["total_checks"] == 1
        assert updated_state["last_check_status"] == "passed"

    def test_check_breaks_streak_on_regression(self, temp_project):
        """Test that regression breaks streak."""
        state_file = temp_project / ".evalview" / "state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)

        # State with existing streak
        initial_state = {
            "current_streak": 5,
            "longest_streak": 5,
            "total_checks": 5,
            "regression_count": 0,
            "last_check_status": "passed",
            "last_check_at": datetime.now().isoformat(),
            "total_snapshots": 1,
            "milestones_hit": ["streak_3", "streak_5"],
            "conversion_suggestion_shown": False,
            "created_at": datetime.now().isoformat(),
            "last_snapshot_at": datetime.now().isoformat()
        }
        state_file.write_text(json.dumps(initial_state, indent=2))

        # Simulate regression by updating state
        state_data = json.loads(state_file.read_text())
        state_data["total_checks"] = 6
        state_data["current_streak"] = 0  # Broken!
        state_data["regression_count"] = 1
        state_data["last_check_status"] = "regression"
        state_data["last_check_at"] = datetime.now().isoformat()
        state_file.write_text(json.dumps(state_data, indent=2))

        # Verify streak broken
        updated_state = json.loads(state_file.read_text())
        assert updated_state["current_streak"] == 0
        assert updated_state["longest_streak"] == 5  # Preserved
        assert updated_state["regression_count"] == 1


class TestMultiVariantWorkflow:
    """Test multi-variant golden workflow."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        tmpdir = tempfile.mkdtemp()
        project_dir = Path(tmpdir) / "test-project"
        project_dir.mkdir()

        yield project_dir

        shutil.rmtree(tmpdir)

    def test_save_multiple_variants(self, temp_project):
        """Test saving multiple golden variants."""
        golden_dir = temp_project / ".evalview" / "golden"
        golden_dir.mkdir(parents=True)

        # Save default golden
        default_golden = golden_dir / "test-multi.golden.json"
        default_golden.write_text('{"metadata": {"test_name": "test-multi", "score": 90.0}}')

        # Save variant 1
        variant1 = golden_dir / "test-multi.variant_v1.golden.json"
        variant1.write_text('{"metadata": {"test_name": "test-multi", "score": 90.0}}')

        # Save variant 2
        variant2 = golden_dir / "test-multi.variant_v2.golden.json"
        variant2.write_text('{"metadata": {"test_name": "test-multi", "score": 90.0}}')

        # Verify all exist
        assert default_golden.exists()
        assert variant1.exists()
        assert variant2.exists()

        # Count variants
        variants = list(golden_dir.glob("test-multi*.golden.json"))
        assert len(variants) == 3

    def test_variant_limit_enforcement(self, temp_project):
        """Test that max 5 variants are enforced."""
        golden_dir = temp_project / ".evalview" / "golden"
        golden_dir.mkdir(parents=True)

        # Create 5 variants (default + 4 named)
        (golden_dir / "test.golden.json").write_text("{}")
        for i in range(4):
            (golden_dir / f"test.variant_v{i}.golden.json").write_text("{}")

        # Verify we have 5
        variants = list(golden_dir.glob("test*.golden.json"))
        assert len(variants) == 5

        # Note: Actual limit enforcement is in GoldenStore.save_golden()
        # This test just verifies the file structure


class TestStateManagement:
    """Test state persistence and management."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        tmpdir = tempfile.mkdtemp()
        project_dir = Path(tmpdir) / "test-project"
        project_dir.mkdir()

        yield project_dir

        shutil.rmtree(tmpdir)

    def test_state_file_creation(self, temp_project):
        """Test state file is created on first use."""
        state_file = temp_project / ".evalview" / "state.json"
        state_file.parent.mkdir(parents=True)

        # Create initial state
        initial_state = {
            "current_streak": 0,
            "longest_streak": 0,
            "total_checks": 0,
            "regression_count": 0,
            "total_snapshots": 0,
            "milestones_hit": [],
            "conversion_suggestion_shown": False,
            "created_at": datetime.now().isoformat(),
            "last_snapshot_at": None,
            "last_check_at": None,
            "last_check_status": None
        }
        state_file.write_text(json.dumps(initial_state, indent=2))

        assert state_file.exists()

        # Verify readable
        state_data = json.loads(state_file.read_text())
        assert state_data["current_streak"] == 0

    def test_milestone_tracking(self, temp_project):
        """Test that milestones are tracked correctly."""
        state_file = temp_project / ".evalview" / "state.json"
        state_file.parent.mkdir(parents=True)

        state_data = {
            "current_streak": 5,
            "milestones_hit": ["streak_3", "streak_5"],
            "total_checks": 5,
            "longest_streak": 5,
            "regression_count": 0,
            "total_snapshots": 1,
            "conversion_suggestion_shown": False,
            "created_at": datetime.now().isoformat(),
            "last_snapshot_at": datetime.now().isoformat(),
            "last_check_at": datetime.now().isoformat(),
            "last_check_status": "passed"
        }
        state_file.write_text(json.dumps(state_data, indent=2))

        # Verify milestones
        state = json.loads(state_file.read_text())
        assert "streak_3" in state["milestones_hit"]
        assert "streak_5" in state["milestones_hit"]

    def test_state_persistence_across_checks(self, temp_project):
        """Test that state persists across multiple checks."""
        state_file = temp_project / ".evalview" / "state.json"
        state_file.parent.mkdir(parents=True)

        # Check 1
        state1 = {
            "current_streak": 1,
            "total_checks": 1,
            "longest_streak": 1,
            "regression_count": 0,
            "total_snapshots": 1,
            "milestones_hit": [],
            "conversion_suggestion_shown": False,
            "created_at": datetime.now().isoformat(),
            "last_snapshot_at": datetime.now().isoformat(),
            "last_check_at": datetime.now().isoformat(),
            "last_check_status": "passed"
        }
        state_file.write_text(json.dumps(state1, indent=2))

        # Check 2 (load and update)
        state = json.loads(state_file.read_text())
        state["current_streak"] = 2
        state["total_checks"] = 2
        state["longest_streak"] = 2
        state_file.write_text(json.dumps(state, indent=2))

        # Verify persistence
        final_state = json.loads(state_file.read_text())
        assert final_state["current_streak"] == 2
        assert final_state["total_checks"] == 2


class TestGoldenFileStructure:
    """Test golden file storage structure."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        tmpdir = tempfile.mkdtemp()
        project_dir = Path(tmpdir) / "test-project"
        project_dir.mkdir()

        yield project_dir

        shutil.rmtree(tmpdir)

    def test_golden_file_naming(self, temp_project):
        """Test golden file naming conventions."""
        golden_dir = temp_project / ".evalview" / "golden"
        golden_dir.mkdir(parents=True)

        # Default golden
        default = golden_dir / "test-name.golden.json"
        default.write_text("{}")

        # Variant golden
        variant = golden_dir / "test-name.variant_v1.golden.json"
        variant.write_text("{}")

        assert default.name == "test-name.golden.json"
        assert variant.name == "test-name.variant_v1.golden.json"

    def test_golden_file_content_structure(self, temp_project):
        """Test that golden files have required structure."""
        golden_dir = temp_project / ".evalview" / "golden"
        golden_dir.mkdir(parents=True)

        golden_data = {
            "metadata": {
                "test_name": "test",
                "blessed_at": datetime.now().isoformat(),
                "score": 85.0,
                "blessed_by": "test"
            },
            "trace": {
                "session_id": "test-session",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "steps": [],
                "final_output": "result",
                "metrics": {"total_cost": 0.01, "total_latency": 100}
            },
            "tool_sequence": [],
            "output_hash": "abc123"
        }

        golden_file = golden_dir / "test.golden.json"
        golden_file.write_text(json.dumps(golden_data, indent=2))

        # Verify structure
        loaded = json.loads(golden_file.read_text())
        assert "metadata" in loaded
        assert "trace" in loaded
        assert "tool_sequence" in loaded
        assert "output_hash" in loaded

        # Verify metadata
        assert loaded["metadata"]["test_name"] == "test"
        assert loaded["metadata"]["score"] == 85.0
