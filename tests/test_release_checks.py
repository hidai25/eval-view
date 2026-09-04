"""Release guards must reject stale versions and validate installable artifacts."""

import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.check_release import check_changelog, check_distributions, check_source, smoke_install


@pytest.fixture
def release_source(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "evalview"\nversion = "1.2.3"\n')
    (tmp_path / "evalview").mkdir()
    (tmp_path / "evalview/__init__.py").write_text('__version__ = "1.2.3"\n')
    (tmp_path / "server.json").write_text(
        json.dumps(
            {
                "version": "1.2.3",
                "packages": [
                    {"registryType": "pypi", "identifier": "evalview", "version": "1.2.3"}
                ],
            }
        )
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [1.2.3] - 2026-09-05\n\n### Fixed\n- A bug.\n"
    )
    return tmp_path


def test_ready_release_has_matching_source_versions(release_source):
    assert check_source(release_source, "v1.2.3") == "1.2.3"


@pytest.mark.parametrize("location", ["runtime", "server", "package"])
def test_version_drift_is_rejected(release_source, location):
    if location == "runtime":
        (release_source / "evalview/__init__.py").write_text('__version__ = "1.2.2"\n')
    else:
        manifest = release_source / "server.json"
        server = json.loads(manifest.read_text())
        target = server if location == "server" else server["packages"][0]
        target["version"] = "1.2.2"
        manifest.write_text(json.dumps(server))
    with pytest.raises(ValueError, match="expected '1.2.3'"):
        check_source(release_source, "v1.2.3")


def test_tag_cannot_claim_a_new_version_for_an_old_package(release_source):
    with pytest.raises(ValueError, match="must match package version"):
        check_source(release_source, "v1.2.4")


def test_unreleased_fix_cannot_be_published_under_the_existing_release(release_source):
    changelog = release_source / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text().replace(
            "## [Unreleased]", "## [Unreleased]\n\n### Fixed\n- Migrated an obsolete API."
        )
    )
    # Normal source builds are still allowed between releases.
    assert check_source(release_source) == "1.2.3"
    with pytest.raises(ValueError, match="pending Unreleased"):
        check_source(release_source, "v1.2.3")


@pytest.mark.parametrize(
    "changelog, reason",
    [
        ("## [1.2.2] - 2026-09-04\n- Old release", "latest release"),
        ("## [1.2.3]\n- Undated", "valid YYYY-MM-DD"),
        ("## [1.2.3] - 2026-02-30\n- Invalid date", "valid YYYY-MM-DD"),
        ("## [1.2.3] - 2026-09-05\n\n### Fixed\n", "no release notes"),
    ],
)
def test_missing_or_unfinished_release_notes_are_rejected(changelog, reason):
    with pytest.raises(ValueError, match=reason):
        check_changelog(changelog, "1.2.3")


def make_distributions(
    dist, *, wheel_version="1.2.3", runtime="1.2.3", sdist_version="1.2.3", typed=True
):
    wheel = dist / "evalview-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "evalview-1.2.3.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: evalview\nVersion: {wheel_version}\n",
        )
        archive.writestr("evalview/__init__.py", f'__version__ = "{runtime}"\n')
        if typed:
            archive.writestr("evalview/py.typed", "")
    with tarfile.open(dist / "evalview-1.2.3.tar.gz", "w:gz") as archive:
        data = f"Metadata-Version: 2.1\nName: evalview\nVersion: {sdist_version}\n".encode()
        member = tarfile.TarInfo("evalview-1.2.3/PKG-INFO")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    return wheel


def test_built_distribution_metadata_matches_release(tmp_path):
    wheel = make_distributions(tmp_path)
    assert check_distributions(tmp_path, "1.2.3") == wheel.resolve()


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"wheel_version": "1.2.2"}, "expected evalview 1.2.3"),
        ({"sdist_version": "1.2.2"}, "expected evalview 1.2.3"),
        ({"runtime": "1.2.2"}, "mismatched runtime"),
        ({"typed": False}, "missing evalview/py.typed"),
    ],
)
def test_built_distribution_regressions_are_rejected(tmp_path, change, reason):
    make_distributions(tmp_path, **change)
    with pytest.raises(ValueError, match=reason):
        check_distributions(tmp_path, "1.2.3")


def test_dirty_dist_directory_is_rejected(tmp_path):
    make_distributions(tmp_path)
    (tmp_path / "evalview-1.2.2-py3-none-any.whl").write_bytes(b"stale artifact")
    with pytest.raises(ValueError, match="exactly one wheel"):
        check_distributions(tmp_path, "1.2.3")


def test_smoke_install_cannot_import_the_checkout(tmp_path, monkeypatch):
    wheel = make_distributions(tmp_path)
    calls = []
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path))
    monkeypatch.setattr("scripts.check_release.venv.EnvBuilder.create", lambda self, path: None)

    def run(command, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["cwd"] != tmp_path
        assert "PYTHONPATH" not in kwargs["env"]
        assert "PYTHONHOME" not in kwargs["env"]
        assert kwargs["env"]["EVALVIEW_TELEMETRY_DISABLED"] == "1"
        assert kwargs["env"]["EVALVIEW_DISABLE_UPDATE_CHECK"] == "1"
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("scripts.check_release.subprocess.run", run)
    smoke_install(wheel, "1.2.3")
    assert str(wheel.resolve()) in calls[0]
    assert any(command[-1] == "--help" for command in calls)
    assert any(command[-3:] == ["-m", "evalview", "--version"] for command in calls)
    assert all("-I" in command for command in calls if Path(command[0]).name.startswith("python"))


def test_install_failure_stops_release_smoke_checks(tmp_path, monkeypatch):
    wheel = make_distributions(tmp_path)
    monkeypatch.setattr("scripts.check_release.venv.EnvBuilder.create", lambda self, path: None)

    def fail_install(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("scripts.check_release.subprocess.run", fail_install)
    with pytest.raises(subprocess.CalledProcessError):
        smoke_install(wheel, "1.2.3")
