"""Validate release metadata and built distributions without publishing anything.

With no --tag, this checks version consistency for development builds. --tag
also enforces a dated, completed changelog entry for the exact release version.
--smoke installs the checked wheel into a fresh virtual environment.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from datetime import date
from email.parser import BytesParser
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 / 3.10
    import tomli as tomllib


def runtime_version(source: str) -> str:
    """Read the version without importing the checkout or its dependencies."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError("evalview/__init__.py must declare a string __version__")


def check_changelog(text: str, version: str) -> None:
    headings = list(re.finditer(r"^## \[([^\]]+)\](?: - ([^\n]+))?\s*$", text, re.MULTILINE))
    releases = [heading for heading in headings if heading[1] != "Unreleased"]
    if not releases or releases[0][1] != version:
        raise ValueError(f"CHANGELOG.md must have [{version}] as its latest release")
    release = releases[0]
    try:
        date.fromisoformat(release[2] or "")
    except ValueError as error:
        raise ValueError(
            f"CHANGELOG.md [{version}] needs a valid YYYY-MM-DD release date"
        ) from error

    def section_body(heading: re.Match[str]) -> str:
        index = headings.index(heading)
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = re.sub(r"<!--.*?-->", "", text[heading.end() : end], flags=re.DOTALL)
        return re.sub(r"^#{3,6}\s.*$", "", body, flags=re.MULTILINE).strip()

    if not section_body(release):
        raise ValueError(f"CHANGELOG.md [{version}] has no release notes")
    for heading in headings:
        if heading[1] == "Unreleased" and section_body(heading):
            raise ValueError(
                "CHANGELOG.md has pending Unreleased entries; assign them to the release"
            )


def check_source(root: Path, tag: str | None = None) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    if project["name"] != "evalview" or not isinstance(version, str):
        raise ValueError("pyproject.toml must declare the evalview package and a string version")
    versions = {
        "evalview.__version__": runtime_version(
            (root / "evalview/__init__.py").read_text(encoding="utf-8")
        ),
    }
    server = json.loads((root / "server.json").read_text(encoding="utf-8"))
    versions["server.json version"] = server["version"]
    packages = [
        package
        for package in server["packages"]
        if package.get("registryType") == "pypi" and package.get("identifier") == "evalview"
    ]
    if not packages:
        raise ValueError("server.json must declare the evalview PyPI package")
    versions.update(
        {f"server.json packages[{i}]": package["version"] for i, package in enumerate(packages)}
    )
    for location, actual in versions.items():
        if actual != version:
            raise ValueError(f"{location} is {actual!r}; expected {version!r} from pyproject.toml")
    if tag is not None:
        if tag != f"v{version}":
            raise ValueError(f"Release tag {tag!r} must match package version: v{version}")
        check_changelog((root / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    return version


def _check_metadata(data: bytes, version: str, artifact: Path) -> None:
    metadata = BytesParser().parsebytes(data)
    if metadata["Name"] != "evalview" or metadata["Version"] != version:
        raise ValueError(
            f"{artifact.name} contains {metadata['Name']} {metadata['Version']}; "
            f"expected evalview {version}"
        )


def check_distributions(dist: Path, version: str) -> Path:
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(f"{dist} must contain exactly one wheel and one source distribution")
    wheel, sdist = wheels[0], sdists[0]
    if (
        not wheel.name.startswith(f"evalview-{version}-")
        or sdist.name != f"evalview-{version}.tar.gz"
    ):
        raise ValueError(
            f"Distribution filenames must match evalview {version}; use a clean dist directory"
        )
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError(f"{wheel.name} must contain exactly one wheel METADATA file")
        _check_metadata(archive.read(metadata_files[0]), version, wheel)
        if runtime_version(archive.read("evalview/__init__.py").decode("utf-8")) != version:
            raise ValueError(f"{wheel.name} contains a mismatched runtime __version__")
        if "evalview/py.typed" not in archive.namelist():
            raise ValueError(f"{wheel.name} is missing evalview/py.typed")
    with tarfile.open(sdist, "r:gz") as archive:
        metadata_file = archive.extractfile(f"evalview-{version}/PKG-INFO")
        if metadata_file is None:
            raise ValueError(f"{sdist.name} is missing PKG-INFO")
        _check_metadata(metadata_file.read(), version, sdist)
    return wheel.resolve()


def smoke_install(wheel: Path, version: str) -> None:
    """Install only the distribution into a clean env, away from the source tree."""
    wheel = wheel.resolve()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(EVALVIEW_TELEMETRY_DISABLED="1", EVALVIEW_DISABLE_UPDATE_CHECK="1", CI="1")
    with tempfile.TemporaryDirectory(prefix="evalview-release-") as directory:
        work = Path(directory)
        environment = work / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        binaries = environment / ("Scripts" if os.name == "nt" else "bin")
        python = binaries / ("python.exe" if os.name == "nt" else "python")
        cli = binaries / ("evalview.exe" if os.name == "nt" else "evalview")

        def run(command: list[str]) -> None:
            subprocess.run(command, cwd=work, env=env, check=True)

        run([str(python), "-I", "-m", "pip", "install", "--disable-pip-version-check", str(wheel)])
        run([str(python), "-I", "-m", "pip", "check"])
        run(
            [
                str(python),
                "-I",
                "-c",
                "import sys, evalview; from importlib.metadata import version; "
                "assert evalview.__version__ == version('evalview') == sys.argv[1]; "
                "from evalview import gate, gate_async; print('Installed import OK')",
                version,
            ]
        )
        run([str(cli), "--version"])
        run([str(cli), "--help"])
        run([str(python), "-I", "-m", "evalview", "--version"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Release tag (vX.Y.Z); also requires completed release notes")
    parser.add_argument(
        "--dist", type=Path, help="Directory containing a freshly built wheel and sdist"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Install and smoke-test the wheel (needs network)"
    )
    args = parser.parse_args(argv)
    if args.smoke and args.dist is None:
        parser.error("--smoke requires --dist")
    root = Path(__file__).resolve().parents[1]
    try:
        version = check_source(root, args.tag)
        if args.dist is not None:
            wheel = check_distributions(args.dist, version)
            if args.smoke:
                smoke_install(wheel, version)
    except (
        ValueError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        print(f"Release check failed: {error}", file=sys.stderr)
        return 1
    scope = "Release preflight" if args.tag else "Version/build consistency (not release readiness)"
    print(f"{scope} OK: evalview {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
