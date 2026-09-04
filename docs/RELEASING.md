# Releasing EvalView

Merging a fix does not update the package on PyPI. A release needs a new version,
dated release notes, a validated distribution, and a successful publish job.
An `[Unreleased]` changelog entry describes source changes, not an available PyPI
upgrade. Do not announce `pip install --upgrade evalview` as delivering a fix until
the published package has been checked.

## Prepare the release

1. Choose the next version for the actual compatibility impact. Update
   `pyproject.toml`, `evalview/__init__.py`, and both the top-level and PyPI package
   versions in `server.json`. Refresh `uv.lock` with `uv lock`.
2. Move the changes being released from `[Unreleased]` into a new
   `## [X.Y.Z] - YYYY-MM-DD` changelog section. Describe migrations, breaking
   changes, minimum dependencies, and workarounds explicitly. Leave `[Unreleased]`
   empty on the release commit.
3. Run the relevant tests and the normal CI checks. Review the affected dogfood
   results; a passing package build does not establish agent behavior correctness.
4. Run a preflight with the chosen tag (replace `X.Y.Z` below). It checks the
   version across package metadata, runtime, and the MCP manifest, plus the date
   and contents of the latest release notes.

```bash
python -m pip install build twine
python scripts/check_release.py --tag vX.Y.Z
```

Python 3.11+ includes the TOML parser. On Python 3.9/3.10, use the development
environment, which includes `tomli`.

## Build and test the artifacts

Use a fresh output directory so artifacts from an earlier version cannot enter
the upload. The following commands do not publish anything:

```bash
release_dist=$(mktemp -d)
python -m build --outdir "$release_dist"
python -m twine check "$release_dist"/*
python scripts/check_release.py --tag vX.Y.Z --dist "$release_dist" --smoke
```

The artifact check requires exactly one wheel and one source distribution with
matching filenames and metadata. It also checks the wheel's runtime version and
type marker. The smoke test creates a fresh virtual environment, installs the
wheel with its dependencies, runs `pip check`, imports the public API, and checks
the console and module CLI entrypoints from outside the source checkout. It needs
package-index access, but no provider credentials or LLM calls.

For an ordinary development build, omit `--tag`. That validates version/build
consistency while allowing pending `[Unreleased]` entries; it does **not** certify
the checkout as ready to release.

## Publish and verify

After reviewing and committing the release preparation, create the matching
`vX.Y.Z` tag and publish its GitHub release. The
[publish workflow](../.github/workflows/publish.yml) checks the release tag and
changelog before building, then validates and smoke-tests the artifacts before
uploading them to PyPI.

The upload deliberately fails if files for that version already exist. Do not
turn an upload failure into success with `--skip-existing`: an existing version
may contain older code. If a publish partially succeeds, inspect the files
already on PyPI and resolve the release explicitly; never overwrite a published
version or claim a skipped upload shipped the fix.

Finally, verify the publish job and install the **specific released version**
from PyPI in a fresh environment. Confirm `evalview --version`, the import version,
and the affected behavior. Only then update installation guidance and announce
that the release contains the fix.

Related: [Operating Model](OPERATING_MODEL.md),
[Internal Dogfooding](INTERNAL_DOGFOODING.md), [Changelog](../CHANGELOG.md).
