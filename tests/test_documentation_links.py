"""Regression coverage for the offline documentation link gate."""

from pathlib import Path
from subprocess import CompletedProcess

from scripts.check_docs_links import check_links, markdown_links, tracked_docs


def test_relative_links_resolve_from_the_source_document(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ADAPTERS.md").write_text("# Adapters\n")
    source = docs / "FRAMEWORK_SUPPORT.md"
    source.write_text("[valid](ADAPTERS.md)\n\n[broken](docs/ADAPTERS.md)\n")

    broken = check_links(tmp_path, [Path("docs/FRAMEWORK_SUPPORT.md")])

    assert len(broken) == 1
    assert str(broken[0]) == ("docs/FRAMEWORK_SUPPORT.md:3: missing local target: docs/ADAPTERS.md")


def test_markdown_parser_ignores_examples_but_checks_reference_and_html_links():
    text = """`[example](missing.md)`

```markdown
[example](missing.md)
```

[guide][ref]

[ref]: guide.md "A guide"

![diagram](diagram.png)

<a href="another.md">Another guide</a><img src="logo.png">
"""
    assert [target for _, target in markdown_links(text)] == [
        "guide.md",
        "diagram.png",
        "another.md",
        "logo.png",
    ]


def test_external_urls_anchors_and_encoded_local_paths(tmp_path):
    (tmp_path / "a file.md").write_text("# Heading\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/README.md").write_text(
        "[space](../a%20file.md#heading)\n"
        "[root](/a%20file.md?raw=1)\n"
        "[directory](../docs/)\n"
        "[anchor](#heading)\n"
        "[web](https://example.com/missing.md)\n"
        "[protocol relative](//example.com/path)\n"
        "[mail](mailto:maintainer@example.com)\n"
    )

    assert check_links(tmp_path, [Path("docs/README.md")]) == []


def test_only_tracked_root_and_docs_markdown_are_selected(tmp_path, monkeypatch):
    # A private draft on disk is intentionally absent from Git's tracked list.
    (tmp_path / "draft.md").write_text("[unpublished](missing.md)")

    def git_ls_files(command, **kwargs):
        assert command == ["git", "ls-files", "-z", "--", "*.md"]
        assert kwargs["cwd"] == tmp_path
        return CompletedProcess(
            command, 0, "README.md\0docs/README.md\0docs/guides/start.md\0examples/README.md\0"
        )

    monkeypatch.setattr("scripts.check_docs_links.subprocess.run", git_ls_files)

    assert tracked_docs(tmp_path) == [
        Path("README.md"),
        Path("docs/README.md"),
        Path("docs/guides/start.md"),
    ]


def test_links_cannot_resolve_to_files_outside_the_checkout(tmp_path):
    (tmp_path / "private.md").write_text("Not part of the repository")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("[outside](../private.md)")

    assert len(check_links(root, [Path("README.md")])) == 1
