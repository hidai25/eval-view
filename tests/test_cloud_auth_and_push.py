"""Tests for the v2 cloud-auth bridge introduced for the launch fix.

Covers:
  - CloudAuth.save_api_token / load round-trip and chmod 600.
  - CloudAuth treats legacy Supabase-A sessions as not-logged-in for push.
  - _resolve_cloud() precedence: env > config > auth-file.
  - _resolve_cloud() honors EVALVIEW_CLOUD_URL as a self-host override.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest


# ----------------------------------------------------------------------
# CloudAuth round-trip
# ----------------------------------------------------------------------


@pytest.fixture()
def auth_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect ~/.evalview/auth.json to a per-test file."""
    target = tmp_path / "auth.json"
    monkeypatch.setattr("evalview.cloud.auth.AUTH_FILE", target)
    yield target


def test_save_api_token_round_trip(auth_file: Path) -> None:
    from evalview.cloud.auth import CloudAuth

    auth = CloudAuth()
    auth.save_api_token(
        api_token="ev_AbCdEfGhIjKl_payload",
        cloud_url="https://evalview.com/api/v1",
        email="user@example.com",
        project_slug="default",
        org_slug="user-abc",
    )

    assert auth.is_logged_in()
    assert auth.get_api_token() == "ev_AbCdEfGhIjKl_payload"
    assert auth.get_cloud_url() == "https://evalview.com/api/v1"
    assert auth.get_email() == "user@example.com"
    assert auth.get_project_slug() == "default"
    assert auth.get_org_slug() == "user-abc"
    assert auth.get_token_prefix() == "ev_AbCdEfGh"


def test_save_api_token_chmods_to_owner_only(auth_file: Path) -> None:
    from evalview.cloud.auth import CloudAuth

    CloudAuth().save_api_token(
        api_token="ev_secret_value",
        cloud_url="https://evalview.com/api/v1",
        email="user@example.com",
        project_slug="default",
        org_slug="user-abc",
    )

    mode = auth_file.stat().st_mode & 0o777
    # Owner read+write only — group/world bits must be zero.
    assert mode == (stat.S_IRUSR | stat.S_IWUSR), oct(mode)


def test_legacy_session_loads_but_is_not_logged_in(auth_file: Path) -> None:
    """Pre-launch ``access_token`` sessions are unusable for push.

    They were minted against a Supabase project the SaaS doesn't read
    from, so we surface them in ``whoami`` (via ``load()``) but report
    ``is_logged_in() == False`` so push/check don't pretend they have
    a credential they can use.
    """
    auth_file.write_text(
        json.dumps(
            {
                "access_token": "stale-supabase-jwt",
                "refresh_token": "stale-refresh",
                "user_id": "uuid",
                "email": "legacy@example.com",
            }
        )
    )

    from evalview.cloud.auth import CloudAuth

    auth = CloudAuth()
    data = auth.load()
    assert data is not None and "access_token" in data
    assert auth.get_email() == "legacy@example.com"
    assert auth.get_api_token() is None
    assert auth.is_logged_in() is False


def test_clear_removes_file(auth_file: Path) -> None:
    from evalview.cloud.auth import CloudAuth

    CloudAuth().save_api_token(
        api_token="ev_xxx",
        cloud_url="https://evalview.com/api/v1",
        email="u@example.com",
        project_slug="default",
        org_slug="u",
    )
    assert auth_file.exists()
    CloudAuth().clear()
    assert not auth_file.exists()


# ----------------------------------------------------------------------
# _resolve_cloud precedence
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with no cloud env set."""
    monkeypatch.delenv("EVALVIEW_API_TOKEN", raising=False)
    monkeypatch.delenv("EVALVIEW_CLOUD_URL", raising=False)


def test_resolve_cloud_env_token_wins(auth_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pinned CI token always wins over the on-disk login."""
    from evalview.cloud.auth import CloudAuth
    from evalview.cloud.push import _resolve_cloud

    CloudAuth().save_api_token(
        api_token="ev_from_login",
        cloud_url="https://staging.evalview.com/api/v1",
        email="u@example.com",
        project_slug="default",
        org_slug="u",
    )
    monkeypatch.setenv("EVALVIEW_API_TOKEN", "ev_from_env")

    token, base = _resolve_cloud()
    assert token == "ev_from_env"
    # No EVALVIEW_CLOUD_URL → production default, NOT the login-file URL.
    assert base == "https://evalview.com/api/v1"


def test_resolve_cloud_falls_back_to_login_file(auth_file: Path) -> None:
    """With no env/config, the login-file token + cloud URL are used."""
    from evalview.cloud.auth import CloudAuth
    from evalview.cloud.push import _resolve_cloud

    CloudAuth().save_api_token(
        api_token="ev_from_login",
        cloud_url="https://staging.evalview.com/api/v1",
        email="u@example.com",
        project_slug="default",
        org_slug="u",
    )
    token, base = _resolve_cloud()
    assert token == "ev_from_login"
    assert base == "https://staging.evalview.com/api/v1"


def test_resolve_cloud_env_url_overrides_login_file_url(
    auth_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``EVALVIEW_CLOUD_URL`` is the self-host escape hatch and always wins."""
    from evalview.cloud.auth import CloudAuth
    from evalview.cloud.push import _resolve_cloud

    CloudAuth().save_api_token(
        api_token="ev_from_login",
        cloud_url="https://evalview.com/api/v1",
        email="u@example.com",
        project_slug="default",
        org_slug="u",
    )
    monkeypatch.setenv("EVALVIEW_CLOUD_URL", "https://self-hosted.example/api/v1")

    token, base = _resolve_cloud()
    assert token == "ev_from_login"
    assert base == "https://self-hosted.example/api/v1"


def test_resolve_cloud_no_credentials_returns_none(auth_file: Path) -> None:
    """No env, no config, no login → token is None and we never push."""
    from evalview.cloud.push import _resolve_cloud

    token, base = _resolve_cloud()
    assert token is None
    # Base URL still resolved so callers don't crash on attribute access.
    assert base == "https://evalview.com/api/v1"


def test_resolve_cloud_default_is_production_not_localhost(
    auth_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard: the launch-blocker default was localhost:3000."""
    from evalview.cloud.push import _resolve_cloud

    monkeypatch.setenv("EVALVIEW_API_TOKEN", "ev_anything")
    _, base = _resolve_cloud()
    assert "localhost" not in base
    assert base.startswith("https://")
