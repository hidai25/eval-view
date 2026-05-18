"""Cloud auth management — stores session in ~/.evalview/auth.json.

Schema (current): the v2 file produced by `evalview login` contains an
``api_token`` (the ``ev_…`` bearer used against the SaaS REST API),
the ``cloud_url`` it was minted against, and the project/org/email it
points at:

    {
        "api_token":   "ev_<base64url>",
        "cloud_url":   "https://evalview.com/api/v1",
        "project_slug": "default",
        "org_slug":    "hidai-l9j2",
        "email":       "hidai@example.com",
        "token_prefix": "ev_AbCdEfGh"
    }

Files written by older builds (Supabase-A GitHub OAuth flow) carried
``access_token``/``refresh_token``/``user_id`` keys. We still read
those for the ``whoami``/``logout`` helpers but treat them as
**unusable for cloud push** — a stale Supabase access token can't
authenticate against /api/v1/results. ``get_api_token()`` returns None
for legacy sessions; the CLI prompts the user to re-run
``evalview login`` to mint an ev_… token.
"""

import json
import stat
from pathlib import Path
from typing import Optional, Dict, Any


AUTH_FILE = Path.home() / ".evalview" / "auth.json"


class CloudAuth:
    """Manages cloud authentication state in ~/.evalview/auth.json."""

    # ------------------------------------------------------------------
    # v2 (current): api-token sessions minted via /cli-auth loopback
    # ------------------------------------------------------------------

    def save_api_token(
        self,
        api_token: str,
        cloud_url: str,
        email: str,
        project_slug: str,
        org_slug: str,
        token_prefix: Optional[str] = None,
    ) -> None:
        """Persist a v2 ev_… session to disk (chmod 600)."""
        AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "api_token": api_token,
            "cloud_url": cloud_url,
            "email": email,
            "project_slug": project_slug,
            "org_slug": org_slug,
            "token_prefix": token_prefix or api_token[:11],
        }
        AUTH_FILE.write_text(json.dumps(data, indent=2))
        AUTH_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)

    def load(self) -> Optional[Dict[str, Any]]:
        """Return parsed auth data, or None if missing/malformed."""
        if not AUTH_FILE.exists():
            return None
        try:
            data = json.loads(AUTH_FILE.read_text())
            if not isinstance(data, dict):
                return None
            # v2 — current
            if "api_token" in data and "cloud_url" in data:
                return data
            # v1 (legacy Supabase-A session). Still load so whoami/logout
            # work, but get_api_token() will return None for it.
            if all(
                k in data
                for k in ("access_token", "refresh_token", "user_id", "email")
            ):
                return data
        except Exception:
            pass
        return None

    def clear(self) -> None:
        """Delete the auth file (logout)."""
        if AUTH_FILE.exists():
            AUTH_FILE.unlink()

    def is_logged_in(self) -> bool:
        """Return True if a usable v2 session exists on disk."""
        data = self.load()
        return data is not None and "api_token" in data

    # ------------------------------------------------------------------
    # Accessors used by push / whoami
    # ------------------------------------------------------------------

    def get_api_token(self) -> Optional[str]:
        """Return the ev_… bearer token, or None for legacy sessions."""
        data = self.load()
        if data is None:
            return None
        return data.get("api_token")

    def get_cloud_url(self) -> Optional[str]:
        data = self.load()
        return data.get("cloud_url") if data else None

    def get_email(self) -> Optional[str]:
        data = self.load()
        return data.get("email") if data else None

    def get_project_slug(self) -> Optional[str]:
        data = self.load()
        return data.get("project_slug") if data else None

    def get_org_slug(self) -> Optional[str]:
        data = self.load()
        return data.get("org_slug") if data else None

    def get_token_prefix(self) -> Optional[str]:
        data = self.load()
        return data.get("token_prefix") if data else None
