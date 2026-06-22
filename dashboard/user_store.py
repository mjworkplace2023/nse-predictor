"""
Persistent store for dashboard allowed emails and admins.

Local file: data/dashboard_users.json (works on your Mac).

Streamlit Cloud: the container disk is wiped on redeploy/restart, so admin adds
disappear unless you also configure ONE of:

  1. GITHUB_TOKEN + GITHUB_REPO  — saves to your GitHub repo (recommended)
  2. ALLOWED_EMAILS in Streamlit secrets — comma-separated permanent emails
  3. [dashboard_users] block in Streamlit secrets (see .env.example)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)

DEFAULT_ADMIN = "mjworkplace2023@gmail.com"
STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "dashboard_users.json"
BACKUP_PATH = STORE_PATH.with_suffix(".json.bak")

# Cached GitHub file SHA for the current process (avoids extra GET on each save)
_github_sha: Optional[str] = None


def store_path() -> Path:
    return STORE_PATH


def _config_value(key: str) -> str:
    value = os.getenv(key, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return ""


def _is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()))


def _normalize(email: str) -> str:
    return email.strip().lower()


def _default_data() -> dict:
    admin = _normalize(DEFAULT_ADMIN)
    return {
        "admins": [admin],
        "allowed_emails": [admin],
    }


def _normalize_data(raw: dict) -> dict:
    admins = [_normalize(e) for e in raw.get("admins", []) if e and str(e).strip()]
    allowed = [_normalize(e) for e in raw.get("allowed_emails", []) if e and str(e).strip()]
    if DEFAULT_ADMIN.lower() not in admins:
        admins.append(DEFAULT_ADMIN.lower())
    if not allowed:
        allowed = list(admins)
    return {
        "admins": sorted(set(admins)),
        "allowed_emails": sorted(set(allowed)),
    }


def _github_settings() -> Optional[tuple[str, str, str, str]]:
    token = _config_value("GITHUB_TOKEN")
    if not token:
        return None
    repo = _config_value("GITHUB_REPO") or "mjworkplace2023/nse-predictor"
    path = _config_value("USER_STORE_GITHUB_PATH") or "data/dashboard_users.json"
    branch = _config_value("GITHUB_BRANCH") or "main"
    return token, repo, path, branch


def _github_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _load_from_github() -> Optional[dict]:
    global _github_sha
    settings = _github_settings()
    if not settings:
        return None
    token, repo, path, branch = settings
    owner, repo_name = repo.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
    try:
        response = requests.get(
            url,
            headers=_github_headers(token),
            params={"ref": branch},
            timeout=20,
        )
        if response.status_code == 404:
            logger.info("GitHub user store not found yet at %s", path)
            _github_sha = None
            return None
        response.raise_for_status()
        body = response.json()
        _github_sha = body.get("sha")
        content = base64.b64decode(body["content"]).decode("utf-8")
        return _normalize_data(json.loads(content))
    except Exception as exc:
        logger.error("Failed to load user store from GitHub: %s", exc)
        return None


def _save_to_github(data: dict) -> Tuple[bool, str]:
    global _github_sha
    settings = _github_settings()
    if not settings:
        return False, "GITHUB_TOKEN not configured"
    token, repo, path, branch = settings
    owner, repo_name = repo.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}"
    normalized = _normalize_data(data)
    payload_bytes = (json.dumps(normalized, indent=2) + "\n").encode("utf-8")
    body = {
        "message": "Update dashboard allowed emails",
        "content": base64.b64encode(payload_bytes).decode("ascii"),
        "branch": branch,
    }
    if _github_sha:
        body["sha"] = _github_sha
    try:
        response = requests.put(
            url,
            headers=_github_headers(token),
            json=body,
            timeout=20,
        )
        if response.status_code == 409 and not _github_sha:
            # File exists but we had no SHA — fetch and retry once
            loaded = _load_from_github()
            if loaded is not None:
                body["sha"] = _github_sha
                response = requests.put(
                    url,
                    headers=_github_headers(token),
                    json=body,
                    timeout=20,
                )
        response.raise_for_status()
        _github_sha = response.json().get("content", {}).get("sha", _github_sha)
        return True, f"Saved to GitHub ({repo}/{path})"
    except Exception as exc:
        detail = exc
        try:
            detail = response.json().get("message", response.text)
        except Exception:
            pass
        logger.error("Failed to save user store to GitHub: %s", detail)
        return False, str(detail)


def _load_from_secrets_block() -> Optional[dict]:
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is None:
            return None
        if "dashboard_users" not in st.secrets:
            return None
        block = st.secrets["dashboard_users"]
        raw = {
            "admins": list(block.get("admins", [])),
            "allowed_emails": list(block.get("allowed_emails", [])),
        }
        return _normalize_data(raw)
    except Exception:
        return None


def _env_allowed_emails() -> Set[str]:
    raw = _config_value("ALLOWED_EMAILS") or os.getenv("ALLOWED_EMAILS", "")
    return {_normalize(e) for e in raw.split(",") if e.strip()}


def get_secret_pinned_emails() -> Set[str]:
    """Emails from secrets/env that cannot be removed via the admin UI."""
    pinned: Set[str] = set(_env_allowed_emails())
    block = _load_from_secrets_block()
    if block:
        pinned |= set(block.get("allowed_emails", []))
        pinned |= set(block.get("admins", []))
    return pinned


def persistence_mode() -> str:
    if _github_settings():
        return "github"
    if _env_allowed_emails() or _load_from_secrets_block():
        return "secrets"
    return "local_file"


def persistence_summary() -> str:
    mode = persistence_mode()
    if mode == "github":
        _, repo, path, branch = _github_settings()  # type: ignore[misc]
        return f"GitHub `{repo}` → `{path}` (branch `{branch}`)"
    if mode == "secrets":
        return "Streamlit secrets (ALLOWED_EMAILS / dashboard_users) — local adds may reset on Cloud reboot"
    return f"Local file `{STORE_PATH}`"


def _try_read(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("User store file is empty: %s", path)
            return None
        return _normalize_data(json.loads(text))
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        logger.warning("Could not read user store %s: %s", path, exc)
        return None


def _save_local(data: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_data(data)
    payload = json.dumps(normalized, indent=2) + "\n"

    if STORE_PATH.exists():
        try:
            shutil.copy2(STORE_PATH, BACKUP_PATH)
        except OSError as exc:
            logger.warning("Could not write user-store backup: %s", exc)

    fd, tmp_path = tempfile.mkstemp(
        dir=STORE_PATH.parent,
        prefix=".dashboard_users_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, STORE_PATH)
        try:
            shutil.copy2(STORE_PATH, BACKUP_PATH)
        except OSError as exc:
            logger.warning("Could not refresh user-store backup: %s", exc)
        logger.info("Saved user store locally to %s", STORE_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _save(data: dict) -> Tuple[bool, str]:
    normalized = _normalize_data(data)
    _save_local(normalized)

    if _github_settings():
        ok, msg = _save_to_github(normalized)
        if ok:
            return True, msg
        return True, f"Saved locally; GitHub sync failed: {msg}"

    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None and not _env_allowed_emails() and not _load_from_secrets_block():
            return (
                True,
                "Saved for this session only — Streamlit Cloud wipes local files on "
                "redeploy. Add GITHUB_TOKEN to secrets for permanent storage, or set "
                "ALLOWED_EMAILS in secrets.",
            )
    except Exception:
        pass

    return True, f"Saved to {STORE_PATH}"


def _load() -> dict:
    github_data = _load_from_github()
    if github_data is not None:
        _save_local(github_data)
        return github_data

    if STORE_PATH.exists() or BACKUP_PATH.exists():
        data = _try_read(STORE_PATH)
        if data is not None:
            return data
        backup = _try_read(BACKUP_PATH)
        if backup is not None:
            logger.warning("Restored user store from backup: %s", BACKUP_PATH)
            _save_local(backup)
            return backup

    secrets_block = _load_from_secrets_block()
    if secrets_block is not None:
        _save_local(secrets_block)
        return secrets_block

    if not STORE_PATH.exists():
        data = _default_data()
        _save_local(data)
        return data

    logger.error("User store unreadable — using defaults in memory")
    return _default_data()


def get_admins() -> Set[str]:
    return set(_load()["admins"])


def is_admin(email: str) -> bool:
    return _normalize(email) in get_admins()


def get_allowed_emails() -> Set[str]:
    data = _load()
    allowed = set(data["allowed_emails"]) | set(data["admins"]) | _env_allowed_emails()
    block = _load_from_secrets_block()
    if block:
        allowed |= set(block["allowed_emails"]) | set(block["admins"])
    return allowed


def list_allowed_emails() -> List[str]:
    return sorted(get_allowed_emails())


def add_allowed_email(email: str) -> Tuple[bool, str]:
    email = _normalize(email)
    if not _is_valid_email(email):
        return False, "Enter a valid email address."
    data = _load()
    if email in data["allowed_emails"] or email in data["admins"]:
        if email in get_allowed_emails():
            return False, f"{email} is already authorized."
    data["allowed_emails"].append(email)
    data["allowed_emails"] = sorted(set(data["allowed_emails"]))
    ok, msg = _save(data)
    if ok:
        return True, f"Added {email}. {msg}"
    return False, msg


def remove_allowed_email(email: str) -> Tuple[bool, str]:
    email = _normalize(email)
    if email in get_secret_pinned_emails():
        return False, f"{email} is pinned in Streamlit secrets and cannot be removed here."
    data = _load()
    if email in data["admins"]:
        return False, "Admin accounts cannot be removed."
    if email not in data["allowed_emails"]:
        return False, f"{email} is not in the stored list."
    data["allowed_emails"] = [e for e in data["allowed_emails"] if e != email]
    ok, msg = _save(data)
    if ok:
        return True, f"Removed {email}. {msg}"
    return False, msg
