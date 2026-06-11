"""
Persistent store for dashboard allowed emails and admins.

Saved to data/dashboard_users.json with automatic backup — survives restarts.
"""

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_ADMIN = "mjworkplace2023@gmail.com"
STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "dashboard_users.json"
BACKUP_PATH = STORE_PATH.with_suffix(".json.bak")


def store_path() -> Path:
    """Return absolute path to the user store file."""
    return STORE_PATH


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


def _try_read(path: Path) -> Optional[dict]:
    """Read and parse a store file. Returns None if missing, empty, or invalid."""
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


def _save(data: dict) -> None:
    """Atomically persist user store and keep a backup copy."""
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
        logger.info("Saved user store to %s", STORE_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _load() -> dict:
    """Load user store from disk; recover from backup if the main file is bad."""
    if not STORE_PATH.exists() and not BACKUP_PATH.exists():
        data = _default_data()
        _save(data)
        return data

    data = _try_read(STORE_PATH)
    if data is not None:
        return data

    backup = _try_read(BACKUP_PATH)
    if backup is not None:
        logger.warning("Restored user store from backup: %s", BACKUP_PATH)
        _save(backup)
        return backup

    logger.error(
        "User store unreadable at %s and %s — using defaults in memory only. "
        "Fix or delete the corrupt file(s) to re-save.",
        STORE_PATH,
        BACKUP_PATH,
    )
    return _default_data()


def _env_allowed_emails() -> Set[str]:
    raw = os.getenv("ALLOWED_EMAILS", "")
    return {_normalize(e) for e in raw.split(",") if e.strip()}


def get_admins() -> Set[str]:
    return set(_load()["admins"])


def is_admin(email: str) -> bool:
    return _normalize(email) in get_admins()


def get_allowed_emails() -> Set[str]:
    data = _load()
    allowed = set(data["allowed_emails"]) | set(data["admins"]) | _env_allowed_emails()
    return allowed


def list_allowed_emails() -> List[str]:
    return sorted(get_allowed_emails())


def add_allowed_email(email: str) -> Tuple[bool, str]:
    email = _normalize(email)
    if not _is_valid_email(email):
        return False, "Enter a valid email address."
    data = _load()
    if email in data["allowed_emails"] or email in data["admins"]:
        return False, f"{email} is already authorized."
    data["allowed_emails"].append(email)
    data["allowed_emails"] = sorted(set(data["allowed_emails"]))
    _save(data)
    return True, f"Added {email}."


def remove_allowed_email(email: str) -> Tuple[bool, str]:
    email = _normalize(email)
    data = _load()
    if email in data["admins"]:
        return False, "Admin accounts cannot be removed."
    if email not in data["allowed_emails"]:
        return False, f"{email} is not in the allowed list."
    data["allowed_emails"] = [e for e in data["allowed_emails"] if e != email]
    _save(data)
    return True, f"Removed {email}."
