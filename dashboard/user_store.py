"""
Persistent store for dashboard allowed emails and admins.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Set, Tuple

DEFAULT_ADMIN = "mjworkplace2023@gmail.com"
STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "dashboard_users.json"


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


def _load() -> dict:
    if not STORE_PATH.exists():
        data = _default_data()
        _save(data)
        return data
    try:
        with open(STORE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        admins = [_normalize(e) for e in data.get("admins", []) if e.strip()]
        allowed = [_normalize(e) for e in data.get("allowed_emails", []) if e.strip()]
        if DEFAULT_ADMIN.lower() not in admins:
            admins.append(DEFAULT_ADMIN.lower())
        if not allowed:
            allowed = list(admins)
        return {"admins": sorted(set(admins)), "allowed_emails": sorted(set(allowed))}
    except (json.JSONDecodeError, OSError):
        data = _default_data()
        _save(data)
        return data


def _save(data: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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
