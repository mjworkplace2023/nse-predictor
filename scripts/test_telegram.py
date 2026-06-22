#!/usr/bin/env python3
"""
Test Telegram delivery to group + personal chat.

Run from project root:
  python scripts/test_telegram.py

Optional — test specific IDs:
  python scripts/test_telegram.py -1003947727003 5966698118
"""

from __future__ import annotations

import json
import os
import sys

import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))


def _env(key: str) -> str:
    return os.getenv(key, "").strip()


TOKEN = _env("TELEGRAM_BOT_TOKEN")
API = f"https://api.telegram.org/bot{TOKEN}"


def api(method: str, **params) -> dict:
    r = requests.get(f"{API}/{method}", params=params, timeout=20)
    try:
        return r.json()
    except Exception:
        return {"ok": False, "description": r.text[:500]}


def api_post(method: str, payload: dict) -> dict:
    r = requests.post(f"{API}/{method}", json=payload, timeout=20)
    try:
        return r.json()
    except Exception:
        return {"ok": False, "description": r.text[:500]}


def print_chat_list() -> None:
    print("\n=== Chats seen by bot (getUpdates) ===")
    print("If empty: open your group and send  @YourBotUsername  test\n")
    data = api("getUpdates")
    if not data.get("ok"):
        print("getUpdates FAILED:", data.get("description"))
        return

    seen: dict[str, str] = {}
    for update in data.get("result", []):
        msg = update.get("message") or {}
        if "migrate_to_chat_id" in msg:
            old = msg.get("chat", {}).get("id")
            new = msg["migrate_to_chat_id"]
            print(f"  MIGRATION: {old} -> {new}  (use the NEW id for TELEGRAM_GROUP_CHAT_ID)")

        for key in ("message", "my_chat_member", "chat_member"):
            obj = update.get(key)
            if not obj or "chat" not in obj:
                continue
            c = obj["chat"]
            cid = str(c["id"])
            label = c.get("title") or c.get("first_name") or c.get("username") or c.get("type")
            seen[cid] = f"{c.get('type')}: {label}"

    if not seen:
        print("  (none — mention your bot in the group, then run again)")
        return

    for cid, label in sorted(seen.items()):
        tag = "GROUP" if cid.startswith("-") else "PERSONAL"
        print(f"  {cid:>18}  [{tag:8}]  {label}")


def test_chat(chat_id: str) -> bool:
    print(f"\n=== Testing chat_id: {chat_id} ===")

    info = api("getChat", chat_id=chat_id)
    if info.get("ok"):
        r = info["result"]
        print(f"  getChat OK  type={r.get('type')}  name={r.get('title') or r.get('first_name')}")
    else:
        print(f"  getChat FAIL  {info.get('description')}")
        print("  -> Bot is NOT in this chat, or chat_id is wrong")

    payload = {
        "chat_id": chat_id,
        "text": f"NSE Predictor test message to chat_id {chat_id}",
    }
    sent = api_post("sendMessage", payload)
    if sent.get("ok"):
        print("  sendMessage OK  Check Telegram now!")
        return True

    print(f"  sendMessage FAIL  {sent.get('description')}")
    return False


def main() -> None:
    if not TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN missing in .env")
        sys.exit(1)

    me = api("getMe")
    if me.get("ok"):
        u = me["result"].get("username", "?")
        print(f"Bot: @{u}")
    else:
        print("ERROR: Invalid bot token —", me.get("description"))
        sys.exit(1)

    group = _env("TELEGRAM_GROUP_CHAT_ID")
    personal = _env("TELEGRAM_CHAT_ID")
    print(f"\nFrom .env:")
    print(f"  TELEGRAM_GROUP_CHAT_ID = {group or '(not set)'}")
    print(f"  TELEGRAM_CHAT_ID       = {personal or '(not set)'}")

    if " " in os.environ.get("TELEGRAM_GROUP_CHAT_ID", ""):
        print("\nWARNING: Remove spaces around = in .env:")
        print("  TELEGRAM_GROUP_CHAT_ID=-1003947727003")

    print_chat_list()

    ids = sys.argv[1:] if len(sys.argv) > 1 else [x for x in (group, personal) if x]
    if not ids:
        print("\nNo chat IDs to test. Set TELEGRAM_GROUP_CHAT_ID in .env")
        sys.exit(1)

    results = {cid: test_chat(cid) for cid in ids}

    print("\n=== SUMMARY ===")
    for cid, ok in results.items():
        status = "OK" if ok else "FAILED"
        dest = "group" if cid.startswith("-") else "personal"
        print(f"  {cid} ({dest}): {status}")

    if group and not results.get(group):
        print(
            "\nGroup failed? Copy the [GROUP] id from getUpdates above, "
            "set TELEGRAM_GROUP_CHAT_ID=<that id> (no spaces), reboot Streamlit."
        )


if __name__ == "__main__":
    main()
