#!/usr/bin/env python3
"""
List Telegram chat IDs seen by your bot (from recent getUpdates).

Usage:
  1. Add the bot to your group (as member or admin).
  2. Post any message in the group that mentions the bot, e.g. @YourBotName hello
     (or disable privacy mode via @BotFather → /setprivacy → Disable).
  3. Run: python scripts/get_telegram_chat_id.py

Group/supergroup IDs are negative (e.g. -1001234567890).
Personal chat IDs are positive numbers.
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
if not TOKEN:
    print("Set TELEGRAM_BOT_TOKEN in .env first.")
    sys.exit(1)

url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
resp = requests.get(url, timeout=15)
data = resp.json()

if not data.get("ok"):
    print("Telegram API error:", json.dumps(data, indent=2))
    sys.exit(1)

chats: dict[str, dict] = {}
migrations: list[tuple[str, str]] = []

for update in data.get("result", []):
    msg = update.get("message") or {}
    if "migrate_to_chat_id" in msg:
        old_id = str(msg.get("chat", {}).get("id", "?"))
        new_id = str(msg["migrate_to_chat_id"])
        migrations.append((old_id, new_id))

    for key in ("message", "channel_post", "my_chat_member", "chat_member"):
        obj = update.get(key)
        if not obj or "chat" not in obj:
            continue
        chat = obj["chat"]
        cid = str(chat["id"])
        chats[cid] = {
            "type": chat.get("type", "?"),
            "title": chat.get("title")
            or chat.get("first_name")
            or chat.get("username")
            or cid,
        }

if not chats:
    print("No chats found. Steps:")
    print("  1. Open your Telegram group")
    print("  2. Send: @<YourBotUsername> test")
    print("  3. Run this script again")
    sys.exit(0)

print("Chats your bot has seen:\n")
for cid, info in sorted(chats.items(), key=lambda x: x[0]):
    kind = info["type"]
    label = "GROUP" if cid.startswith("-") else "PERSONAL"
    print(f"  {cid:>16}  [{label:8}]  {kind}: {info['title']}")

if migrations:
    print("\n⚠️  Group was upgraded to supergroup — use the NEW id:")
    for old_id, new_id in migrations:
        print(f"  old {old_id}  →  TELEGRAM_GROUP_CHAT_ID={new_id}")

print("\nSet in .env:\n")
groups = [cid for cid in chats if cid.startswith("-")]
personals = [cid for cid in chats if not cid.startswith("-")]

if groups:
    print("  # Group alerts (required for Telegram group):")
    print(f"  TELEGRAM_GROUP_CHAT_ID={groups[0]}")
    print("\n  # Optional — also send to your private bot chat:")
    if personals:
        print(f"  TELEGRAM_CHAT_ID={personals[0]}")
    else:
        print("  TELEGRAM_CHAT_ID=your_personal_user_id")
    print("\n  # Or both in one line:")
    print(f"  TELEGRAM_CHAT_ID={groups[0]},{personals[0] if personals else 'YOUR_USER_ID'}")
else:
    print("  (no group found yet)")
    print("  1. Add bot to your Telegram group")
    print("  2. Send: @<YourBotUsername> test")
    print("  3. Run this script again")
    if personals:
        print(f"\n  Personal chat only (current): TELEGRAM_CHAT_ID={personals[0]}")

print(
    "\nNote: If an old group id like -39477270 stopped working, the supergroup id "
    "is usually -10039477270 (prepend -100 after the minus sign)."
)
