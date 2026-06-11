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
for update in data.get("result", []):
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

print("\nSet in .env (group only):")
groups = [cid for cid in chats if cid.startswith("-")]
if groups:
    print(f"  TELEGRAM_CHAT_ID={groups[0]}")
    print("\nOr send to both group and your private chat:")
    print(f"  TELEGRAM_CHAT_ID={groups[0]},{os.getenv('TELEGRAM_CHAT_ID', '')}")
else:
    print("  (no group found yet — mention the bot in your group and rerun)")
