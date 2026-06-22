"""Admin page — manage allowed dashboard emails."""

import streamlit as st

from alerts.telegram_alert import (
    list_known_telegram_chats,
    send_telegram_test,
    telegram_config_summary,
)
from dashboard.user_store import (
    add_allowed_email,
    get_admins,
    get_secret_pinned_emails,
    list_allowed_emails,
    persistence_mode,
    persistence_summary,
    remove_allowed_email,
    store_path,
)


def render_admin_page() -> None:
    st.title("👤 Admin — User Access")
    st.caption("Add or remove emails that can log in to the dashboard.")

    mode = persistence_mode()
    if mode == "github":
        st.success(f"Persistent storage: {persistence_summary()}")
    elif mode == "secrets":
        st.info(
            f"Partial persistence via secrets. {persistence_summary()} "
            "For admin adds that survive reboot, set **GITHUB_TOKEN** in Streamlit secrets."
        )
    else:
        st.warning(
            "Emails saved to a **local file only**. On Streamlit Cloud this file is "
            "**wiped on redeploy/restart** (every few hours). Fix: add **GITHUB_TOKEN** "
            "to Streamlit secrets, or set **ALLOWED_EMAILS** for permanent users."
        )

    admins = get_admins()

    st.subheader("Add email")
    with st.form("add_email_form", clear_on_submit=True):
        new_email = st.text_input("Email address", placeholder="user@example.com")
        submitted = st.form_submit_button("Add email", type="primary")
        if submitted:
            ok, msg = add_allowed_email(new_email)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    st.markdown("---")
    st.subheader("Authorized emails")

    allowed = list_allowed_emails()
    if not allowed:
        st.info("No authorized emails yet.")
        return

    for email in allowed:
        col_email, col_role, col_action = st.columns([4, 2, 2])
        with col_email:
            st.markdown(f"**{email}**")
        with col_role:
            if email in admins:
                st.markdown("🛡️ Admin")
            elif email in pinned:
                st.markdown("📌 Secrets")
            else:
                st.markdown("User")
        with col_action:
            if email not in admins and email not in pinned:
                if st.button("Remove", key=f"remove_{email}", use_container_width=True):
                    ok, msg = remove_allowed_email(email)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    st.caption(
        f"{len(allowed)} authorized · {len(admins)} admin(s). "
        "Admin accounts cannot be removed from this page."
    )
    st.caption(f"Local path: `{store_path()}` · Mode: **{mode}**")

    st.markdown("---")
    st.subheader("📨 Telegram test")
    st.caption(f"Configured: {telegram_config_summary()}")

    if st.button("Run Telegram test", type="primary", key="admin_tg_test"):
        with st.spinner("Testing…"):
            ok, detail, sent_to, errors = send_telegram_test()
        if ok and not errors:
            st.success(detail)
        elif sent_to and errors:
            st.warning(f"Partial delivery — {detail}")
        else:
            st.error(detail)
        for err in errors:
            st.code(err)

    known = list_known_telegram_chats()
    if known:
        st.markdown("**Chats your bot has seen** (use GROUP id for `TELEGRAM_GROUP_CHAT_ID`):")
        for cid, name in sorted(known):
            tag = "GROUP" if cid.startswith("-") else "personal"
            st.text(f"{cid}  [{tag}]  {name}")
    else:
        st.info(
            "Bot has not seen your group yet. Add the bot to the group, send "
            "`@YourBotUsername test`, then click **Run Telegram test** again."
        )
