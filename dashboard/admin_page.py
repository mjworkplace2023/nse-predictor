"""Admin page — manage allowed dashboard emails."""

import streamlit as st

from dashboard.user_store import (
    add_allowed_email,
    get_admins,
    list_allowed_emails,
    remove_allowed_email,
)


def render_admin_page() -> None:
    st.title("👤 Admin — User Access")
    st.caption("Add or remove emails that can log in to the dashboard.")

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
            else:
                st.markdown("User")
        with col_action:
            if email not in admins:
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
