"""
Email OTP authentication for the Streamlit dashboard.

Allowed emails are managed via the admin page (stored in data/dashboard_users.json).
OTP is sent via SMTP; sessions expire after SESSION_DURATION_MINUTES.
"""

import logging
import os
import re
import secrets
import smtplib
import ssl
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict

import streamlit as st

from dashboard.user_store import get_allowed_emails, is_admin

logger = logging.getLogger(__name__)

LOGO_PATH = Path(__file__).resolve().parent / "assets" / "butterfly_investment_logo.svg"

OTP_LENGTH = 6
OTP_VALID_MINUTES = int(os.getenv("OTP_VALID_MINUTES", "3"))
OTP_VALID_SECONDS = OTP_VALID_MINUTES * 60
SESSION_MINUTES = int(os.getenv("SESSION_DURATION_MINUTES", "60"))
SESSION_SECONDS = SESSION_MINUTES * 60
OTP_RESEND_SECONDS = int(os.getenv("OTP_RESEND_SECONDS", "60"))
MAX_VERIFY_ATTEMPTS = 5

# In-memory OTP store: email -> {otp, expires_at, attempts, last_sent_at}
_otp_store: Dict[str, dict] = {}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()))


def _generate_otp() -> str:
    return "".join(secrets.choice("0123456789") for _ in range(OTP_LENGTH))


def _smtp_configured() -> bool:
    return bool(
        os.getenv("SMTP_HOST")
        and os.getenv("SMTP_USER")
        and os.getenv("SMTP_PASSWORD")
    )


def _send_otp_email(to_email: str, otp: str) -> tuple[bool, str]:
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "").replace(" ", "")
    from_email = os.getenv("SMTP_FROM_EMAIL", user)
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() != "false"

    subject = "BFI NSE Predictor Login Code"
    body = (
        f"Your one-time login code is: {otp}\n\n"
        f"This code expires in {OTP_VALID_MINUTES} minutes.\n"
        "If you did not request this, you can ignore this email."
    )

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.ehlo()
            if use_tls:
                server.starttls(context=ctx)
                server.ehlo()
            server.login(user, password)
            server.sendmail(from_email, [to_email], msg.as_string())
        return True, ""
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed for %s", user)
        return False, (
            "Gmail rejected the app password. Create a new App Password at "
            "https://myaccount.google.com/apppasswords "
            "(requires 2-Step Verification), update SMTP_PASSWORD in .env, and restart the app."
        )
    except smtplib.SMTPException as exc:
        logger.error("SMTP error sending OTP to %s: %s", to_email, exc)
        return False, f"Email server error: {exc}"
    except OSError as exc:
        logger.error("Network error sending OTP to %s: %s", to_email, exc)
        return False, (
            "Could not reach the mail server. Check your network or try a different connection "
            "(corporate networks often block SMTP port 587)."
        )
    except Exception as exc:
        logger.error("Failed to send OTP email to %s: %s", to_email, exc)
        return False, "Could not send email. Check SMTP settings in .env."


def _store_otp(email: str, otp: str) -> None:
    now = _utcnow()
    _otp_store[email] = {
        "otp": otp,
        "expires_at": now + timedelta(minutes=OTP_VALID_MINUTES),
        "attempts": 0,
        "last_sent_at": now,
    }


def _can_resend(email: str) -> tuple[bool, int]:
    entry = _otp_store.get(email)
    if not entry:
        return True, 0
    if _utcnow() > entry["expires_at"]:
        return True, 0
    elapsed = (_utcnow() - entry["last_sent_at"]).total_seconds()
    if elapsed >= OTP_RESEND_SECONDS:
        return True, 0
    return False, int(OTP_RESEND_SECONDS - elapsed)


def request_otp(email: str) -> tuple[bool, str]:
    """Validate email and send OTP. Returns (success, message)."""
    email = email.strip().lower()
    allowed = get_allowed_emails()

    if not allowed:
        return False, "No authorized emails configured. Ask the admin to add your email."
    if not _is_valid_email(email):
        return False, "Enter a valid email address."
    if email not in allowed:
        return False, "This email is not authorized to access the dashboard."

    if not _smtp_configured():
        return False, "Email service is not configured. Set SMTP_* variables in .env."

    can_send, wait_secs = _can_resend(email)
    if not can_send:
        return False, f"Please wait {wait_secs} seconds before requesting another code."

    otp = _generate_otp()
    sent, err = _send_otp_email(email, otp)
    if not sent:
        return False, err

    _store_otp(email, otp)
    return True, f"A {OTP_LENGTH}-digit code was sent to {email}."


def verify_otp(email: str, otp: str) -> tuple[bool, str]:
    """Verify OTP and start session. Returns (success, message)."""
    email = email.strip().lower()
    otp = otp.strip()

    entry = _otp_store.get(email)
    if not entry:
        return False, "No active code for this email. Click Send OTP first."

    if _utcnow() > entry["expires_at"]:
        _otp_store.pop(email, None)
        return False, "Code expired. Request a new one."

    entry["attempts"] += 1
    if entry["attempts"] > MAX_VERIFY_ATTEMPTS:
        _otp_store.pop(email, None)
        return False, "Too many failed attempts. Request a new code."

    if otp != entry["otp"]:
        remaining = MAX_VERIFY_ATTEMPTS - entry["attempts"]
        return False, f"Incorrect code. {remaining} attempt(s) left."

    _otp_store.pop(email, None)
    expires = _utcnow() + timedelta(minutes=SESSION_MINUTES)
    st.session_state.auth_user = email
    st.session_state.auth_expires = expires
    st.session_state.auth_step = "logged_in"
    return True, f"Logged in. Session active for {SESSION_MINUTES} minutes."


def logout() -> None:
    for key in ("auth_user", "auth_expires", "auth_step", "login_email", "otp_sent"):
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    user = st.session_state.get("auth_user")
    expires = st.session_state.get("auth_expires")
    if not user or not expires:
        return False
    if _utcnow() > expires:
        logout()
        return False
    return True


def session_remaining_minutes() -> int:
    return session_remaining_seconds() // 60


def session_remaining_seconds() -> int:
    expires = st.session_state.get("auth_expires")
    if not expires:
        return 0
    return max(0, int((expires - _utcnow()).total_seconds()))


def session_remaining_display() -> str:
    secs = session_remaining_seconds()
    if secs <= 0:
        return "Expired"
    m, s = divmod(secs, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h {m}m"
    return f"{m}m {s}s"


def otp_remaining_seconds(email: str) -> int:
    entry = _otp_store.get(email.strip().lower())
    if not entry:
        return 0
    return max(0, int((entry["expires_at"] - _utcnow()).total_seconds()))


def is_otp_expired(email: str) -> bool:
    email = email.strip().lower()
    entry = _otp_store.get(email)
    if not entry:
        return True
    if _utcnow() > entry["expires_at"]:
        _otp_store.pop(email, None)
        return True
    return False


def _init_login_state() -> None:
    st.session_state.setdefault("auth_step", "email")
    st.session_state.setdefault("login_email", "")
    st.session_state.setdefault("otp_sent", False)


def _render_login_page() -> None:
    _init_login_state()

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 4rem;
            max-width: 1040px;
            min-height: 90vh;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            min-height: 720px;
            padding: 3rem 4rem;
        }
        [data-testid="stImage"] { text-align: center; margin-bottom: 0.5rem; }
        [data-testid="stImage"] img { max-height: 176px; width: auto; margin: 0 auto; }
        .login-brand-sub {
            text-align: center; color: #78909c; letter-spacing: 0.22em;
            font-size: 1rem; margin: 0.25rem 0 1.5rem 0; text-transform: uppercase;
        }
        .login-heading {
            text-align: center; color: #37474f; font-size: 1.4rem;
            font-weight: 600; margin: 0 0 0.75rem 0;
        }
        .login-desc {
            text-align: center; color: #666; font-size: 1.1rem;
            margin: 0 0 1.5rem 0; line-height: 1.6;
        }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stAlert"] {
            text-align: center;
        }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stAlert"] p,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stAlert"] div {
            text-align: center;
            justify-content: center;
        }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stTextInput"] label p {
            text-align: center;
        }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stTextInput"] input {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([0.15, 4.7, 0.15])
    with center:
        with st.container(border=True):
            if LOGO_PATH.exists():
                lc1, lc2, lc3 = st.columns([1, 3, 1])
                with lc2:
                    st.image(str(LOGO_PATH), use_container_width=True)
            else:
                st.markdown(
                    "<h3 style='text-align:center;color:#1a237e;margin:0;'>"
                    "🦋 Butterfly Investment</h3>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<p class="login-brand-sub">NSE Stock Predictor</p>'
                '<p class="login-heading">Sign in to continue</p>'
                '<p class="login-desc">Use your authorized email.<br>'
                "A one-time code will be sent to your inbox.</p>",
                unsafe_allow_html=True,
            )

            if not get_allowed_emails():
                st.error("No authorized emails configured. Contact the administrator.")
                st.stop()

            if not _smtp_configured():
                st.warning(
                    "SMTP is not configured. Add `SMTP_HOST`, `SMTP_USER`, and "
                    "`SMTP_PASSWORD` to `.env`."
                )

            _fp1, form_col, _fp2 = st.columns([1, 2.2, 1])
            with form_col:
                st.info(
                    f"OTP valid for **{OTP_VALID_SECONDS} seconds** ({OTP_VALID_MINUTES} min) · "
                    f"Session lasts **{SESSION_MINUTES} min** after login."
                )

                email = st.text_input(
                    "Email address",
                    value=st.session_state.login_email,
                    placeholder="you@company.com",
                    disabled=st.session_state.otp_sent,
                )

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    send_otp = st.button("Send OTP", type="primary", use_container_width=True)
                with btn_col2:
                    if st.session_state.otp_sent:
                        if st.button("Change email", use_container_width=True):
                            st.session_state.otp_sent = False
                            st.session_state.login_email = ""
                            st.session_state.auth_step = "email"
                            st.rerun()

                if send_otp:
                    ok, msg = request_otp(email)
                    if ok:
                        st.session_state.login_email = email.strip().lower()
                        st.session_state.otp_sent = True
                        st.session_state.auth_step = "otp"
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

                if st.session_state.otp_sent and st.session_state.login_email:
                    st.divider()
                    _render_otp_section(st.session_state.login_email)


def _render_otp_section(email: str) -> None:
    """OTP entry, live countdown, verify, and regenerate when expired."""

    @st.fragment(run_every=timedelta(seconds=1))
    def _otp_countdown() -> None:
        if is_otp_expired(email):
            st.error(
                f"OTP expired. It was valid for **{OTP_VALID_SECONDS} seconds**. "
                "Generate a new code below."
            )
        else:
            remaining = otp_remaining_seconds(email)
            st.success(f"OTP active — **{remaining}** seconds remaining")

    _otp_countdown()

    otp = st.text_input(
        "Enter OTP",
        max_chars=OTP_LENGTH,
        placeholder="• • • • • •",
        help=f"6-digit code · expires in {OTP_VALID_SECONDS} seconds",
    )

    verify_col, resend_col = st.columns(2)
    with verify_col:
        verify_clicked = st.button(
            "Verify & Login",
            type="primary",
            use_container_width=True,
            disabled=is_otp_expired(email),
        )
    with resend_col:
        can_resend, wait_secs = _can_resend(email)
        resend_label = "Generate New OTP" if is_otp_expired(email) else "Resend OTP"
        resend_clicked = st.button(
            resend_label,
            use_container_width=True,
            disabled=not can_resend and not is_otp_expired(email),
            help=f"Wait {wait_secs}s between resends" if not can_resend else None,
        )

    if resend_clicked:
        ok, msg = request_otp(email)
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

    if verify_clicked:
        if is_otp_expired(email):
            st.error("OTP expired. Click **Generate New OTP** to receive a fresh code.")
        else:
            ok, msg = verify_otp(email, otp)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
                if "expired" in msg.lower() or "attempts" in msg.lower():
                    st.session_state.otp_sent = False
                    st.rerun()


def render_top_session_bar() -> None:
    """Top-right session info and logout (main dashboard header)."""
    if not is_authenticated():
        return
    user = st.session_state.auth_user
    remaining = session_remaining_display()
    admin_badge = " · Admin" if is_admin(user) else ""

    st.markdown(
        f"""
        <div style="text-align:right; font-size:0.85rem; line-height:1.5;">
            <div><strong>{user}</strong>{admin_badge}</div>
            <div style="color:#666;">Session: {remaining} left</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Log out", key="top_logout", use_container_width=True):
        logout()
        st.rerun()


def require_login() -> None:
    """Show login page and stop the app if the user is not authenticated."""
    if is_authenticated():
        return
    _render_login_page()
    st.stop()


def current_user() -> str:
    return st.session_state.get("auth_user", "")
