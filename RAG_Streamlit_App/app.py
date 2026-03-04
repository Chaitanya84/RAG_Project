import os
import sys

# Ensure the app root is on sys.path so config can be imported from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load config FIRST so OAUTHLIB_INSECURE_TRANSPORT is set before OAuth imports
from config import REDIRECT_URI, CLIENT_SECRET_PATH, DEBUG_MODE, logger

# Must be set before importing google_auth_oauthlib
if DEBUG_MODE:
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

import json
import urllib.parse
import streamlit as st
import requests as http_requests
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

st.set_page_config(page_title="RAG Streamlit App", layout="centered")

# Session state for login
if "user_logged_in" not in st.session_state:
    st.session_state.user_logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None

query_params = st.query_params

# Load client secrets once
with open(CLIENT_SECRET_PATH, "r") as _f:
    _client_secrets = json.load(_f)["web"]

CLIENT_ID = _client_secrets["client_id"]
CLIENT_SECRET = _client_secrets["client_secret"]
AUTH_URI = _client_secrets["auth_uri"]
TOKEN_URI = _client_secrets["token_uri"]

SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid",
]


def build_auth_url() -> str:
    """Build Google OAuth2 authorization URL manually (no PKCE)."""
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{AUTH_URI}?{urllib.parse.urlencode(params)}"


def exchange_code_for_tokens(code: str) -> dict:
    """Exchange authorization code for tokens via direct HTTP POST (no PKCE)."""
    data = {
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    resp = http_requests.post(TOKEN_URI, data=data)
    if resp.status_code != 200:
        raise Exception(f"Token exchange failed: {resp.status_code} {resp.text}")
    return resp.json()


st.title("Welcome App")

if st.session_state.user_logged_in:
    st.success(f"Logged in as: {st.session_state.user_email}")
    st.write("Use the sidebar to go to **MainPage** to process a PDF directory.")
    if st.button("Logout"):
        st.session_state.user_logged_in = False
        st.session_state.user_email = None
        st.rerun()
else:
    st.write("Please log in using your Google account to continue.")

    # Handle OAuth callback via query params FIRST (before rendering login button)
    if "code" in query_params:
        code = query_params["code"]

        try:
            tokens = exchange_code_for_tokens(code)

            id_info = id_token.verify_oauth2_token(
                tokens["id_token"],
                google_requests.Request(),
                CLIENT_ID,
            )

            st.session_state.user_logged_in = True
            st.session_state.user_email = id_info.get("email", "")
            logger.info("User logged in: %s", st.session_state.user_email)

            # Clear query params and rerun
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            logger.error("OAuth login failed: %s", e)
            st.error(f"Login failed: {e}")
            st.query_params.clear()
    else:
        # Google OAuth Login
        auth_url = build_auth_url()

        st.markdown(
            f"""
            <a href="{auth_url}" style="text-decoration: none;">
                <button style="
                    background-color: #4285F4;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    border-radius: 4px;
                    cursor: pointer;
                ">
                    Login with Google
                </button>
            </a>
            """,
            unsafe_allow_html=True,
        )