import os
import sys

# Ensure the app root is on sys.path so config can be imported from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load config FIRST so OAUTHLIB_INSECURE_TRANSPORT is set before OAuth imports
from config import REDIRECT_URI, CLIENT_SECRET_PATH, DEBUG_MODE, logger

# Must be set before importing google_auth_oauthlib
if DEBUG_MODE:
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

import streamlit as st
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests

st.set_page_config(page_title="RAG Streamlit App", layout="centered")

# Session state for login
if "user_logged_in" not in st.session_state:
    st.session_state.user_logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None

query_params = st.query_params


def create_flow() -> Flow:
    """
    Create a Flow object from client_secret.json.
    Uses the absolute path from config so it works regardless of cwd.
    PKCE is disabled (autogenerate_code_verifier=False) because Streamlit
    re-runs the entire script on the OAuth callback, creating a new Flow
    that loses the original code_verifier.
    """
    return Flow.from_client_secrets_file(
        CLIENT_SECRET_PATH,
        scopes=[
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "openid",
        ],
        redirect_uri=REDIRECT_URI,
        autogenerate_code_verifier=False,
    )


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
            flow = create_flow()
            flow.fetch_token(code=code)
            credentials = flow.credentials

            req = requests.Request()
            client_config = flow.client_config
            client_id = client_config.get("client_id")

            id_info = id_token.verify_oauth2_token(
                credentials.id_token,
                req,
                client_id,
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
        # Google OAuth Login — only show button when not handling callback
        flow = create_flow()
        auth_url, _ = flow.authorization_url(
            prompt="consent",
            access_type="offline",
        )

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