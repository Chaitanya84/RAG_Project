import os
import sys

# Ensure the app root is on sys.path so config can be imported from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests

from config import REDIRECT_URI, CLIENT_SECRET_PATH, DEBUG_MODE, logger

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
    """
    return Flow.from_client_secrets_file(
        CLIENT_SECRET_PATH,
        scopes=[
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "openid",
        ],
        redirect_uri=REDIRECT_URI,
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

    # Google OAuth Login
    flow = create_flow()
    auth_url, _ = flow.authorization_url(prompt="consent")

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

    # Handle OAuth callback via query params
    if "code" in query_params:
        code_values = query_params["code"]
        code = code_values[0] if isinstance(code_values, list) else code_values

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
        except Exception as e:
            logger.error("OAuth login failed: %s", e)
            st.error(f"Login failed: {e}")

        # Remove "code" from query params and re-run to avoid re-processing
        new_qp = {k: v for k, v in query_params.items() if k != "code"}
        st.query_params = new_qp

        st.rerun()