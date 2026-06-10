"""
Home page for Streamlit authentication interface.
"""

import logging
import uuid

import streamlit as st

st.set_page_config(page_title="LangGraph Chat")

# Hide sidebar nav for cleaner look
hide_sidebar_style = """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="a",
)
logger = logging.getLogger(__name__)

# Redirect already-named users straight to chat
if "session_id" in st.session_state:
    st.switch_page("pages/Chat.py")

st.title("👋 Welcome to LangGraph Assistant")
st.write("An adaptive RAG system that retrieves and reasons over your documents.")

with st.form("name_form"):
    name = st.text_input("What's your name?", placeholder="E.g. Alex")
    submit = st.form_submit_button("Start Chatting →")

if submit:
    if not name.strip():
        st.error("Please enter your name to continue.")
    else:
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["username"] = name.strip()
        logger.info(
            "User '%s' started session: %s",
            name.strip(),
            st.session_state["session_id"],
        )
        st.switch_page("pages/Chat.py")