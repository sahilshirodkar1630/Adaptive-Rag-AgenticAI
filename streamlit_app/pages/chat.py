"""
Chat page for the Streamlit application.
"""

import streamlit as st

from utils.api_client import query_backend, document_upload_rag

st.set_page_config(
    page_title="LangGraph Chat",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": None,
    },
)
hide_sidebar_style = """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# Auth guard — redirect if no session
if "session_id" not in st.session_state:
    st.switch_page("Home.py")

# Initialize logout confirmation state
if "show_logout_confirm" not in st.session_state:
    st.session_state.show_logout_confirm = False

# Header
col1, col2 = st.columns([10, 2])
with col1:
    st.title(f"💬 Hey {st.session_state['username']}, ask me anything!")
with col2:
    st.write("")
    if st.button("🚪 Start Over", use_container_width=True):
        st.session_state.show_logout_confirm = True

# Start Over confirmation
if st.session_state.show_logout_confirm:
    st.warning("This will clear your session and chat history. Continue?")
    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        if st.button("✅ Yes, start over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.switch_page("Home.py")
    with col_cancel:
        if st.button("❌ Cancel"):
            st.session_state.show_logout_confirm = False

# Document upload section
with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file:
        file_description = st.text_input(
            "📄 Describe your document (required)",
            max_chars=300,
            placeholder="E.g. LangGraph tutorial with workflows and code examples",
        )

        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = {}

        file_key = f"{uploaded_file.name}_{file_description}"

        if file_description:
            if file_key not in st.session_state.uploaded_files:
                success = document_upload_rag(uploaded_file, file_description)
                if success:
                    st.success(f"Uploaded: {uploaded_file.name}")
                    st.session_state.uploaded_files[file_key] = True
                else:
                    st.error(f"Upload failed: {uploaded_file.name}")
            else:
                st.info(f"Already uploaded: {uploaded_file.name}")
        else:
            st.warning("Please describe your document before uploading.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    response = query_backend(user_input, st.session_state["session_id"])
    st.session_state.chat_history.append(("assistant", response))
    st.rerun()

# Display chat history
for role, text in st.session_state.chat_history:
    st.chat_message(role).write(text)