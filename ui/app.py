import os
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))

load_dotenv(env_path)


def _resolve_backend_url() -> str:
    """Resolve and sanitize backend base URL from env/config."""
    raw_value = os.getenv("BACKEND_URL", "").strip()

    # Render env vars are sometimes pasted with quotes/backticks.
    cleaned = raw_value.strip("\"'`").rstrip("/")

    if cleaned:
        return cleaned

    # Local fallback only when BACKEND_URL is absent.
    return "http://localhost:8000"


def _resolve_logfire_token() -> str:
    """Read token from env and clean accidental wrapping quotes."""
    raw_value = os.getenv("LOGFIRE_TOKEN", "").strip()
    return raw_value.strip("\"'`")


def _get_backend_auth_headers(audience: str) -> dict:
    """Attach a Google-signed ID token so Cloud Run's IAM invoker check
    (see terraform/cloud_run.tf: backend_invoker) lets this call through.

    Only works when running with Google credentials available (e.g. the
    UI's Cloud Run service account, or `gcloud auth application-default
    login` locally). Falls back to no auth header for local dev against
    an unauthenticated local backend.
    """
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token

        token = google.oauth2.id_token.fetch_id_token(
            google.auth.transport.requests.Request(), audience
        )
        return {"Authorization": f"Bearer {token}"}
    except Exception:
        return {}


def _call_backend_query(base_url: str, payload: dict) -> dict:
    """Call backend /query and return parsed JSON with better error messages."""
    url = f"{base_url}/query"

    # Render-hosted backends are usually public; signed ID tokens are needed
    # mostly for Cloud Run protected invoker paths.
    use_auth = ".run.app" in base_url
    headers = _get_backend_auth_headers(base_url) if use_auth else {}

    response = requests.post(url, json=payload, headers=headers, timeout=60)

    # If token-based auth was attempted and denied, retry once without token.
    if response.status_code in (401, 403) and headers:
        response = requests.post(url, json=payload, headers={}, timeout=60)

    if response.status_code >= 400:
        body_preview = response.text[:400].replace("\n", " ")
        raise RuntimeError(
            f"Backend HTTP {response.status_code} from {url}: {body_preview}"
        )

    try:
        return response.json()
    except Exception as json_error:
        body_preview = response.text[:400].replace("\n", " ")
        raise RuntimeError(
            f"Invalid JSON from backend {url}: {body_preview}"
        ) from json_error

try:
    token = _resolve_logfire_token()
    if token:
        logfire.configure(token=token)
        LOGFIRE_STATUS = "Connected & Tracing"
    else:
        # Keep the app fully functional even when observability is disabled.
        logfire.configure(send_to_logfire=False)
        LOGFIRE_STATUS = "Disabled (no token configured)"
except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    logfire.configure(send_to_logfire=False)
    LOGFIRE_STATUS = "Disabled (token/auth invalid)"



st.set_page_config(
    page_title="StackSage",
    page_icon="🧙",
    layout="wide",
    initial_sidebar_state="expanded",
)
    
AI_AVATAR = "🤖"
USER_AVATAR = "👤"
BACKEND_BASE_URL = _resolve_backend_url()

# --- SESSION MANAGEMENT ---

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info("New session created", session_id=st.session_state.session_id)

if 'messages' not in st.session_state:
    st.session_state.messages = []   


# --- SIDEBAR ---

with st.sidebar:
    st.title("StackSage")
    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.caption(f"Backend URL: {BACKEND_BASE_URL}")
    st.info(f'Memory ID: {st.session_state.session_id[:8]}')

    if st.button("Clear History"):
        logfire.warn(f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}")
        st.session_state.messages = []
        st.rerun() 

# --- MAIN CHAT ---
st.title("🧙 StackSage — Your Web Dev Assistant")

# Custom CSS styling for premium look & feel
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Sidebar styled element */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Render existing chat messages
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input

if prompt := st.chat_input("Ask about HTML, CSS, JavaScript, React, Node.js, MongoDB, Docker, or Kubernetes..."):

# START TRACE: User Interaction
    
    with logfire.span("💬 User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.status("Thinking...", expanded=True) as status:
                data = None

                try:

                    with logfire.span("📡 Calling RAG Backend"):
                        base_url = BACKEND_BASE_URL
                        payload = {"query": prompt, "thread_id": st.session_state.session_id}
                        data = _call_backend_query(base_url, payload)

                    # show reasoning steps from backend

                    steps = data.get('thought_process',[])
                    for step in steps:
                        st.write(f"⚙️ {step}")

                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)

                    # --- SHOW SOURCES (NESTED EXPANDABLES) ---

                    sources = data.get('sources',[])

                    if sources:
                        with st.expander("📚 Sources"):
                            for i, src in enumerate(sources):
                                if isinstance(src, dict):
                                    source_name = src.get("source", "unknown")
                                    score = src.get("score")
                                    content = src.get("content", "")
                                    score_label = f" · score {score:.2f}" if isinstance(score, (int, float)) else ""
                                    preview = content[:100].replace("\n", " ") + "..."
                                    with st.expander(f"[{i+1}] {source_name}{score_label} — {preview}"):
                                        st.caption(f"Source: {source_name}")
                                        st.info(content)
                                else:
                                    # Backward-compat: older API responses returned plain strings
                                    preview = str(src)[:100].replace("\n", " ") + "..."
                                    with st.expander(f"chunk {i+1}: {preview}"):
                                        st.info(src)
                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Backend Connection Failed")
                    st.caption(f"Target backend: {BACKEND_BASE_URL}")
                    st.code(str(e))
                    status.update(label="❌ Request failed", state="error", expanded=True)


        # Final placeholder                    

        if data is not None:
            answer_placeholder = st.empty()
            full_answer = data.get("answer", "No Response")
            print('answer', full_answer)
            curr_text=''

            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text)
                time.sleep(0.02)
            answer_placeholder.markdown(full_answer)
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("✅ Answer displayed to user", 
                          user_query=prompt, 
                          session_id=st.session_state.session_id,
                          answer=full_answer[:500])

                                    


                
                        
                
            
            
        



