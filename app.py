"""
app.py - Chat UI for the local Ollama agent.

Run with:
    streamlit run app.py

Features
--------
- Real chat interface (multi-turn, remembers the whole conversation)
- Upload a PDF in the sidebar; its text is extracted and given to the model
  as context, so you can ask questions about the document
- Every assistant reply has download buttons (.txt and .docx) in case you
  want the answer as a file instead of just reading it in chat
- "New conversation" button to clear history / detach the PDF
"""

import io
import streamlit as st
from pypdf import PdfReader
from docx import Document

from agent import new_conversation, run_agent_step, MODEL

MAX_PDF_CHARS = 15000  # keep the prompt size sane for a small local model

st.set_page_config(page_title="Local AI Agent", page_icon="🤖", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def make_docx_bytes(text: str) -> bytes:
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = new_conversation()
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# --------------------------------------------------------------------------- #
# Sidebar — PDF upload + controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("📄 Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded is not None and uploaded.name != st.session_state.pdf_name:
        with st.spinner("Reading PDF..."):
            text = extract_pdf_text(uploaded)
        if not text:
            st.warning("Couldn't extract any text from this PDF (it may be scanned/image-only).")
        else:
            if len(text) > MAX_PDF_CHARS:
                text = text[:MAX_PDF_CHARS] + "\n\n[...truncated...]"
            st.session_state.messages.append(
                {
                    "role": "system",
                    "content": f"The user uploaded a PDF named '{uploaded.name}'. "
                    f"Extracted text follows:\n\n{text}",
                }
            )
            st.session_state.pdf_name = uploaded.name
            st.success(f"Loaded {uploaded.name} ({len(text):,} chars)")
    elif st.session_state.pdf_name:
        st.caption(f"📎 Active document: **{st.session_state.pdf_name}**")

    st.divider()
    st.caption(f"Model: `{MODEL}`")
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.messages = new_conversation()
        st.session_state.pdf_name = None
        st.rerun()

# --------------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------------- #
st.title("🤖 Local AI Agent")
st.caption("Chat with your local Ollama model. Upload a PDF in the sidebar to ask questions about it.")

# Render history (skip system messages — those are PDF context / instructions)
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] not in ("user", "assistant") or not msg.get("content"):
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            col1, col2 = st.columns([1, 1])
            with col1:
                st.download_button(
                    "⬇️ Save as .txt",
                    msg["content"],
                    file_name=f"response_{i}.txt",
                    key=f"txt_{i}",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    "⬇️ Save as .docx",
                    make_docx_bytes(msg["content"]),
                    file_name=f"response_{i}.docx",
                    key=f"docx_{i}",
                    use_container_width=True,
                )

# --------------------------------------------------------------------------- #
# Chat input
# --------------------------------------------------------------------------- #
user_input = st.chat_input("Ask something, or ask about the uploaded PDF...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, st.session_state.messages = run_agent_step(st.session_state.messages)
            except Exception as e:
                reply = f"⚠️ Error talking to Ollama: {e}\n\nMake sure `ollama serve` is running and the model is pulled."
                st.session_state.messages.append({"role": "assistant", "content": reply})
        st.markdown(reply)

    st.rerun()
