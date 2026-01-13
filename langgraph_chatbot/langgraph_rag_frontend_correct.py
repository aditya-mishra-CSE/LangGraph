
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage

from langGraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# ================= FIX FOR GEMINI OUTPUT =================
def extract_ai_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)

# ================= SESSION INIT =================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

def new_chat():
    tid = str(uuid.uuid4())
    st.session_state.thread_id = tid
    st.session_state.message_history = []
    if tid not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(tid)

# ================= SIDEBAR =================
st.sidebar.title("📄 LangGraph PDF Chatbot")
st.sidebar.button("➕ New Chat", on_click=new_chat)

uploaded = st.sidebar.file_uploader("Upload PDF", type="pdf")
if uploaded:
    ingest_pdf(uploaded.getvalue(), st.session_state.thread_id, uploaded.name)
    st.sidebar.success("PDF Indexed")

st.sidebar.subheader("Past Chats")
for tid in st.session_state.chat_threads[::-1]:
    if st.sidebar.button(tid):
        st.session_state.thread_id = tid
        state = chatbot.get_state(
            config={"configurable": {"thread_id": tid}}
        )
        st.session_state.message_history = [
            {
                "role": "user" if m.type == "human" else "assistant",
                "content": extract_ai_text(m.content),
            }
            for m in state.values.get("messages", [])
        ]
        st.rerun()

# ================= MAIN =================
st.title("🤖 Multi-Utility Chatbot")

for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something…")

if user_input:
    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": st.session_state.thread_id}},
    )

    raw = response["messages"][-1].content
    ai_text = extract_ai_text(raw)

    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_text}
    )

    with st.chat_message("assistant"):
        st.markdown(ai_text)

    meta = thread_document_metadata(st.session_state.thread_id)
    if meta:
        st.caption(
            f"📄 {meta['filename']} | "
            f"{meta['chunks']} chunks | "
            f"{meta['documents']} pages"
        )
