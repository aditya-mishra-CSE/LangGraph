
from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ================= LLM =================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ================= Thread Stores =================
_THREAD_RETRIEVERS: Dict[str, any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

# ================= PDF INGEST =================
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        path = f.name

    try:
        docs = PyPDFLoader(path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        ).split_documents(docs)

        vs = FAISS.from_documents(chunks, embeddings)
        _THREAD_RETRIEVERS[thread_id] = vs.as_retriever(k=4)

        _THREAD_METADATA[thread_id] = {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
        }
        return _THREAD_METADATA[thread_id]
    finally:
        os.remove(path)

# ================= TOOL =================
@tool
def rag_tool(query: str, thread_id: str) -> dict:
    """
    Retrieve information from the uploaded PDF for this chat thread.
    """
    retriever = _THREAD_RETRIEVERS.get(thread_id)
    if not retriever:
        return {"context": []}

    docs = retriever.invoke(query)
    return {"context": [d.page_content for d in docs]}

tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)

# ================= STATE =================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ================= NODE =================
def chat_node(state: ChatState, config=None):
    thread_id = config["configurable"]["thread_id"]

    system = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "If the question is about the uploaded PDF, use rag_tool with the thread_id. "
            "Otherwise answer normally."
        )
    )

    response = llm_with_tools.invoke(
        [system, *state["messages"]],
        config=config,
    )
    return {"messages": [response]}

# ================= GRAPH =================
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

chatbot = graph.compile(checkpointer=checkpointer)

# ================= HELPERS =================
def retrieve_all_threads():
    return list({
        c.config["configurable"]["thread_id"]
        for c in checkpointer.list(None)
    })

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(thread_id, {})
