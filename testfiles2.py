import os
import uuid
import requests
import streamlit as st
from typing import TypedDict, List, Optional
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZE = 500
TOP_K = 3

# --------------------------------------------------
# 2. Prompts
# --------------------------------------------------
SYSTEM_PROMPT = """
You are a helpful AI assistant.
"""

RAG_SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer strictly using ONLY the provided context.
If the answer is not present, say "Not found in the document."
"""

RAG_PROMPT = """
Context:
{context}

Question:
{question}
"""

ROUTER_PROMPT = """
Classify the user query into one category only:
LLM or REJECT.
Return ONLY one word.
"""

# --------------------------------------------------
# 3. State
# --------------------------------------------------
class ChatState(TypedDict):
    user_query: str
    history: List[dict]
    use_rag: bool
    final_answer: Optional[str]

# --------------------------------------------------
# 4. Helper Functions
# --------------------------------------------------
def call_llm(messages, temperature=0.3) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=temperature,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


def scrape_url(url: str) -> str:
    html = requests.get(url, timeout=10).text
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(p.get_text() for p in soup.find_all("p"))


def chunk_text(text: str):
    words = text.split()
    for i in range(0, len(words), CHUNK_SIZE):
        yield " ".join(words[i:i + CHUNK_SIZE])


def clear_index():
    index.delete(delete_all=True)


def ingest_url(url: str):
    clear_index()
    text = scrape_url(url)

    vectors = []
    for chunk in chunk_text(text):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedder.encode(chunk).tolist(),
            "metadata": {"text": chunk},
        })

    if vectors:
        index.upsert(vectors=vectors)


def retrieve_context(query: str) -> str:
    query_emb = embedder.encode(query).tolist()
    res = index.query(
        vector=query_emb,
        top_k=TOP_K,
        include_metadata=True,
    )

    return "\n".join(m["metadata"]["text"] for m in res["matches"])

# --------------------------------------------------
# 5. Agent Nodes
# --------------------------------------------------
def rag_node(state: ChatState) -> ChatState:
    context = retrieve_context(state["user_query"])

    if not context.strip():
        state["final_answer"] = "Not found in the document."
        return state

    answer = call_llm([
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": RAG_PROMPT.format(
            context=context,
            question=state["user_query"]
        )}
    ], temperature=0)

    state["final_answer"] = answer
    return state


def router_node(state: ChatState) -> ChatState:
    decision = call_llm(
        [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": state["user_query"]},
        ],
        temperature=0,
    )

    if decision.strip() == "REJECT":
        state["final_answer"] = "This request cannot be processed."
    else:
        state["final_answer"] = call_llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            *state["history"],
            {"role": "user", "content": state["user_query"]},
        ])

    return state

# --------------------------------------------------
# 6. Graph
# --------------------------------------------------
workflow = StateGraph(ChatState)

workflow.add_node("rag", rag_node)
workflow.add_node("router", router_node)

workflow.add_conditional_edges(
    START,
    lambda s: "rag" if s["use_rag"] else "router"
)

workflow.add_edge("rag", END)
workflow.add_edge("router", END)

app = workflow.compile()

# --------------------------------------------------
# 7. Streamlit UI
# --------------------------------------------------
def main():
    st.set_page_config("Hybrid RAG Chatbot", "🧠")
    st.title("🧠 Hybrid Chatbot (RAG + Normal)")

    if "history" not in st.session_state:
        st.session_state.history = []

    if "last_url" not in st.session_state:
        st.session_state.last_url = None

    use_rag = st.checkbox("🔍 Answer only from URL (RAG)")

    url = None
    if use_rag:
        url = st.text_input(
            "Enter webpage URL",
            placeholder="https://example.com"
        )

        if url and url != st.session_state.last_url:
            with st.spinner("Indexing new document..."):
                ingest_url(url)
                st.session_state.last_url = url
            st.success("Document indexed.")

    # Display history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something..."):
        st.chat_message("user").markdown(prompt)

        state: ChatState = {
            "user_query": prompt,
            "history": st.session_state.history[-6:],
            "use_rag": use_rag,
            "final_answer": None,
        }

        with st.spinner("Thinking..."):
            result = app.invoke(state)

        answer = result["final_answer"]

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ])

        st.session_state.history = st.session_state.history[-20:]

if __name__ == "__main__":
    main()
