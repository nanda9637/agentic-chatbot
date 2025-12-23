import os
import streamlit as st
from typing import TypedDict, List, Literal, Optional

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient
from langgraph.graph import StateGraph, START, END
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MAX_MEMORY_TURNS = 6   # 👈 short-term memory window
MAX_REVISIONS = 1

# --------------------------------------------------
# 2. Prompts
# --------------------------------------------------
ROUTER_PROMPT = """
You are a routing agent.
Classify the user query into one category only:
LLM, SEARCH, or REJECT.

Return ONLY one word.
"""

LLM_SYSTEM_PROMPT = """
You are a helpful, professional AI assistant.
Be concise, accurate, and safe.
Use markdown when helpful.
"""

# --------------------------------------------------
# 3. State Definition
# --------------------------------------------------
class ChatState(TypedDict):
    user_query: str
    history: List[dict]

    route: str
    plan: Optional[str]

    execution_result: Optional[str]
    final_answer: Optional[str]

    needs_revision: bool
    revision_count: int


# --------------------------------------------------
# 4. LLM Helper
# --------------------------------------------------
def call_llm(messages, model="llama-3.3-70b-versatile", temperature=0.3):
    response = groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


# --------------------------------------------------
# 5. Agent Nodes
# --------------------------------------------------
def router_node(state: ChatState) -> ChatState:
    decision = call_llm(
        [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": state["user_query"]},
        ],
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    state["route"] = decision if decision in {"LLM", "SEARCH", "REJECT"} else "REJECT"
    return state


def planner_node(state: ChatState) -> ChatState:
    state["plan"] = call_llm(
        [
            {"role": "system", "content": "Create a brief execution plan."},
            {"role": "user", "content": state["user_query"]},
        ],
        model="llama-3.1-8b-instant",
        temperature=0,
    )
    return state


def executor_node(state: ChatState) -> ChatState:
    # 👇 SHORT-TERM MEMORY WINDOW
    recent_history = state["history"][-MAX_MEMORY_TURNS:]

    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": state["user_query"]})

    if state["route"] == "SEARCH":
        state["execution_result"] = tavily_client.qna_search(
            query=state["user_query"]
        )
    else:
        state["execution_result"] = call_llm(messages)

    return state


def critic_node(state: ChatState) -> ChatState:
    verdict = call_llm(
        [
            {"role": "system", "content": "Reply ONLY with OK or REVISE."},
            {"role": "user", "content": state["execution_result"]},
        ],
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    state["needs_revision"] = verdict.startswith("REVISE")
    state["revision_count"] += 1

    if not state["needs_revision"] or state["revision_count"] > MAX_REVISIONS:
        state["final_answer"] = state["execution_result"]

    return state


def reject_node(state: ChatState) -> ChatState:
    state["final_answer"] = "This request cannot be processed currently."
    return state


# --------------------------------------------------
# 6. Graph Construction
# --------------------------------------------------
workflow = StateGraph(ChatState)

workflow.add_node("router", router_node)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("critic", critic_node)
workflow.add_node("reject", reject_node)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    lambda s: "reject" if s["route"] == "REJECT" else "planner",
)

workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "critic")

workflow.add_conditional_edges(
    "critic",
    lambda s: "executor"
    if s["needs_revision"] and s["revision_count"] <= MAX_REVISIONS
    else END,
)

workflow.add_edge("reject", END)

app = workflow.compile()

# --------------------------------------------------
# 7. Streamlit UI
# --------------------------------------------------
def main():
    st.set_page_config(page_title="Agentic Chatbot", page_icon="🧠")
    st.title("🧠 Agentic Chatbot (Short-Term Memory)")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something..."):
        st.chat_message("user").markdown(prompt)

        state = {
            "user_query": prompt,
            "history": st.session_state.history,
            "route": "",
            "plan": None,
            "execution_result": None,
            "final_answer": None,
            "needs_revision": False,
            "revision_count": 0,
        }

        with st.spinner("Thinking..."):
            result = app.invoke(state)

        answer = result["final_answer"]

        with st.chat_message("assistant"):
            st.markdown(answer)

        # Persist short-term memory
        st.session_state.history.append(
            {"role": "user", "content": prompt}
        )
        st.session_state.history.append(
            {"role": "assistant", "content": answer}
        )

        # Optional: hard cap history size
        st.session_state.history = st.session_state.history[-20:]


if __name__ == "__main__":
    main()
