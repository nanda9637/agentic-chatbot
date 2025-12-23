
# src/graph.py
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END

from router import route_query
from agents import get_llm_response
from agents import get_search_response

# --- Shared State ---
class ChatState(TypedDict, total=False):
    user_query: str
    route: Literal["LLM_ONLY", "SEARCH", "Please try again later"]
    final_answer: Optional[str]

# --- Nodes ---
def router_node(state: ChatState) -> ChatState:
    decision = route_query(state["user_query"])
    state["route"] = decision
    return state

def llm_agent_node(state: ChatState) -> ChatState:
    state["final_answer"] = get_llm_response(state["user_query"])
    return state

def tavily_agent_node(state: ChatState) -> ChatState:
    state["final_answer"] = get_search_response(state["user_query"])
    return state

# --- Graph Setup ---
graph = StateGraph(ChatState)
graph.add_node("router", router_node)
graph.add_node("llm_agent", llm_agent_node)
graph.add_node("tavily_agent", tavily_agent_node)

# Conditional routing
def route_edge(state: ChatState) -> str:
    if state["route"] == "LLM_ONLY":
        return "llm_agent"
    elif state["route"] == "SEARCH":
        return "tavily_agent"
    else:
        return "llm_agent"

graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_edge, {
    "llm_agent": "llm_agent",
    "tavily_agent": "tavily_agent"
})
graph.add_edge("llm_agent", END)
graph.add_edge("tavily_agent", END)

# Compile graph
app = graph.compile()

# --- Example Invocation ---
if __name__ == "__main__":
    query = input("Enter your question: ")
    final_state = app.invoke({"user_query": query})

    print("\n=== Routing Decision ===")
    print(final_state["route"])
    print("\n=== User Query ===")
    print(final_state["user_query"])
    print("\n=== Final Answer ===")
    print(final_state["final_answer"])
