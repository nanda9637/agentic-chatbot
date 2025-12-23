
import os
import sys
from groq import Groq
from dotenv import load_dotenv
from tavily import TavilyClient

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from prompt import LLM_PROMPT


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_key = Groq(api_key=GROQ_API_KEY)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_key = TavilyClient(api_key=TAVILY_API_KEY)


def get_llm_response(query: str, history=None) -> str:
    """
    Generate an answer using Groq LLM for explanatory queries.
    """
    messages = [{"role": "system", "content": LLM_PROMPT}]
    
    # Add conversation history if available
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": query})

    LLM_response = groq_key.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.4,
        max_tokens=800
    )
    return LLM_response.choices[0].message.content.strip()


def get_search_response(query: str) -> dict:
    """
    Fetch fresh info using Tavily search
    """
    search_response = tavily_key.qna_search(
        query=query,
        max_results=2,      
    )
    return search_response.strip()
