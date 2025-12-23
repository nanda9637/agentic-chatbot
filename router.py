
import os
import sys
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from prompt import ROUTER_PROMPT

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


def route_query(user_query: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": user_query.strip()}
        ],
        temperature=0
    )
        
    decision = response.choices[0].message.content.strip().upper()
    if decision not in ["LLM", "SEARCH", "Please try again later"]:
        decision = "Please try again later"

    decision = response.choices[0].message.content.strip()
    #print(ROUTER_PROMPT.format(query=user_query))
    return decision


# query = "What is the current pope of France?"
# route = route_query(query)
# print(route)
