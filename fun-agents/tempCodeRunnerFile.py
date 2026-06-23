import os
from pathlib import Path
from unittest import result
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )

structured_llm = llm.with_structured_output(SearchQuery)
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")
print(output)
def multiply(a: int, b: int) -> int:
    return a * b
print(multiply(int(input("Enter first number: ")), int(input("Enter second number: "))))
llm_with_tools = llm.bind_tools([multiply])
msg = llm_with_tools.invoke("What is 2 times 3?")
print(msg.tool_calls)
tool_call = msg.tool_calls[0]
result = multiply(
    tool_call["args"]["a"],
    tool_call["args"]["b"]
)

print(result)


