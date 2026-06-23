import os
from pathlib import Path
from unittest import result
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph import graph
from pydantic import BaseModel, Field
from langgraph.func import entrypoint, task
import warnings
from langgraph.graph import StateGraph, START, END

warnings.filterwarnings("ignore")

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



# Tasks
@task
def generate_joke(topic: str):
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a short joke about {topic}")
    return msg.content


def check_punchline(joke: str):
    """Gate function to check if the joke has a punchline"""
    # Simple check - does the joke contain "?" or "!"
    if "?" in joke or "!" in joke:
        return "Fail"

    return "Pass"


@task
def improve_joke(joke: str):
    """Second LLM call to improve the joke"""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {joke}")
    return msg.content


@task
def polish_joke(joke: str):
    """Third LLM call for final polish"""
    msg = llm.invoke(f"Add a surprising twist to this joke: {joke}")
    return msg.content


@entrypoint()
def prompt_chaining_workflow(topic: str):
    original_joke = generate_joke(topic).result()
    if check_punchline(original_joke) == "Pass":
        return original_joke

    improved_joke = improve_joke(original_joke).result()
    return polish_joke(improved_joke).result()
topic = input("Enter a topic for a joke: ")
# Invoke
stream = prompt_chaining_workflow.stream_events(topic, version="v3")
for snapshot in stream.values:
    print(snapshot)
    print("\n")

builder = StateGraph(dict)

def joke_node(state):
    return {"output": "This is a joke"}

builder.add_node("joke", joke_node)
builder.add_edge(START, "joke")
builder.add_edge("joke", END)

graph = builder.compile() 

# result
user_input = input("Enter a topic: ")
result = graph.invoke({"input": user_input})
print(result)
