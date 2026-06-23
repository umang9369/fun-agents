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

# now we create a function of three tasks that will generate a poem and joke and a story 

@task
def call_llm_1(topic: str):
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a joke about {topic}")
    return msg.content


@task
def call_llm_2(topic: str):
    """Second LLM call to generate story"""
    msg = llm.invoke(f"Write a story about {topic}")
    return msg.content


@task
def call_llm_3(topic):
    """Third LLM call to generate poem"""
    msg = llm.invoke(f"Write a poem about {topic}")
    return msg.content

#this is the aggregator function that will combine the joke and story into a single output

@task
def aggregator(topic, joke, story, poem):
    """Combine the joke and story into a single output"""

    combined = f"Here's a story, joke, and poem about {topic}!\n\n"
    combined += f"STORY:\n{story}\n\n"
    combined += f"JOKE:\n{joke}\n\n"
    combined += f"POEM:\n{poem}"
    return combined

#build workflow

@entrypoint()
def parallelization_workflow(topic: str):
    joke_fut = call_llm_1(topic)
    story_fut = call_llm_2(topic)
    poem_fut= call_llm_3(topic)
    return aggregator(topic, joke_fut.result(), story_fut.result(), poem_fut.result()).result()


#invoking the function

topic = input("Enter a topic for a joke: ")
# Invoke
stream = parallelization_workflow.stream_events(topic, version="v3")
for snapshot in stream.values:
    print(snapshot)
