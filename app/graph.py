import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage
import operator

# Define State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    model_provider: str

# Define Nodes
def call_llm(state: State):
    messages = state["messages"]
    provider = state.get("model_provider", "openai")
    
    if provider == "openai":
        llm = ChatOpenAI(model="gpt-3.5-turbos", api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    response = llm.invoke(messages)
    return {"messages": [response]}

# Build Graph
builder = StateGraph(State)
builder.add_node("llm", call_llm)
builder.set_entry_point("llm")
builder.add_edge("llm", END)

# Compile Graph
graph = builder.compile()
