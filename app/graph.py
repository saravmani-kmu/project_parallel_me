import os
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import operator

# Define State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    model_provider: str

# Initialize Tavily search tool
tavily_tool = TavilySearchResults(
    max_results=5,
    api_key=os.getenv("TAVILY_API_KEY")
)

tools = [tavily_tool]

# Define Nodes
def call_llm(state: State):
    messages = state["messages"]
    provider = state.get("model_provider", "openai")
    
    if provider == "openai":
        llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    elif provider == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    elif provider == "groq":
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Conditional routing function
def should_continue(state: State) -> Literal["tools", "end"]:
    """Determine whether to continue with tool calls or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, route to the "tools" node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, end the workflow
    return "end"

# Build Graph
builder = StateGraph(State)

# Add nodes
builder.add_node("llm", call_llm)
builder.add_node("tools", ToolNode(tools))

# Set entry point
builder.set_entry_point("llm")

# Add conditional edges
builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# After tools are called, go back to the LLM
builder.add_edge("tools", "llm")

# Compile Graph
graph = builder.compile()

