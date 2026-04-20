


from dotenv import load_dotenv
load_dotenv()
import os

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGSMITH_TRACING_V2']="true"
os.environ['LANGSMITH_PROJECT']="langsmithlearning"
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"


 

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage
import os


import os
from langchain.chat_models import init_chat_model
llm=init_chat_model("groq:llama-3.3-70b-versatile")
# llm

class State(TypedDict):
    messages:Annotated[list[BaseMessage], add_messages]

## Stategraph
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

def make_tool_graph():
    ## create a graph with tool call

    @tool
    def add(a:float, b:float) -> float:
        """Add two numbers"""
        return a + b

    tools = [add]
    tool_node = ToolNode(tools=[add])
    llm_with_tools = llm.bind_tools([add])


    def call_llm_model(state:State):
        return {"messages":[llm_with_tools.invoke(state["messages"])]}

    #graph
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", call_llm_model)
    builder.add_node("tools", ToolNode(tools))

    ## Add edges
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm", 
        tools_condition)
    builder.add_edge("tools", "tool_calling_llm")

    graph = builder.compile()
    return graph

    # from IPython.display import display
    # from langgraph.graph import StateGraph
    # display(graph)

tool_agent = make_tool_graph()

