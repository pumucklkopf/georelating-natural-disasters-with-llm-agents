from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from model_interaction.chatAI import ChatAIHandler
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from pydantic import BaseModel, Field

# Initialize the ChatAIHandler
chatAI = ChatAIHandler()
model = chatAI.get_model("mistral-large-instruct")

messages = [
    SystemMessage(
        content="You are a helpful assistant! Your name is Bob."
    ),
    HumanMessage(
        content="What is your name?"
    )
]

def search_agent(state: MessagesState):
    response = model.invoke(state.messages)
    return {"messages": [response]}

def search_refiner(state: MessagesState):
    response = model.invoke(state.messages)
    return {"messages": [response],
            "next": response["next_agent"]}

def candidate_selector(state: MessagesState):
    response = model.invoke(state.messages)
    return {"messages": [response]}

def selection_refiner(state: MessagesState):
    response = model.invoke(state.messages)
    return {"messages": [response]}

def relation_reasoner(state: MessagesState):
    response = model.invoke(state.messages)
    return {"messages": [response]}

def relation_refiner(state: MessagesState):
    response = model.invoke(state.messages)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node(search_agent)
builder.add_node(search_refiner)
builder.add_node(candidate_selector)
builder.add_node(selection_refiner)
builder.add_node(relation_reasoner)
builder.add_node(relation_refiner)

# define the flow explicitly
builder.add_edge(START, "search_agent")
builder.add_edge("search_agent", "search_refiner")
builder.add_conditional_edges("search_refiner", lambda state: state["next"])
builder.add_conditional_edges(
    "Researcher",
    search_refiner,
    {"continue": "chart_generator", "call_tool": "call_tool", END: END},
)

# conditional edge: if requests doesn't return status code 200, we need to decide what to do:
# -LLM retry for 14	invalid parameter, 15 no result found, 21 invalid input
# -Deterministically wait and retry: 13	database timeout, 18 daily limit of credits exceeded, 19 hourly limit of credits exceeded, 20 weekly limit of credits exceeded, 22 server overloaded exception
# -Exit execution with error handling for all other errors