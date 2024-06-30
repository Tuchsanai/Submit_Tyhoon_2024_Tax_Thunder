from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from retriver import query_pipe, format_docs, retriver_chain
from tax_calculate import tax_calculator
from router import question_router


class State(TypedDict):
    messages: Annotated[list, add_messages]
    current_node: None


graph_builder = StateGraph(State)


llm_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. You always answer in Thai."),
        ("placeholder", "{conversation}"),
    ]
)

llm = llm_prompt | ChatOpenAI(
    api_key="sk-L1fn4b3Oe2dPbao8JREMWXoAqJcqB7PXcgKaT57G8IUo5uVH",
    openai_api_base="https://api.opentyphoon.ai/v1",
    model="typhoon-v1.5x-70b-instruct",
    temperature=0.4,
    max_tokens=1850,
)


## Nodes
def rag(state: State):
    context = query_pipe(state["messages"][0].content)
    context = format_docs(context)[:2]
    return {
        "messages": [
            retriver_chain.invoke(
                {"question": state["messages"][0].content, "context": context}
            )
        ],
        "current_node": "llm",
    }


def tax_calculation(state: State):
    messages = state["messages"][0].content

    result = tax_calculator(messages)["result"]
    return {"messages": [AIMessage(content=result)], "current_node": "llm"}


def normal_llm(state: State):
    return {
        "messages": [llm.invoke({"conversation": state["messages"]})],
        "current_node": "llm",
    }


## Edges
def route_question(state: State):
    if state.get("current_node") is None:
        query = state["messages"][0].content
        select_node = question_router.invoke({"question": query})
        if select_node.node == "tax_calculation":
            return "tax_calculation"
        elif select_node.node == "rag":
            return "rag"
    else:
        return "llm"


graph_builder.add_node("rag", rag)
graph_builder.add_node("tax_calculation", tax_calculation)
graph_builder.add_node("llm", normal_llm)

graph_builder.set_conditional_entry_point(
    route_question, {"tax_calculation": "tax_calculation", "rag": "rag", "llm": "llm"}
)
graph_builder.set_finish_point("tax_calculation")
graph_builder.set_finish_point("rag")
graph_builder.set_finish_point("llm")
