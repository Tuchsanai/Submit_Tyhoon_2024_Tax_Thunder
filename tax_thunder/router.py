from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

class RouteQuery(BaseModel):
    """Route a user query to the most relevant node."""

    node: Literal["tax_calculation", "rag"] = Field(
        ...,
        description="Route the user question to either 'tax_calculation' or 'rag' based on the query content.",
    )

llm = ChatOpenAI(
        api_key="xxxxxxx",
        openai_api_base="https://api.opentyphoon.ai/v1",
        model="typhoon-v1.5x-70b-instruct",
        temperature=0,
        max_tokens=1000,
    )

structured_llm_router = llm.with_structured_output(RouteQuery)
system = """You are an expert at routing a user question to a rag system or tax calculation.
RAG (Retrieval-Augmented Generation): Route to this for questions about tax rules, policies, definitions, or general information that doesn't require immediate calculation.
Tax Calculation: Route to this if the question explicitly asks for a calculation, involves specific numbers, or requires computing a tax amount.
If a question is ambiguous or could potentially require both systems, choose the rag system."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router