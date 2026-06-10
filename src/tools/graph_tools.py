"""
Tools for graph routing and document grading.
"""

from typing import Literal

from langchain_core.prompts import PromptTemplate

from src.config.settings import Config
from src.llms.groq import llm
from src.models.state import State
from src.models.verification_result import VerificationResult

config = Config()


def routing_tool(state: State) -> Literal["retriever", "general_llm", "web_search"]:
    """
    Route the graph to the appropriate node based on query classification.

    Args:
        state (State): The current state of the graph.

    Returns:
        The next node to execute: "retriever", "general_llm", or "web_search".
    """
    if state["route"] == "index":
        return "retriever"
    elif state["route"] == "general":
        return "general_llm"
    else:
        return "web_search"


def doc_tool(state: State) -> Literal["rewrite", "generate"]:
    """
    Determine whether the query needs rewriting based on grading score.

    Args:
        state (State): The current state of the graph.

    Returns:
        The next node: "generate" if score is "yes", otherwise "rewrite".
    """
    score = state["binary_score"]
    rewrite_count = state.get("rewrite_count", 0)
    print(f"[doc_tool] score={score}, rewrite_count={rewrite_count}")
    if score == "yes":
        return "generate"
    elif rewrite_count >= 3:
        # Retrieval failed after 2 rewrites — fall back to web search
        print("[doc_tool] Rewrite limit reached, falling back to web_search")
        return "web_search"
    else:
        return "rewrite"


def verify_answer(state: State) -> Literal["__end__", "generate"]:
    """
    Verify whether the final answer is faithful to the retrieved context.

    Args:
        state (State): The current state of the graph.

    Returns:
        "__end__" if answer is faithful, otherwise "generate" to retry.
    """
    if state["route"] == "general":
        return "__end__"

    question = state["latest_query"]
    context = state["messages"][-1].content
    final_answer = state["messages"][-1].content

    verify_prompt = PromptTemplate(
        template=config.prompt("verify_prompt"),
        input_variables=["question", "context", "final_answer"]
    )
    llm_with_verification = llm.with_structured_output(VerificationResult)

    verify_chain = verify_prompt | llm_with_verification

    result = verify_chain.invoke({
        "question": question,
        "context": context,
        "final_answer": final_answer
    })

    if result.faithful:
        return "__end__"
    else:
        print("Generating again as answer is not faithful.")
        return "generate"
