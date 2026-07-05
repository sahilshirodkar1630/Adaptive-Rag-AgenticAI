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


def route_after_verify(state: State) -> Literal["__end__", "generate"]:
    """
    Verify whether the final answer is faithful to the retrieved context.

    Args:
        state (State): The current state of the graph.

    Returns:
        "__end__" if answer is faithful, otherwise "generate" to retry.
    """
    if state["verified"]:
        return "__end__"
    elif (state["verify_count"] or 0) >= 2:
        print("Verify limit reached, returning best answer available.")
        return "__end__"
    else:
        print("Answer not faithful, regenerating.")
        return "generate"