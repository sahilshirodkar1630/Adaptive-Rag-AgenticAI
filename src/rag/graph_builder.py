"""
Graph builder module for the adaptive RAG system.
"""

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph

from src.rag.retriever_setup import get_retriever
from src.config.settings import Config
from src.llms.groq import llm
from src.models.grade import Grade
from src.models.route_identifier import RouteIdentifier
from src.models.state import State
from src.models.verification_result import VerificationResult
from src.tools.graph_tools import routing_tool, doc_tool
from src.rag.retriever_setup import get_raw_retriever
from src.tools.graph_tools import routing_tool, doc_tool,route_after_verify

config = Config()


# Node implementations
def query_classifier(state: State):
    """
    Classify the query to determine if it's related to indexed documents.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated state with route and latest_query.
    """
    question = state["messages"][-1].content
    retriever = get_raw_retriever() 
    context = retriever.invoke(question)
    print("docs received from FAISS")
    print(context)

    llm_with_structured_output = llm.with_structured_output(RouteIdentifier)
    classify_prompt = PromptTemplate(
        template=config.prompt("classify_prompt"),
        input_variables=["question", "context"]
    )
    chain = classify_prompt | llm_with_structured_output
    result = chain.invoke({"question": question, "context": context})
    print("result received is in query classifier")
    print(result.route)

    return {
        "messages": state["messages"],
        "route": result.route,
        "latest_query": question,
        "rewrite_count": 0,        # ← add
        "verify_count": 0,         # ← add
        "verified": False,         # ← add
        "retrieved_context": None  # ← add
    }


def general_llm(state: State):
    """
    Fetch general common knowledge result from the LLM.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages from LLM.
    """
    result = llm.invoke(state["messages"])
    print("inside general llm")
    print(result)
    return {"messages": result}


def retriever_node(state: State):
    """
    Retrieve results from vector stores using the reAct agent.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages with tool calls.
    """

    config = Config()

    # Build fresh retriever tool — picks up current _faiss_vectorstore
    fresh_tools = [get_retriever()]

    prompt = ChatPromptTemplate.from_messages([
        ("system", config.prompt("system_prompt")),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}")
    ])

    react_agent = create_react_agent(llm, fresh_tools, prompt)
    agent_executor = AgentExecutor(
        agent=react_agent,
        tools=fresh_tools,
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=True,
        return_intermediate_steps=True
    )

    messages = state["latest_query"]
    result = agent_executor.invoke({"input": messages})

    # Extract tool calls
    intermediate_steps = result.get("intermediate_steps", [])
    tool_calls = []
    if intermediate_steps:
        for action, tool_result in intermediate_steps:
            tool_calls.append({
                "tool": action.tool,
                "input": action.tool_input,
            })

    # If agent hit iteration limit, extract best answer from last tool result
    output = result["output"]
    if "Agent stopped" in output or "iteration" in output.lower():
        if intermediate_steps:
            # Use the last tool result as the output
            last_tool_result = intermediate_steps[-1][1]
            output = last_tool_result if isinstance(last_tool_result, str) else str(last_tool_result)
        else:
            output = "I was unable to retrieve relevant information. Please try rephrasing your question."

    new_message = AIMessage(
        content=result["output"],
        additional_kwargs={"tool_calls": tool_calls},
    )

    return {
        "messages": [new_message]
    }


def grade(state: State):
    """
    Grade the results retrieved from vector stores.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated state with binary_score.
    """
    grading_prompt = PromptTemplate(
        template=config.prompt("grading_prompt"),
        input_variables=["question", "context"]
    )
    context = state["messages"][-1].content
    question = state["latest_query"]

    llm_with_grade = llm.with_structured_output(Grade)

    chain_graded = grading_prompt | llm_with_grade
    result = chain_graded.invoke({"question": question, "context": context})

    print(result)
    return {"messages": state["messages"], "binary_score": result.binary_score}


def rewrite_query(state: State):
    """
    Rewrite the query to get better retrieval results.

    Args:
        state (State): State of the question.

    Returns:
        dict: Updated latest_query.
    """
    count = state["rewrite_count"] or 0
    query = state["latest_query"]
    rewrite_prompt = PromptTemplate(
        template=config.prompt("rewrite_prompt"),
        input_variables=["query"]
    )
    chain = rewrite_prompt | llm
    result = chain.invoke({"query": query})
    print(result)

    return {
        "latest_query": result.content,
        "rewrite_count": count + 1
    }


def generate(state: State):
    """
    Generate the final answer for the user.

    Args:
        state (State): State of the question.

    Returns:
        dict: Generated response.
    """
    context = state["messages"][-1].content

    generate_prompt = PromptTemplate(
        template=config.prompt("generate_prompt"),
        input_variables=["context"]
    )

    generate_chain = generate_prompt | llm
    result = generate_chain.invoke({"context": context})

    return {
        "retrieved_context": context,
        "verify_count": (state["verify_count"] or 0) + 1,
        "messages": [{"role": "assistant", "content": result.content}]
        }


def web_search(state: State):
    """
    Search the web for the rewritten query.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Search results as messages.
    """
    # Initialize the Tavily tool
    search_tool = TavilySearchResults()

    # Search a query
    result = search_tool.invoke(state["latest_query"])

    contents = [item["content"] for item in result if "content" in item]
    print(contents)

    return {
        "messages": [{"role": "assistant", "content": "\n\n".join(contents)}]
    }

def verify_answer_node(state: State) -> dict:
    if state["route"] == "general":
        return {"verified": True}

    question = state["latest_query"]
    context = state["retrieved_context"]
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

    return {"verified": result.faithful}   # returns dict ✅

# Build the graph
graph = StateGraph(State)

graph.add_node("query_analysis", query_classifier)
graph.add_node("retriever", retriever_node)
graph.add_node("grade", grade)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("general_llm", general_llm)
graph.add_node("verify", verify_answer_node)

graph.add_edge(START, "query_analysis")
graph.add_edge("web_search", "generate")
graph.add_edge("retriever", "grade")
graph.add_edge("rewrite", "retriever")
graph.add_conditional_edges("query_analysis", routing_tool)
graph.add_conditional_edges("grade", doc_tool)
graph.add_edge("generate", "verify")
graph.add_conditional_edges("verify", route_after_verify)
graph.add_edge("general_llm", END)

builder = graph.compile()

 #Auto-generate graph visualization
try:
    graph_png = builder.get_graph().draw_mermaid_png()
    with open("adaptive_RAG.png", "wb") as f:
        f.write(graph_png)
    print("Graph visualization updated: adaptive_RAG.png")
except Exception as e:
    print(f"Could not update graph visualization: {e}")

