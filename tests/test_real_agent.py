"""
Integration test with the REAL retrieval stack: a real FAISS index, the real
retriever tool from get_retriever(), and the real LangChain create_react_agent +
AgentExecutor, which parse the ReAct text and call the tool themselves.

Only the language models are scripted (no API keys): one fake chat model writes
the agent's Thought / Action / Final Answer text, and the FakeLLM from the route
tests answers the classifier, grader, generator and verifier.
"""

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain.agents import create_react_agent as real_create_react_agent

import src.rag.graph_builder as gb
import src.rag.retriever_setup as rs
from src.models.grade import Grade
from src.models.route_identifier import RouteIdentifier
from tests.test_graph_routes import FakeLLM, FakeRetriever, verdict

DOC_TEXT = "Rishi completed his B.E. in Information Technology at Goa University in 2022."


@pytest.fixture
def real_index(monkeypatch):
    """A real FAISS index holding one uploaded document."""
    store = FAISS.from_documents(
        [Document(page_content=DOC_TEXT, metadata={"source": "resume.txt"})],
        DeterministicFakeEmbedding(size=16),
    )
    monkeypatch.setattr(rs, "_faiss_vectorstore", store)
    return store


def run_with_real_agent(monkeypatch, agent_responses, script):
    llm = FakeLLM(script)
    agent_llm = FakeListChatModel(responses=agent_responses)

    monkeypatch.setattr(gb, "llm", llm)
    monkeypatch.setattr(gb, "get_raw_retriever", lambda: FakeRetriever())
    # Real agent and executor; only the model that writes the ReAct text is scripted.
    monkeypatch.setattr(
        gb, "create_react_agent",
        lambda _llm, tools, prompt: real_create_react_agent(agent_llm, tools, prompt),
    )

    result = gb.builder.invoke({"messages": [HumanMessage(content="Where did Rishi study?")]})
    for kind, remaining in llm.script.items():
        assert not remaining, f"Scripted '{kind}' responses not used: {remaining}"
    return result, llm


def test_real_agent_chunks_reach_grade_generate_and_verify(monkeypatch, real_index):
    result, llm = run_with_real_agent(monkeypatch, [
        "Thought: I should search the uploaded document.\n"
        "Action: retriever_customer_uploaded_documents\n"
        "Action Input: Rishi education",
        "Thought: I now have enough information to answer\n"
        "Final Answer: He studied somewhere in Goa.",
    ], {
        "route": [RouteIdentifier(route="index")],
        "grade": [Grade(binary_score="yes")],
        "generate": ["Rishi studied at Goa University."],
        "verify": [verdict(True)],
    })

    # The text the real retriever tool returned (the FAISS chunk) is the source...
    assert DOC_TEXT in llm.calls["grade"][0]
    assert DOC_TEXT in llm.calls["generate"][0]
    assert DOC_TEXT in llm.calls["verify"][0]
    # ...not the agent's vaguer summary.
    assert "somewhere in Goa" not in llm.calls["generate"][0]
    assert result["messages"][-1].content == "Rishi studied at Goa University."


def test_real_agent_parse_error_then_recovery(monkeypatch, real_index):
    """A malformed reply is handled by the real executor and never becomes the source."""
    result, llm = run_with_real_agent(monkeypatch, [
        "I think I know the answer already.",  # no Action, no Final Answer: parse error
        "Thought: I should search the uploaded document.\n"
        "Action: retriever_customer_uploaded_documents\n"
        "Action Input: Rishi education",
        "Thought: I now have enough information to answer\n"
        "Final Answer: Rishi studied at Goa University.",
    ], {
        "route": [RouteIdentifier(route="index")],
        "grade": [Grade(binary_score="yes")],
        "generate": ["Rishi studied at Goa University."],
        "verify": [verdict(True)],
    })

    context = llm.calls["generate"][0]
    assert DOC_TEXT in context
    assert "Invalid Format" not in context and "I think I know" not in context
