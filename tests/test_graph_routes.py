"""
Route tests for the LangGraph workflow, using a scripted fake LLM.

The fake LLM returns fixed answers in order and records every prompt it receives,
so each test can check what a node was given, not just what came out.
FAISS, the ReAct agent and Tavily are replaced with stubs: no keys, no network.
"""

from collections import defaultdict

import pytest
from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda

import src.rag.graph_builder as gb
from src.models.grade import Grade
from src.models.route_identifier import RouteIdentifier
from src.models.verification_result import VerificationResult

RETRIEVER_TOOL = "retriever_customer_uploaded_documents"
CHUNK = "Rishi completed his B.E. at Goa University in 2022."
AGENT_SUMMARY = "According to the document, Rishi studied at Goa University."
WEB_RESULT = "The RBI repo rate is 5.5 percent as of the latest policy."


def _text(value):
    """Turn whatever a node passed to the LLM into plain text."""
    if hasattr(value, "to_string"):
        return value.to_string()
    if isinstance(value, list):
        return "\n".join(getattr(m, "content", str(m)) for m in value)
    return str(value)


class FakeLLM(Runnable):
    """Scripted stand-in for ChatGroq."""

    def __init__(self, script):
        # script: {"route": [...], "grade": [...], "verify": [...],
        #          "generate": [...], "rewrite": [...], "general": [...]}
        self.script = {key: list(values) for key, values in script.items()}
        self.calls = defaultdict(list)  # kind -> list of prompt texts

    def _next(self, kind):
        assert self.script.get(kind), f"Unexpected extra '{kind}' call"
        return self.script[kind].pop(0)

    def invoke(self, input, config=None, **kwargs):
        text = _text(input)
        if isinstance(input, list):  # general_llm passes the message list
            kind = "general"
        elif "Rewrite the query" in text:
            kind = "rewrite"
        else:
            kind = "generate"
        self.calls[kind].append(text)
        return AIMessage(content=self._next(kind))

    def with_structured_output(self, schema, **kwargs):
        kind = {RouteIdentifier: "route", Grade: "grade", VerificationResult: "verify"}[schema]

        def run(prompt_value):
            self.calls[kind].append(_text(prompt_value))
            return self._next(kind)

        return RunnableLambda(run)


class FakeRetriever:
    def invoke(self, query):
        return [CHUNK]


class FakeRetrieverTool:
    name = RETRIEVER_TOOL


class FakeAgentExecutor:
    """Stands in for AgentExecutor: one retriever tool call, then a Final Answer."""

    calls = []
    parse_error_only = False  # simulate an agent that never managed to call the tool

    def __init__(self, **kwargs):
        pass

    def invoke(self, inputs):
        FakeAgentExecutor.calls.append(inputs["input"])
        if FakeAgentExecutor.parse_error_only:
            step = (AgentAction(tool="_Exception", tool_input="bad format", log=""),
                    "Invalid Format: Missing 'Action:' after 'Thought:'")
        else:
            step = (AgentAction(tool=RETRIEVER_TOOL, tool_input=inputs["input"], log=""), CHUNK)
        return {"output": AGENT_SUMMARY, "intermediate_steps": [step]}


class FakeTavily:
    calls = []

    def invoke(self, query):
        FakeTavily.calls.append(query)
        return [{"url": "https://example.com", "content": WEB_RESULT}]


def run_graph(monkeypatch, script, question, parse_error_only=False):
    llm = FakeLLM(script)
    FakeAgentExecutor.calls = []
    FakeAgentExecutor.parse_error_only = parse_error_only
    FakeTavily.calls = []
    monkeypatch.setattr(gb, "llm", llm)
    monkeypatch.setattr(gb, "get_raw_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(gb, "get_retriever", lambda: FakeRetrieverTool())
    monkeypatch.setattr(gb, "create_react_agent", lambda *args, **kwargs: None)
    monkeypatch.setattr(gb, "AgentExecutor", FakeAgentExecutor)
    monkeypatch.setattr(gb, "TavilySearchResults", lambda *args, **kwargs: FakeTavily())

    result = gb.builder.invoke({"messages": [HumanMessage(content=question)]})
    for kind, remaining in llm.script.items():
        assert not remaining, f"Scripted '{kind}' responses not used: {remaining}"
    return result, llm


def verdict(faithful, explanation=""):
    return VerificationResult(faithful=faithful, explanation=explanation)


def test_index_route_answers_and_verifies_once(monkeypatch):
    result, llm = run_graph(monkeypatch, {
        "route": [RouteIdentifier(route="index")],
        "grade": [Grade(binary_score="yes")],
        "generate": ["Rishi studied at Goa University."],
        "verify": [verdict(True)],
    }, "Where did Rishi study?")

    assert result["messages"][-1].content == "Rishi studied at Goa University."
    assert len(llm.calls["generate"]) == 1
    assert len(FakeAgentExecutor.calls) == 1


def test_verify_retry_regenerates_from_the_source(monkeypatch):
    """The retry must go back to the retrieved chunk and see why it was rejected."""
    result, llm = run_graph(monkeypatch, {
        "route": [RouteIdentifier(route="index")],
        "grade": [Grade(binary_score="yes")],
        "generate": ["Rishi studied at IIT Bombay.", "Rishi studied at Goa University."],
        "verify": [
            verdict(False, "The claim about IIT Bombay is not supported by the context."),
            verdict(True),
        ],
    }, "Where did Rishi study?")

    first_generate, second_generate = llm.calls["generate"]
    assert CHUNK in first_generate
    # The retry is built from the source, not from the rejected answer...
    assert CHUNK in second_generate, "Retry did not see the retrieved chunk"
    # ...and it is told why the first attempt failed.
    assert "not supported by the context" in second_generate, "Retry did not get the verifier's feedback"

    # Both verifications are judged against the same source.
    first_verify, second_verify = llm.calls["verify"]
    assert CHUNK in first_verify and CHUNK in second_verify
    assert result["messages"][-1].content == "Rishi studied at Goa University."


def test_parse_error_text_is_never_used_as_the_source(monkeypatch):
    """A ReAct parse error ("_Exception" step) must not become the context."""
    result, llm = run_graph(monkeypatch, {
        "route": [RouteIdentifier(route="index")],
        "grade": [Grade(binary_score="yes")],
        "generate": ["Rishi studied at Goa University."],
        "verify": [verdict(True)],
    }, "Where did Rishi study?", parse_error_only=True)

    for prompt in llm.calls["grade"] + llm.calls["generate"] + llm.calls["verify"]:
        assert "Invalid Format" not in prompt
    assert AGENT_SUMMARY in llm.calls["generate"][0]  # falls back to the agent's answer


def test_three_failed_rewrites_fall_back_to_web_search(monkeypatch):
    result, llm = run_graph(monkeypatch, {
        "route": [RouteIdentifier(route="index")],
        "grade": [Grade(binary_score="no")] * 4,
        "rewrite": ["rewritten 1", "rewritten 2", "rewritten 3"],
        "generate": ["The repo rate is 5.5 percent."],
        "verify": [verdict(True)],
    }, "What is the RBI repo rate?")

    assert len(FakeAgentExecutor.calls) == 4  # first try + 3 rewrites
    assert FakeTavily.calls == ["rewritten 3"]
    assert WEB_RESULT in llm.calls["generate"][0]
    assert result["messages"][-1].content == "The repo rate is 5.5 percent."


def test_search_route_uses_web_results(monkeypatch):
    result, llm = run_graph(monkeypatch, {
        "route": [RouteIdentifier(route="search")],
        "generate": ["The repo rate is 5.5 percent."],
        "verify": [verdict(True)],
    }, "Latest RBI repo rate?")

    assert FakeTavily.calls == ["Latest RBI repo rate?"]
    assert WEB_RESULT in llm.calls["generate"][0]
    assert WEB_RESULT in llm.calls["verify"][0]
    assert FakeAgentExecutor.calls == []


def test_general_route_answers_directly(monkeypatch):
    result, llm = run_graph(monkeypatch, {
        "route": [RouteIdentifier(route="general")],
        "general": ["Hello! How can I help?"],
    }, "Hi, how are you?")

    assert result["messages"][-1].content == "Hello! How can I help?"
    assert llm.calls["generate"] == [] and llm.calls["verify"] == []
    assert FakeAgentExecutor.calls == [] and FakeTavily.calls == []
