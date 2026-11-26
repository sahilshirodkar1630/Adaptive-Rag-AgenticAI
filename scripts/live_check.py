"""
Live smoke test of the LangGraph workflow against the real services
(Groq, Tavily, HuggingFace embeddings, FAISS). MongoDB and the API are not used.

Usage, from the project root, with GROQ_API_KEY and TAVILY_API_KEY in .env:
    python scripts/live_check.py

It indexes a small sample document, asks four questions and prints, for every
node that runs, what it wrote to the graph state: the route, the grade, the
source the answer is grounded in, the verifier's verdict and the final answer.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from src.rag.retriever_setup import retriever_chain  # noqa: E402
from src.rag.graph_builder import builder  # noqa: E402

SAMPLE_DOCUMENT = (
    "Ananya Rao is a data engineer at Contoso Analytics in Pune, where she has "
    "worked since July 2021. She builds batch pipelines in PySpark and maintains "
    "the company's Airflow deployment. Ananya completed her B.Tech in Computer "
    "Engineering at Savitribai Phule Pune University in 2019. Before Contoso she "
    "spent two years as a junior developer at Fabrikam Software. Outside work she "
    "plays competitive chess."
)
DESCRIPTION = "Profile of Ananya Rao, a data engineer: her job, education and background."

QUESTIONS = [
    ("document", "Where did Ananya study, and when did she graduate?"),
    ("not in document", "What is Ananya Rao's salary at Contoso?"),
    ("web", "What is the latest repo rate set by the Reserve Bank of India?"),
    ("small talk", "Hi, how are you today?"),
]

FIELDS = ["route", "latest_query", "binary_score", "rewrite_count",
          "retrieved_context", "verify_count", "verified", "verify_feedback"]


def short(value, limit=160):
    text = str(value).replace("\n", " ")
    text = text.encode("ascii", "replace").decode("ascii")  # safe for any console
    return text if len(text) <= limit else text[:limit] + "..."


def index_sample():
    (ROOT / "description.txt").write_text(f"Tool Instruction: {DESCRIPTION}", encoding="utf-8")
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(
        [Document(page_content=SAMPLE_DOCUMENT, metadata={"source": "sample"})]
    )
    if not retriever_chain(chunks):
        raise SystemExit("Could not build the FAISS index")


def run(label, question):
    print("\n" + "=" * 78)
    print(f"[{label}] {question}")
    print("=" * 78)
    final_answer = None
    for update in builder.stream({"messages": [HumanMessage(content=question)]},
                                 stream_mode="updates"):
        for node, changes in update.items():
            changes = changes or {}
            details = [f"{key}={short(changes[key])}" for key in FIELDS if key in changes]
            print(f"  -> {node:<15} " + ("; ".join(details) if details else ""))
            messages = changes.get("messages")
            if messages:
                last = messages[-1] if isinstance(messages, list) else messages
                final_answer = last["content"] if isinstance(last, dict) else getattr(last, "content", last)
    print(f"  ANSWER: {short(final_answer, 400)}")


def main():
    print("Indexing the sample document (first run may download the embedding model)...")
    index_sample()
    failures = 0
    for label, question in QUESTIONS:
        try:
            run(label, question)
        except Exception as error:  # keep going so one failure doesn't hide the rest
            failures += 1
            print(f"  ERROR: {type(error).__name__}: {short(error, 400)}")
    print("\n" + ("All questions completed." if not failures else f"{failures} question(s) failed."))


if __name__ == "__main__":
    main()
