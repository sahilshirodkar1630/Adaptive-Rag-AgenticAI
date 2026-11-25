"""
Test setup: lets the graph be imported without API keys, model downloads or network.

- Dummy API keys, so ChatGroq can be constructed (it is never called in tests).
- A stand-in for langchain_huggingface, so the embedding model is not downloaded.
"""

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")


class _FakeEmbeddings:
    def __init__(self, *args, **kwargs):
        pass

    def embed_documents(self, texts):
        return [[0.0] * 3 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 3


_hf = types.ModuleType("langchain_huggingface")
_hf.HuggingFaceEmbeddings = _FakeEmbeddings
sys.modules.setdefault("langchain_huggingface", _hf)
