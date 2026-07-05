# 🤖 Adaptive RAG — Agentic AI Chatbot

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Motor-brightgreen.svg)](https://motor.readthedocs.io/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3.3--70B-purple.svg)](https://groq.com/)

---

## 📖 Overview

**Adaptive RAG** is a production-grade, agentic AI chatbot that intelligently adapts its retrieval strategy based on the nature of each user query. Unlike traditional RAG systems that blindly retrieve from a vector store on every request, this system first **classifies the query** and routes it to the most appropriate pipeline — indexed document retrieval, general LLM reasoning, or real-time web search.

Built with a **two-level agentic architecture**: LangGraph orchestrates the macro routing strategy, while a ReAct agent handles micro-level reasoning within the retrieval path. The system maintains persistent multi-turn conversation history per user session via MongoDB, and exposes a clean REST API consumed by a Streamlit web interface.

---

## ✨ Key Features

### 🧭 Intelligent Query Routing
Every query is classified before processing. The system retrieves speculative context from the vector store and asks the LLM: *"Is this context relevant enough to answer this question?"* Based on the answer, the query is routed to one of three pipelines — indexed retrieval, general LLM, or web search — ensuring the most accurate and efficient response path every time.

### 🔍 Advanced RAG Pipeline (Adaptive + Corrective)
The retrieval pipeline goes beyond simple fetch-and-generate. Retrieved context is graded for relevance. If the context scores poorly, the query is automatically rewritten and retrieval is retried. After the rewrite budget is exhausted, the system falls back to web search rather than generating a poor answer. This grading → rewrite → retry loop ensures answer quality over raw speed.

### ✅ Self-Verification (Faithfulness Checking)
Once an answer is generated, a verification step checks whether it's actually grounded in the retrieved context — allowing faithful summarization, paraphrase, or reasonable inference, but flagging fabricated facts. If the answer isn't verified as faithful, the graph regenerates, up to a bounded retry limit, before returning the best available answer.

### 🤖 Agentic AI Architecture
The system uses a **two-level agentic design**. At the macro level, LangGraph manages the overall workflow with conditional branching and stateful execution. At the micro level, a ReAct (Reasoning + Acting) agent operates within the retrieval node — reasoning about how to query the tool, interpreting results, and self-correcting within a bounded iteration limit.

### 🧠 State Management
All graph nodes share a single typed `State` object that flows through the entire LangGraph execution. State fields include the full conversation message list, the current route, the relevance score, the latest query string, and a rewrite counter. The `add_messages` annotation ensures message history is appended rather than overwritten across nodes.

### 🔌 API-First Architecture
The backend is a FastAPI application exposing clean REST endpoints for querying and document upload. The frontend communicates exclusively via HTTP — making the backend independently testable, deployable, and replaceable. All LLM prompts are externalized in a YAML file, completely decoupled from application logic.

### 💻 User Interface
A Streamlit web application provides a chat interface, sidebar document upload with description input, and session management. Users enter their name to start a session — a UUID is generated and persisted throughout the conversation. A "Start Over" button clears the session and returns to the home screen.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Streamlit Web Application                   │   │
│  │   • Chat Interface        • Session Management           │   │
│  │   • Document Upload       • Name-based Identity          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │  HTTP (REST)
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  POST /rag/query            POST /rag/documents/upload   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestration                      │
│                                                                 │
│   ┌───────────────┐     ┌──────────────────────────────────┐    │
│   │ query_analysis│───▶ │  routing_tool()                  │   │
│   │               │     │  "index" / "general" / "search"  │    │
│   └───────────────┘     └──────────┬───────────┬───────────┘    │
│                                    │           │                │
│              ┌─────────────────────┤           │                │
│              │                     │           │                │
│              ▼                     ▼           ▼                │
│       ┌────────────┐        ┌──────────┐  ┌──────────┐          │
│       │  retriever │        │general   │  │web_search│          │
│       │ (ReAct     │        │_llm      │  │(Tavily)  │          │
│       │  Agent)    │        └────┬─────┘  └────┬─────┘          │
│       └─────┬──────┘             │             │                │
│             │                    │             │                │
│             ▼                    │             │                │
│          ┌──────┐                │             │                │
│          │grade │                │             │                │
│          └──┬───┘                │             │                │
│             │                    │             │                │
│     ┌───────┴────────┐           │             │                │
│     │ doc_tool()     │           │             │                │
│     │ yes / no       │           │             │                │
│     └───────┬────────┘           │             │                │
│             │                    │             │                │
│    ┌────────┴──────┐             │             │                │
│    ▼               ▼             │             │                │
│ ┌──────┐      ┌────────┐         │             │                │
│ │genera│      │rewrite │──(retry, <3)──▶ retriever              │
│ │  te  │      │        │──(fallback, ≥3)──▶ web_search          │
│ └──┬───┘      └────────┘                       │                │
│    │                                           │                │
│    ▼◀──────────────────────────────────────────┘                │
│ ┌────────┐                                                      │
│ │generate│◀────────────────────────────────────┐               │
│ └───┬────┘                                      │               │
│     ▼                                           │               │
│ ┌────────┐   not faithful, retries < 2          │               │
│ │ verify │───────────────────────────────────────┘              │
│ └───┬────┘                                                      │
│     │ faithful, or retries ≥ 2                                  │
│     ▼                                                           │
│    END                                                          │
└─────────────────────────────────────────────────────────────────┘
                         │               │
            ┌────────────┘               └───────────────┐
            ▼                                            ▼
┌───────────────────┐                       ┌───────────────────────┐
│   FAISS / Qdrant  │                       │      MongoDB          │
│   Vector Store    │                       │   Chat History        │
│   (HuggingFace    │                       │   (per session_id)    │
│   Embeddings)     │                       └───────────────────────┘
└───────────────────┘
```

---

## 🔀 Graph Nodes

| Node | File | Function | Purpose |
|---|---|---|---|
| `query_analysis` | `graph_builder.py` | `query_classifier()` | Classifies query and performs speculative retrieval to determine route |
| `retriever` | `graph_builder.py` | `retriever_node()` | Builds a fresh ReAct agent at query time and invokes it with the retriever tool |
| `grade` | `graph_builder.py` | `grade()` | Scores retrieved context for relevance — returns `"yes"` or `"no"` |
| `rewrite` | `graph_builder.py` | `rewrite_query()` | Rewrites the query for better vector search, increments rewrite counter |
| `generate` | `graph_builder.py` | `generate()` | Synthesizes a clean, readable answer from retrieved context |
| `web_search` | `graph_builder.py` | `web_search()` | Searches the web via Tavily for real-time or niche information |
| `general_llm` | `graph_builder.py` | `general_llm()` | Calls the LLM directly for general knowledge and casual conversation |
| `generate` | `graph_builder.py` | `generate()` | Synthesizes a clean, readable answer from the selected context |
| `verify` | `graph_builder.py` | `verify_answer()` | Checks whether the generated answer is faithful to its supporting context; increments the verification counter |


### Conditional Edge Functions (`graph_tools.py`)

**`routing_tool(state)`** — reads `state["route"]` set by `query_classifier`:
- `"index"` → `retriever`
- `"general"` → `general_llm`
- `"search"` → `web_search`

**`doc_tool(state)`** — reads `state["binary_score"]` and `state["rewrite_count"]`:
- `score == "yes"` → `generate`
- `score == "no"` and `rewrite_count < 2` → `rewrite`
- `score == "no"` and `rewrite_count >= 2` → `web_search` (fallback)

**`route_after_verify(state)`** — reads `state["verified"]` and `state["verify_count"]`:
- `verified == True` → `__end__`
- `verified == False` and `verify_count < 2` → `generate` (regenerate and re-verify)
- `verified == False` and `verify_count >= 2` → `__end__` (return best available answer)

---

## 📁 Project Structure

```
AdaptiveRag/
├── src/
│   ├── main.py                         # FastAPI app entry point
│   ├── api/
│   │   └── routes.py                   # POST /rag/query, POST /rag/documents/upload
│   ├── config/
│   │   ├── settings.py                 # YAML prompt loader (Config class)
│   │   └── prompts.yaml                # All LLM prompt templates
│   ├── core/
│   │   ├── config.py                   # Env vars — API keys, DB URLs
│   │   └── logger.py                   # Logging configuration
│   ├── db/
│   │   └── mongo_client.py             # Async MongoDB client via Motor
│   ├── llms/
│   │   └── groq.py                     # ChatGroq — LLaMA 3.3 70B instance
│   ├── memory/
│   │   ├── chat_history_mongo.py       # Persistent MongoDB chat history
│   │   └── chathistory_in_memory.py    # In-memory fallback (dev only)
│   ├── models/
│   │   ├── state.py                    # LangGraph State TypedDict
│   │   ├── query_request.py            # Pydantic: query + session_id
│   │   ├── verification_result.py      # Pydantic: faithful + explanation
│   │   └── (optional) grade.py / route_identifier.py  # Structured LLM output for grading/routing
│   ├── rag/
│   │   ├── graph_builder.py            # All node implementations + graph assembly (`builder`)
│   │   ├── reAct_agent.py              # ReAct agent factory function
│   │   ├── retriever_setup.py          # FAISS vector store + retriever tools
│   │   └── document_upload.py          # File validation, chunking, indexing
│   └── tools/
│       ├── common_tools.py             # LLM-powered description enhancer
│       └── graph_tools.py              # routing_tool / doc_tool / route_after_verify edge functions
│
├── streamlit_app/                      # Optional Streamlit frontend
│   ├── home.py                         # Name entry + UUID session creation
│   ├── pages/
│   │   └── chat.py                     # Chat UI + sidebar document upload
│   └── utils/
│       └── api_client.py               # HTTP client for the FastAPI backend
│
├── .env                                # API keys — never commit this
├── requirements.txt
└── README.md
```

---
---

## 🧠 Key Design Decisions

### ReAct Agent Built at Query Time, Not Import Time
The ReAct agent is created inside a factory function (e.g. `build_agent_executor()`) that's called fresh on every query rather than once at module import. This ensures the agent always picks up the current FAISS vector store after a document upload — a module-level agent built at import time would be frozen with the empty dummy store from server startup.

```python
# WRONG — frozen at import time, misses documents uploaded later
tools = [get_retriever()]
agent_executor = AgentExecutor(tools=tools, ...)

# CORRECT — built fresh per query, always uses the current vector store
def build_agent_executor():
    fresh_tools = [get_retriever()]
    return AgentExecutor(tools=fresh_tools, ...)
```

### Two Separate Retriever Interfaces
`get_retriever()` returns a LangChain **tool** (name + description + formatted output) for agent use. `get_raw_retriever()` returns a plain retriever producing `List[Document]` objects directly — used by the query classifier, which needs real document objects for classification context rather than an agent-formatted string.

### Rewrite Counter as a Loop Guard
The `grade → rewrite → retriever` loop has no natural exit if retrieval keeps failing. A `rewrite_count` field on `State` caps retries at 3 attempts before falling back to web search, guaranteeing the graph terminates cleanly instead of hitting LangGraph's recursion limit.

### Verification Counter as a Second Loop Guard
Similarly, `verify_count` caps the generate → verify → regenerate loop at 2 attempts. If the answer still isn't verified as faithful after that, the graph returns the best answer produced rather than looping indefinitely.

### Prompts Fully Externalized in YAML
All LLM prompt templates live in `prompts.yaml` and are loaded via the `Config` class. No prompt text lives in Python files, so prompt changes never require touching application logic and prompts can be versioned independently.

### MongoDB Over In-Memory for Chat History
Motor (the async MongoDB driver) pairs naturally with FastAPI's async model without blocking the event loop. Messages are stored as documents — `session_id`, `type`, `content`, `timestamp`, `additional_kwargs` — requiring no schema migrations as the message format evolves. History survives restarts and scales across multiple instances, unlike a class-level Python dict.

### Name-Based Session Identity
The Streamlit frontend asks for a name and generates a UUID as the `session_id`, sent with every request and used as the MongoDB partition key. It gives isolated, persistent conversation history per session without formal authentication — sufficient for a demonstration project. Upgrading to verified identity later only requires changing how `session_id` is derived, not the storage layer.

---


## 🔌 API Endpoints

### `POST /rag/query`
Process a user query through the adaptive RAG pipeline.

**Request:**
```json
{
  "query": "Where did Satyam study?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response:**
```json
{
  "result": {
    "type": "ai",
    "content": "Satyam studied B.E. in Electronics Engineering at SAKEC Chembur with a CGPA of 9.22/10."
  }
}
```

---

### `POST /rag/documents/upload`
Upload a PDF or TXT document for indexing into the vector store.

**Headers:**
```
X-Description: Resume of Satyam Gupta, software engineer at Webstac
```

**Form Data:**
```
file: <PDF or TXT binary>
```

**Response:**
```json
{ "status": true }
```

The description is enhanced by the LLM into a retriever tool instruction before storage. This enhanced description is what guides the ReAct agent's decision of when to call the retriever tool.

---

## 🚀 Usage Guide

### Prerequisites

- Python 3.9+
- Docker Desktop (for MongoDB)
- [Groq API key](https://console.groq.com) — free tier available
- [Tavily API key](https://tavily.com) — free tier available

---

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Adaptive-Rag.git
cd Adaptive-Rag

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

### Environment Configuration

Create a `.env` file in the project root:

```env
# LLM
GROQ_API_KEY=your_groq_api_key_here

# Web Search
TAVILY_API_KEY=your_tavily_api_key_here

# Database
MONGO_URL=mongodb://localhost:27017

# Vector Store (Qdrant — for production use)
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_CODE_COLLECTION=codebase
QDRANT_DOCS_COLLECTION=guidelines
```

> ⚠️ Never commit `.env` to version control. Add it to `.gitignore`.

---

### Running the Application

**Step 1 — Start MongoDB**
```bash
docker run -d --name adaptive-rag-mongo -p 27017:27017 mongo:latest
```

**Step 2 — Start the FastAPI backend**
```bash
python -m uvicorn src.main:app --reload
```
- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`

**Step 3 — Start the Streamlit frontend**
```bash
streamlit run streamlit_app/home.py
```
- UI: `http://localhost:8501`

**Step 4 — Use the application**
1. Enter your name on the home screen — a UUID session is created
2. Upload a PDF or TXT from the sidebar with a short description
3. Ask questions in the chat input
4. The system automatically routes to the best strategy and returns an answer
5. Click "Start Over" to clear your session and return to the home screen

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM | Groq — LLaMA 3.3 70B Versatile | Fast inference for all LLM calls |
| Workflow | LangGraph | Graph-based agentic orchestration |
| Agent | LangChain ReAct | Reasoning + acting within retrieval node |
| Web Framework | FastAPI | Async REST API with automatic docs |
| Frontend | Streamlit | Chat UI and document upload |
| Vector Store | FAISS (local) / Qdrant (production) | Semantic document retrieval |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | 384-dim sentence embeddings, CPU-friendly |
| Chat History | MongoDB + Motor | Async persistent session storage |
| Web Search | Tavily | Optimized search API for LLM use |
| Validation | Pydantic v2 | Request models + structured LLM output |

---

## ⚡ Performance Considerations

- **Embedding model runs on CPU** — `all-MiniLM-L6-v2` is 80MB and produces 384-dimensional vectors. It's fast enough for local use. For production, switch to GPU inference or a hosted embeddings API.
- **ReAct agent is capped at 2 iterations** — `max_iterations=2` in `AgentExecutor` prevents runaway tool calls and keeps latency predictable.
- **Rewrite loop capped at 2 retries** — `rewrite_count >= 2` falls back to web search rather than looping indefinitely.
- **FAISS is in-memory** — vector store resets on server restart. Switch to Qdrant (already stubbed in `retriever_setup.py`) for persistent, production-grade storage.
- **`builder.invoke()` is synchronous** — blocks the FastAPI event loop under concurrent load. Production upgrade: replace with `await builder.ainvoke()`.

---

## 📚 Documentation References
-  CODE_STYLE_GUIDE.md - Comprehensive coding standards
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain ReAct Agents](https://python.langchain.com/docs/modules/agents/agent_types/react/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Groq API](https://console.groq.com/docs/openai)
- [Tavily Search API](https://docs.tavily.com/)
- [FAISS](https://faiss.ai/)
- [Motor — Async MongoDB](https://motor.readthedocs.io/en/stable/)
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## ❓ FAQ

**Q: What file types are supported for upload?**
A: PDF and TXT only. The system validates the file extension before processing.

**Q: Is chat history preserved across browser refreshes?**
A: Yes — as long as the same session UUID is sent with each request, MongoDB returns the full conversation history. The Streamlit frontend holds the session ID in `st.session_state` which persists within a browser tab session.

**Q: What happens if I upload a new document?**
A: The new document replaces the existing FAISS vector store. The current implementation supports one active document collection at a time. For multi-document support, switch to Qdrant with named collections.

**Q: Why does the system sometimes route to web search even for document questions?**
A: After two failed rewrite attempts, the grader still returns `"no"` (retrieved context is not relevant), and the system falls back to web search rather than generating a low-quality answer. This is intentional — a web answer is better than a hallucinated one.

**Q: Can the LLM be swapped out?**
A: Yes. The `llm` instance in `groq.py` follows the LangChain `BaseChatModel` interface. Any LangChain-compatible LLM (OpenAI, Anthropic, Mistral) can be substituted by changing the single `llm` import used across all nodes.

---

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Vector search powered by [FAISS](https://faiss.ai/) with [Qdrant](https://qdrant.tech/) as the production target
- LLM inference by [Groq](https://groq.com/) running LLaMA 3.3 70B
- Web search by [Tavily](https://tavily.com/)
- UI powered by [Streamlit](https://streamlit.io/)
- Thanks to the open-source community for the tools that made this possible

---

## 📈 Project Status

- ✅ Core adaptive RAG pipeline implemented (classify → route → retrieve/search → grade → generate)
- ✅ Intelligent query routing (index / general / search)
- ✅ Grading + rewrite loop with fallback guard (max 3 rewrites)
- ✅ Self-verification (faithfulness) loop with fallback guard (max 2 retries)
- ✅ Document upload, chunking, and FAISS indexing
- ✅ MongoDB persistent chat history + in-memory fallback
- ✅ All prompts externalized in YAML
- ⏳ Production deployment hardening (Qdrant migration, fully async graph invocation, auth)

---

## 🗺️ Roadmap


- [ ] Replace FAISS with Qdrant for persistent multi-document support
- [ ] Multi-language query and response support
- [ ] Extended LLM provider support (OpenAI, Anthropic, Mistral)
- [ ] Advanced authentication with verified session identity
- [ ] Streaming responses via Server-Sent Events
- [ ] Analytics dashboard for query routing distribution
- [ ] Performance benchmarks and cost optimization
- [ ] Real-time collaboration across shared sessions

---

*Built as a demonstration of adaptive, agentic RAG architecture using LangGraph and LangChain.*
