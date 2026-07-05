# Document Flow - Visual Guide


## 🔄 How Documents Flow Through the System

### **Scenario 1: First Retriever Access (No Documents Yet)**

```
┌─────────────────────────────────────────────────┐
│         FIRST CALL TO get_retriever()           │
│         (before any document is uploaded)        │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────┐
    │  Create DUMMY Document             │
    │  - Content: "No documents have     │
    │    been uploaded yet..."           │
    │  - Purpose: Init FAISS vectorstore │
    └────────────┬───────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │  Initialize FAISS Vectorstore      │
    │  Documents in store: 1 (dummy)     │
    │  Status: ✅ Ready                  │
    └────────────────────────────────────┘
```

---

### **Scenario 2: User Uploads Document**

```
┌────────────────────────────────────────────┐
│  USER UPLOADS DOCUMENT                     │
│  POST /rag/documents/upload                │
│  File: "Python_Guide.pdf"                  │
│  Description: "Python tutorial"            │
└──────────────┬─────────────────────────────┘
               │
               ▼
        ┌─────────────────────┐
        │  Load PDF File      │
        │  Extract content    │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Split into Chunks  │
        │  Chunk Size: 1000   │
        │  Output: 20 chunks  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │  retriever_chain(chunks)     │
        │  FAISS.from_documents(chunks)│
        │  ⚠️ REPLACES existing store  │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │  FAISS Store Replaced        │
        │  Documents in store: 20      │
        │  - dummy: ❌ gone            │
        │  - 20 real chunks (Python)   │
        │  Status: ✅ Ready            │
        └──────────────────────────────┘
```

---

### **Scenario 3: User Queries Document**

```
┌────────────────────────────────────────────┐
│  USER ASKS QUESTION                        │
│  POST /rag/query                           │
│  Query: "What is Python?"                  │
└──────────────┬──────────────────────────────┘
               │
               ▼
        ┌─────────────────────────────┐
        │  Query Router               │
        │  - Classify query type      │
        │  - Route to appropriate     │
        │    pipeline                 │
        └──────────┬──────────────────┘
                   │
                   ▼
        ┌─────────────────────────────┐
        │  get_retriever() called     │
        │  - Get CURRENT FAISS store  │
        │  - Returns tool             │
        └──────────┬──────────────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │  FAISS Search                │
        │  Search query: "What is...?" │
        │                              │
        │  Search current store:       │
        │  ✅ Find matching chunks     │
        │     from Python_Guide.pdf    │
        │  (dummy no longer present)   │
        │                              │
        │  Results: 3-5 relevant       │
        │  document chunks             │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌───────────────────────────────┐
        │  Pass to LLM                  │
        │  - Context: Found chunks      │
        │  - Question: User query       │
        └──────────┬────────────────────┘
                   │
                   ▼
        ┌────────────────────────────────┐
        │  Generate Answer               │
        │  Answer based on YOUR doc      │
        │  "Python is a programming..."  │
        └──────────┬─────────────────────┘
                   │
                   ▼
        ┌──────────────────────────────┐
        │  Return to User              │
        │  Answer: "Python is..."      │
        │  Status: ✅ Complete         │
        └──────────────────────────────┘
```

---

### **Scenario 4: Upload a Second Document — Replaces, Doesn't Add**

```
┌──────────────────────────────────────────┐
│  USER UPLOADS ANOTHER DOCUMENT            │
│  File: "Web_Dev_Guide.pdf"                │
└────────────┬───────────────────────────��──┘
             │
             ▼
    ┌──────────────────────┐
    │  Load & Process      │
    │  Split into chunks   │
    │  Output: 15 chunks   │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │  retriever_chain(chunks)     │
    │  FAISS.from_documents(chunks)│
    │  ⚠️ Builds a NEW store from  │
    │     ONLY these 15 chunks     │
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────┐
    │  FAISS Store Replaced Again  │
    │  Documents in store: 15      │
    │  - Python Guide chunks: ❌   │
    │    (gone — overwritten)      │
    │  - Web Dev Guide chunks: ✅  │
    │                              │
    │  ⚠️ Can ONLY search          │
    │  Web_Dev_Guide.pdf now —     │
    │  Python_Guide.pdf is no      │
    │  longer queryable            │
    └──────────────────────────────┘
```

---

## 📊 Document Storage State

### **Timeline (Corrected)**

```
TIME 1: First retriever access, no uploads yet
┌─────────────────────────┐
│ FAISS Vectorstore       │
├─────────────────────────┤
│ ✓ Dummy (1)             │
│                         │
│ Total: 1 document       │
└─────────────────────────┘

TIME 2: Upload Python Guide
┌─────────────────────────┐
│ FAISS Vectorstore       │
├─────────────────────────┤
│ ✓ Python Guide (20)     │
│ ✗ Dummy — replaced      │
│                         │
│ Total: 20 documents     │
└─────────────────────────┘

TIME 3: Query
┌─────────────────────────┐
│ Search Results:         │
├─────────────────────────┤
│ Found in:               │
│ ✓ Python Guide chunks   │
│                         │
│ Return: Relevant chunks │
└─────────────────────────┘

TIME 4: Upload Web Dev Guide
┌─────────────────────────┐
│ FAISS Vectorstore       │
├─────────────────────────┤
│ ✓ Web Dev Guide (15)    │
│ ✗ Python Guide — REPLACED,│
│   no longer queryable   │
│                         │
│ Total: 15 documents     │
└─────────────────────────┘

TIME 5: Query Again
┌──────────────────────────┐
│ Can only search the      │
│ MOST RECENT upload:      │
├──────────────────────────┤
│ Search: "web development"│
│ Found in:                │
│ ✓ Web Dev Guide chunks   │
│ ✗ Python Guide           │
│   (no longer in store)   │
│                          │
│ Return: Web Dev matches  │
│ only                     │
└──────────────────────────┘
```

---

## 🔑 Key Takeaways

```
┌─────────────────────────────────────────────────┐
│ ACTUAL FLOW (verified against retriever_setup.py)│
├─────────────────────────────────────────────────┤
│                                                 │
│ 1. First access (no uploads):                   │
│    Create dummy → Vectorstore ready             │
│                                                 │
│ 2. Upload Document:                             │
│    Load → Split → REPLACE store with new chunks │
│                                                 │
│ 3. Query:                                       │
│    Search current store → LLM answer            │
│                                                 │
│ 4. Upload Another:                              │
│    REPLACES again — previous upload is gone     │
│                                                 │
│ RESULT: ⚠️ ONLY THE MOST RECENT DOCUMENT IS     │
│         SEARCHABLE AT ANY GIVEN TIME            │
│                                                 │
│ Want multiple documents searchable together?    │
│ → See DOCUMENT_UPLOAD_FLOW.md for the           │
│   `add_documents()` fix, or migrate to Qdrant.  │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Ready to Test!

You can now:

1. **Start the system**
   ```bash
   python -m uvicorn src.main:app --reload
   ```

2. **Upload your document**
   ```bash
   curl -X POST http://localhost:8000/rag/documents/upload \
     -H "X-Description: My Document" \
     -F "file=@my_file.pdf"
   ```

3. **Query it**
   ```bash
   curl -X POST http://localhost:8000/rag/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Your question", "session_id": "user_1"}'
   ```

4. **Get answers from YOUR document** — just remember that uploading a new one replaces it. ✅

---

**Status**: ✅ Single-document flow works correctly
**Multi-document support**: ⚠️ Not yet implemented (replace-on-upload, not accumulate)
**Ready to use**: Yes, for one active document at a time