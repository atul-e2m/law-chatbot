# Law Firm RAG Chatbot

A production-ready CLI chatbot that answers legal questions using Retrieval-Augmented Generation (RAG). It ingests your firm's legal documents (PDFs and text files), retrieves the most relevant passages, and returns **two responses** per query: a formal legal memo and a plain-English explanation.

## Architecture

```
law_chatbot/
├── docs/            # Drop your .pdf and .txt files here
├── chroma_db/       # ChromaDB vector store (auto-created on first ingest)
├── ingest.py        # Document ingestion pipeline
├── chatbot.py       # CLI chat loop
├── prompts.py       # System prompts and user prompt template
├── requirements.txt
├── .env             # Your API key (create from .env.example)
└── .env.example
```

- **LLM**: Claude `claude-sonnet-4-6` via the Anthropic SDK
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers (local, no API key needed)
- **Vector store**: ChromaDB (persistent, on-disk)
- **Document loading**: pypdf (PDFs) + plain text (.txt)
- **Chunking**: 500-word chunks, 50-word overlap
- **Retrieval**: Top-5 cosine similarity

## Setup

### 1. Install dependencies

```bash
cd law_chatbot
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
cp .env.example .env
# Edit .env and replace `your_key_here` with your actual key
```

### 3. Add legal documents

Place `.pdf` or `.txt` files inside `law_chatbot/docs/`.

### 4. Ingest documents

```bash
python ingest.py
```

This embeds all documents and stores them in ChromaDB. Re-run any time you add new documents.

### 5. Start the chatbot

```bash
python chatbot.py
```

## Example Session

```
You: What are the termination clauses in the employment agreement?

[LEGAL RESPONSE]
ISSUE: ...
RULE: ...
ANALYSIS: ...
CONCLUSION: ...

[LAYMAN RESPONSE]
In plain terms, the agreement says you can be let go if...
```

Type `quit` or `exit` to stop.

## Notes

- The embedding model downloads ~80 MB on first run (cached automatically).
- Re-running `ingest.py` drops and recreates the collection — useful for updates.
- The chatbot gracefully handles queries that find no relevant passages.
