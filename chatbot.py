"""
chatbot.py — CLI chatbot for the law firm RAG system.

For every user query it:
  1. Embeds the query with sentence-transformers
  2. Retrieves the top-k most relevant chunks from ChromaDB
  3. Calls Claude twice (legal mode + layman mode) via the Anthropic SDK
  4. Prints both responses clearly labelled

Usage:
    python chatbot.py
"""

import os
import sys
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from prompts import LEGAL_SYSTEM_PROMPT, LAYMAN_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "legal_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# Claude model — explicitly requested by the user
CLAUDE_MODEL = "claude-sonnet-4-6"

# Token budgets (per the prompt-engineer's recommendations)
LEGAL_MAX_TOKENS = 2000
LAYMAN_MAX_TOKENS = 900

NO_CONTEXT_MESSAGE = (
    "No relevant document passages were found for your query. "
    "Please make sure you have ingested documents with `python ingest.py` "
    "and that the query relates to the ingested content."
)


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def build_retriever():
    """Connect to ChromaDB and return (collection, embedder)."""
    if not CHROMA_DIR.exists():
        print(
            f"[ERROR] ChromaDB directory not found at {CHROMA_DIR}.\n"
            "Run `python ingest.py` first."
        )
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(
            f"[ERROR] Collection '{COLLECTION_NAME}' not found.\n"
            "Run `python ingest.py` first."
        )
        sys.exit(1)

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    return collection, embedder


def build_anthropic_client() -> anthropic.Anthropic:
    """Return an Anthropic client, failing early if the API key is missing."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "[ERROR] ANTHROPIC_API_KEY not set.\n"
            "Create a .env file (see .env.example) or export the variable."
        )
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(query: str, collection, embedder, top_k: int = TOP_K) -> list[dict]:
    """
    Embed `query`, query ChromaDB, and return up to `top_k` result dicts.
    Each dict has keys: 'text', 'source', 'chunk_index'.
    """
    query_embedding = embedder.encode([query], convert_to_list=True)[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
        })
    return chunks


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into the {context} placeholder string."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Passage {i} | Source: {chunk['source']} | Chunk: {chunk['chunk_index']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def call_claude(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
) -> str:
    """Send a single-turn message to Claude and return the text response."""
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def get_dual_response(
    client: anthropic.Anthropic,
    context: str,
    question: str,
) -> tuple[str, str]:
    """
    Call Claude twice — once for legal mode, once for layman mode.
    Returns (legal_response, layman_response).
    """
    user_message = USER_PROMPT_TEMPLATE.format(context=context, question=question)

    legal_response = call_claude(
        client,
        system_prompt=LEGAL_SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=LEGAL_MAX_TOKENS,
    )

    layman_response = call_claude(
        client,
        system_prompt=LAYMAN_SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=LAYMAN_MAX_TOKENS,
    )

    return legal_response, layman_response


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_separator(char: str = "-", width: int = 72):
    print(char * width)


def print_responses(legal: str, layman: str):
    print()
    print_separator("=")
    print("[LEGAL RESPONSE]")
    print_separator("=")
    print(legal)
    print()
    print_separator("=")
    print("[LAYMAN RESPONSE]")
    print_separator("=")
    print(layman)
    print_separator("=")
    print()


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def main():
    print("Initialising Law Firm RAG Chatbot...")
    collection, embedder = build_retriever()
    anthropic_client = build_anthropic_client()

    doc_count = collection.count()
    print(f"Connected to ChromaDB. {doc_count} chunk(s) available.")
    print(f"Model: {CLAUDE_MODEL} | Top-k retrieval: {TOP_K}")
    print("\nType your legal question below. Type 'quit' or 'exit' to stop.\n")
    print_separator()

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        # Retrieval
        chunks = retrieve_context(query, collection, embedder)

        if not chunks:
            print(f"\n[INFO] {NO_CONTEXT_MESSAGE}\n")
            continue

        print(f"\n[INFO] Retrieved {len(chunks)} passage(s). Querying Claude...\n")

        # Generation
        try:
            context = format_context(chunks)
            legal_resp, layman_resp = get_dual_response(anthropic_client, context, query)
            print_responses(legal_resp, layman_resp)
        except anthropic.APIStatusError as e:
            print(f"\n[ERROR] Anthropic API error {e.status_code}: {e.message}\n")
        except anthropic.APIConnectionError:
            print("\n[ERROR] Could not connect to the Anthropic API. Check your internet connection.\n")
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}\n")


if __name__ == "__main__":
    main()
