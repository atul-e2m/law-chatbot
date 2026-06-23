"""
ingest.py — Load legal documents from law_chatbot/docs/, chunk them,
embed with sentence-transformers, and store in ChromaDB.

Usage:
    python ingest.py
"""

import os
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "legal_docs"
CHUNK_SIZE = 500        # tokens (approximated as words for simplicity)
CHUNK_OVERLAP = 50      # tokens of overlap between consecutive chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, local, no API key required


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_txt(path: Path) -> str:
    """Read a plain-text file and return its content."""
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf(path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def load_document(path: Path) -> str:
    """Dispatch to the right loader based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix == ".txt":
        return load_txt(path)
    else:
        print(f"  [SKIP] Unsupported file type: {path.name}")
        return ""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks.

    Approximates tokens as whitespace-split words.  Each chunk is
    `chunk_size` words long; consecutive chunks share `overlap` words.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap   # slide forward, keeping overlap

    return chunks


# ---------------------------------------------------------------------------
# Main ingestion routine
# ---------------------------------------------------------------------------

def ingest():
    """Ingest all supported documents found in DOCS_DIR into ChromaDB."""

    # Collect documents
    supported_extensions = {".pdf", ".txt"}
    doc_paths = [
        p for p in DOCS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]

    if not doc_paths:
        print(
            f"No documents found in {DOCS_DIR}.\n"
            "Place .pdf or .txt files there and re-run ingest.py."
        )
        sys.exit(0)

    print(f"Found {len(doc_paths)} document(s) in {DOCS_DIR}")

    # Load and chunk
    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []

    for doc_path in doc_paths:
        print(f"  Loading: {doc_path.name}")
        text = load_document(doc_path)
        if not text.strip():
            print(f"  [WARN] Empty content extracted from {doc_path.name}")
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"    -> {len(chunks)} chunk(s)")

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_path.stem}__chunk_{idx:04d}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source": doc_path.name,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            })

    if not all_chunks:
        print("No text could be extracted. Nothing to ingest.")
        sys.exit(0)

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    # Embed
    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads ~80 MB)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding chunks...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True, convert_to_list=True)

    # Store in ChromaDB
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Drop existing collection so re-runs start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB upsert in batches to stay within default limits
    BATCH = 500
    for i in range(0, len(all_chunks), BATCH):
        collection.add(
            ids=all_ids[i:i + BATCH],
            documents=all_chunks[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
            metadatas=all_metadatas[i:i + BATCH],
        )

    print(
        f"\nIngestion complete. {len(all_chunks)} chunks stored in "
        f"ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR}."
    )


if __name__ == "__main__":
    ingest()
