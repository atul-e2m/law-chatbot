"""
ingest.py — Load legal documents from law_chatbot/docs/, chunk them,
embed with sentence-transformers, and store in a FAISS index.

Usage:
    python ingest.py
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOCS_DIR = Path(__file__).parent / "docs"
FAISS_DIR = Path(__file__).parent / "faiss_db"
INDEX_FILE = FAISS_DIR / "index.faiss"
META_FILE = FAISS_DIR / "index.pkl"
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
    """Ingest all supported documents found in DOCS_DIR into a FAISS index."""

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
            all_chunks.append(chunk)
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
    embeddings = embedder.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

    # Normalise for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    # Build FAISS index (inner product on L2-normalised vectors = cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Persist index and metadata
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadatas": all_metadatas}, f)

    print(
        f"\nIngestion complete. {len(all_chunks)} chunks stored in "
        f"FAISS index at {FAISS_DIR}."
    )


if __name__ == "__main__":
    ingest()
