"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store

Cài đặt:
    pip install langchain-text-splitters chromadb openai python-dotenv
"""

import os
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME = "drug_law_docs"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# Chunking: RecursiveCharacterTextSplitter
#   - Vì sao "recursive": an toàn & phổ biến nhất, tách theo thứ tự ưu tiên
#     đoạn → dòng → câu → từ nên giữ được ngữ nghĩa tốt cho cả văn bản pháp
#     luật (cấu trúc Điều/Khoản) lẫn bài báo.
#   - CHUNK_SIZE = 800 ký tự: văn bản pháp luật có các "Điều" khá dài; 800
#     đủ chứa trọn một khoản/ý mà không quá to làm loãng embedding.
#   - CHUNK_OVERLAP = 120 (~15%): giữ ngữ cảnh ở ranh giới chunk, tránh cắt
#     ngang câu làm mất nghĩa khi retrieval.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding: OpenAI text-embedding-3-small
#   - Vì sao: chất lượng đa ngôn ngữ tốt (kể cả tiếng Việt), gọi qua API nên
#     không cần cài torch/sentence-transformers nặng (phù hợp môi trường này).
#   - dimension = 1536.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Vector store: ChromaDB (local persistent, đơn giản, không cần Docker)
VECTOR_STORE = "chromadb"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            if not chunk_text.strip():
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng OpenAI embeddings.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    texts = [c["content"] for c in chunks]
    embeddings = []
    batch_size = 100  # batch để giảm số request và tránh vượt giới hạn token
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend([d.embedding for d in resp.data])
        print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}")

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào ChromaDB (persistent, local)."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Xoá collection cũ để index lại sạch sẽ (idempotent).
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    print(f"  ✓ Collection '{COLLECTION_NAME}' có {collection.count()} chunks "
          f"tại {CHROMA_DIR}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
