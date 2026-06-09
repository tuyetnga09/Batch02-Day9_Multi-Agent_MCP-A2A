"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Triển khai:
    - Embed query bằng cùng model OpenAI text-embedding-3-small (Task 4).
    - Query ChromaDB collection 'drug_law_docs' bằng cosine similarity.
    - Chroma trả về cosine *distance*; ta đổi sang similarity = 1 - distance.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

from src.task4_chunking_indexing import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

load_dotenv()


@lru_cache(maxsize=1)
def _get_collection():
    """Mở (cache) ChromaDB collection đã index ở Task 4."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


@lru_cache(maxsize=1)
def _get_openai():
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _embed_query(query: str) -> list[float]:
    resp = _get_openai().embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return resp.data[0].embedding


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    collection = _get_collection()
    query_embedding = _embed_query(query)

    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = res["documents"][0]
    metadatas = res["metadatas"][0]
    distances = res["distances"][0]

    results = [
        {
            "content": doc,
            "score": 1.0 - dist,  # cosine distance → similarity
            "metadata": meta,
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
    # Chroma trả về sẵn theo distance tăng dần; sort lại cho chắc chắn.
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] ({r['metadata'].get('type')}/"
              f"{r['metadata'].get('source')}) {r['content'][:100]}...")
