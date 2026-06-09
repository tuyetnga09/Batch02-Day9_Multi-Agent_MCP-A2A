"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Triển khai:
    Corpus dùng đúng các chunk đã tạo ở Task 4 (load_documents + chunk_documents)
    để lexical search và semantic search cùng làm việc trên một tập chunk → dễ
    merge ở Task 9.
"""

import re
from functools import lru_cache

from src.task4_chunking_indexing import chunk_documents, load_documents


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho tiếng Việt: lowercase + tách theo ký tự chữ/số.

    Dùng \\w với re.UNICODE để giữ nguyên ký tự có dấu tiếng Việt
    (á, ử, đ, ...). Đủ tốt cho BM25 vì BM25 so khớp theo token.
    """
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


@lru_cache(maxsize=1)
def _build():
    """Load corpus (các chunk Task 4) và dựng BM25 index. Cache 1 lần."""
    from rank_bm25 import BM25Okapi

    corpus = chunk_documents(load_documents())
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)  # k1=1.5, b=0.75 mặc định
    return bm25, corpus


# Giữ tương thích với chữ ký gốc của đề bài.
CORPUS: list[dict] = []


def build_bm25_index(corpus: list[dict]):
    """Xây dựng BM25 index từ corpus tuỳ ý (theo chữ ký đề bài)."""
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending. Chỉ trả về chunk có score > 0.
    """
    bm25, corpus = _build()
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        if score <= 0:
            continue
        results.append({
            "content": corpus[idx]["content"],
            "score": float(score),
            "metadata": corpus[idx]["metadata"],
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] ({r['metadata'].get('type')}/"
              f"{r['metadata'].get('source')}) {r['content'][:100]}...")
