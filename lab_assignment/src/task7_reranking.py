"""
Task 7 — Reranking Module.

Triển khai cả 3 phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) — MẶC ĐỊNH.
    - MMR (Maximal Marginal Relevance): tự implement.
    - RRF (Reciprocal Rank Fusion): tự implement.

Cross-encoder mặc định gọi Jina API. Nếu không có JINA_API_KEY hoặc lỗi mạng,
tự fallback sang reranker offline dựa trên độ trùng lặp token (token overlap)
để module luôn chạy được.
"""

import math
import os
import re

from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_MODEL = "jina-reranker-v2-base-multilingual"
JINA_URL = "https://api.jina.ai/v1/rerank"


def _tokens(text: str) -> set:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _token_overlap_scores(query: str, candidates: list[dict]) -> list[float]:
    """Reranker offline: tỉ lệ token query xuất hiện trong document (Jaccard-ish).

    Dùng làm fallback khi Jina API không khả dụng.
    """
    q = _tokens(query)
    if not q:
        return [0.0] * len(candidates)
    scores = []
    for c in candidates:
        d = _tokens(c["content"])
        overlap = len(q & d) / len(q)            # bao nhiêu % từ khoá query xuất hiện
        coverage = len(q & d) / len(q | d) if (q | d) else 0.0  # Jaccard
        scores.append(0.7 * overlap + 0.3 * coverage)
    return scores


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates bằng cross-encoder (Jina Reranker v2 multilingual).

    Returns:
        Top_k candidates, re-scored & sorted by relevance descending.
    """
    if not candidates:
        return []

    documents = [c["content"] for c in candidates]

    # --- Đường chính: Jina Reranker API ---
    if JINA_API_KEY:
        try:
            import requests

            resp = requests.post(
                JINA_URL,
                headers={"Authorization": f"Bearer {JINA_API_KEY}"},
                json={
                    "model": JINA_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_n": top_k,
                },
                timeout=30,
            )
            resp.raise_for_status()
            reranked = resp.json()["results"]
            return [
                {**candidates[r["index"]], "score": float(r["relevance_score"])}
                for r in reranked
            ]
        except Exception as exc:
            print(f"  ⚠ Jina rerank lỗi ({exc}); dùng fallback token-overlap.")

    # --- Fallback offline ---
    scores = _token_overlap_scores(query, candidates)
    rescored = [{**c, "score": s} for c, s in zip(candidates, scores)]
    rescored.sort(key=lambda x: x["score"], reverse=True)
    return rescored[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidate vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Yêu cầu mỗi candidate có key 'embedding'.
    """
    selected, remaining = [], list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx, best_score = None, float("-inf")
        for idx in remaining:
            emb = candidates[idx]["embedding"]
            relevance = _cosine(query_embedding, emb)
            max_sim = max(
                (_cosine(emb, candidates[s]["embedding"]) for s in selected),
                default=0.0,
            )
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score, best_idx = mmr, idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [{**candidates[i], "score": _cosine(query_embedding, candidates[i]["embedding"])}
            for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp nhiều ranked list (vd: semantic + lexical).

    RRF(d) = Σ_r 1 / (k + rank_r(d)),  k=60 (Cormack et al. 2009)
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for content, score in sorted_items[:top_k]:
        results.append({**content_map[content], "score": score})
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """Unified reranking interface. Mặc định dùng cross-encoder (Jina)."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        raise NotImplementedError("Gọi rerank_mmr() trực tiếp với query_embedding")
    elif method == "rrf":
        raise NotImplementedError("Gọi rerank_rrf() trực tiếp với ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
        {"content": "Hướng dẫn lập trình Python", "score": 0.5, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
