"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho RAG mà KHÔNG cần vector store: nó xây "tree" theo cấu trúc tài
liệu (mục/chương/trang) rồi LLM duyệt cây để tìm node liên quan — reasoning-based
retrieval thay vì embedding similarity.

Cài đặt:
    pip install pageindex fpdf2

Lưu ý quan trọng:
    PageIndex CHỈ nhận file PDF. Tài liệu của ta đang ở dạng .md (data/standardized),
    nên bước upload sẽ gộp toàn bộ markdown thành 1 file PDF (font Unicode để hiển
    thị tiếng Việt) rồi submit. doc_id được cache lại để khỏi upload lại mỗi lần.

Quy trình API:
    submit_document(pdf) -> doc_id
    is_retrieval_ready(doc_id) -> poll tới khi True
    submit_query(doc_id, query) -> retrieval_id
    get_retrieval(retrieval_id) -> retrieved_nodes
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PAGEINDEX_DIR = Path(__file__).parent.parent / "data" / "pageindex"
CORPUS_PDF = PAGEINDEX_DIR / "drug_law_corpus.pdf"
DOC_ID_CACHE = PAGEINDEX_DIR / "doc_id.json"

# Font Unicode có sẵn trên macOS để render tiếng Việt trong PDF.
_FONT_CANDIDATES = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
]


def _client():
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _find_font() -> str:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    raise RuntimeError("Không tìm thấy font Unicode để tạo PDF tiếng Việt.")


def _build_corpus_pdf() -> Path:
    """Gộp toàn bộ markdown trong data/standardized/ thành 1 PDF Unicode."""
    from fpdf import FPDF

    PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
    font_path = _find_font()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("uni", "", font_path)

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        pdf.add_page()
        pdf.set_font("uni", size=14)
        # Tiêu đề mục giúp PageIndex dựng tree rõ ràng.
        pdf.multi_cell(0, 8, f"[{md_file.parent.name}] {md_file.stem}",
                       new_x="LMARGIN", new_y="NEXT", wrapmode="CHAR")
        pdf.ln(2)
        pdf.set_font("uni", size=11)
        text = md_file.read_text(encoding="utf-8")
        for para in text.split("\n"):
            para = para.strip()
            if para:
                # wrapmode=CHAR để bẻ được cả token dài (URL, bảng markdown).
                pdf.multi_cell(0, 6, para, new_x="LMARGIN", new_y="NEXT",
                               wrapmode="CHAR")
    pdf.output(str(CORPUS_PDF))
    print(f"  ✓ Tạo PDF: {CORPUS_PDF}")
    return CORPUS_PDF


def upload_documents(force: bool = False) -> str:
    """
    Tạo PDF từ corpus, upload lên PageIndex và đợi xử lý xong.
    Trả về doc_id (được cache lại trong data/pageindex/doc_id.json).
    """
    if not force and DOC_ID_CACHE.exists():
        doc_id = json.loads(DOC_ID_CACHE.read_text())["doc_id"]
        print(f"  ✓ Dùng doc_id đã cache: {doc_id}")
        return doc_id

    pi = _client()
    pdf_path = _build_corpus_pdf()

    print("  Uploading lên PageIndex...")
    resp = pi.submit_document(file_path=str(pdf_path))
    doc_id = resp["doc_id"]
    print(f"  ✓ doc_id = {doc_id}; đang chờ PageIndex xử lý (build tree)...")

    # Poll tới khi document sẵn sàng cho retrieval.
    deadline = time.monotonic() + 600  # tối đa 10 phút
    while time.monotonic() < deadline:
        if pi.is_retrieval_ready(doc_id):
            print("  ✓ Document đã sẵn sàng cho retrieval.")
            break
        time.sleep(10)
    else:
        print("  ⚠ Hết thời gian chờ — document có thể vẫn đang xử lý.")

    PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
    DOC_ID_CACHE.write_text(json.dumps({"doc_id": doc_id}))
    return doc_id


def _iter_content_dicts(relevant_contents):
    """relevant_contents có thể lồng (list-of-lists) → duyệt phẳng ra dict."""
    for item in relevant_contents or []:
        if isinstance(item, list):
            yield from _iter_content_dicts(item)
        elif isinstance(item, dict):
            yield item


def _format_nodes(retrieved_nodes: list[dict], top_k: int) -> list[dict]:
    """Chuẩn hoá retrieved_nodes của PageIndex về format chung."""
    results = []
    for rank, node in enumerate(retrieved_nodes[:top_k]):
        parts = [c.get("relevant_content", "")
                 for c in _iter_content_dicts(node.get("relevant_contents"))]
        text = "\n".join(p for p in parts if p) or node.get("text", "")
        results.append({
            "content": text,
            # PageIndex không trả score số; gán điểm giảm dần theo thứ hạng.
            "score": round(1.0 - rank * (1.0 / max(top_k, 1)), 4),
            "metadata": {
                "title": node.get("title", ""),
                "node_id": node.get("id") or node.get("node_id", ""),
            },
            "source": "pageindex",
        })
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval qua PageIndex. Dùng làm fallback ở Task 9.

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}
    """
    pi = _client()
    doc_id = upload_documents()  # tự dùng cache nếu đã upload

    sub = pi.submit_query(doc_id=doc_id, query=query)
    retrieval_id = sub["retrieval_id"]

    # Poll kết quả retrieval.
    deadline = time.monotonic() + 120
    result = {}
    while time.monotonic() < deadline:
        result = pi.get_retrieval(retrieval_id)
        if result.get("status") == "completed":
            break
        time.sleep(3)

    nodes = result.get("retrieved_nodes", [])
    return _format_nodes(nodes, top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng trái phép chất ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['metadata'].get('title')} :: "
                  f"{r['content'][:100]}...")
