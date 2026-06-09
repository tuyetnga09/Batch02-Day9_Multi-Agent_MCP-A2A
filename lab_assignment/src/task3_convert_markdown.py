"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục

Ghi chú triển khai:
    - File pháp luật (.docx/.pdf) → dùng MarkItDown. Nếu MarkItDown không
      khả dụng, fallback sang bộ trích text .docx bằng thư viện chuẩn
      (docx thực chất là file zip chứa word/document.xml).
    - File báo (.json từ Task 2) → đọc trực tiếp content_markdown và gắn
      thêm metadata header (title, source URL, ngày crawl).
"""

import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _docx_to_markdown_fallback(filepath: Path) -> str:
    """Trích text từ .docx bằng thư viện chuẩn (khi không có MarkItDown).

    .docx là một file zip; nội dung nằm ở word/document.xml. Ta đọc từng
    paragraph (<w:p>) và nối các đoạn text (<w:t>) bên trong.
    """
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(filepath) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)

    lines = []
    for para in root.iter(f"{ns}p"):
        texts = [node.text for node in para.iter(f"{ns}t") if node.text]
        line = "".join(texts).strip()
        if line:
            lines.append(line)
    return "\n\n".join(lines)


def _convert_doc(md, filepath: Path) -> str:
    """Convert 1 file pháp luật sang markdown, ưu tiên MarkItDown."""
    if md is not None:
        try:
            return md.convert(str(filepath)).text_content
        except Exception as exc:
            print(f"  ⚠ MarkItDown lỗi ({exc}); dùng fallback.")
    if filepath.suffix.lower() == ".docx":
        return _docx_to_markdown_fallback(filepath)
    raise RuntimeError(f"Không convert được {filepath.name} (cần MarkItDown cho PDF).")


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from markitdown import MarkItDown
        md = MarkItDown()
    except ImportError:
        md = None
        print("  ⚠ Chưa cài MarkItDown — dùng fallback trích text .docx.")

    count = 0
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            text = _convert_doc(md, filepath)
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(text, encoding="utf-8")
            print(f"  ✓ Saved: {output_path} ({len(text)} chars)")
            count += 1
    print(f"--- Legal: {count} file(s) ---")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))

            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            content = header + data.get("content_markdown", "")
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path} ({len(content)} chars)")
            count += 1
    print(f"--- News: {count} file(s) ---")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
