"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai

Ghi chú triển khai:
    Script ưu tiên dùng Crawl4AI (theo gợi ý của đề bài). Nếu môi trường
    không cài được Crawl4AI / Playwright (ví dụ Python quá mới), script tự
    động fallback sang `requests` + bộ parse HTML của thư viện chuẩn để vẫn
    crawl được nội dung. Cả hai đường đều xuất ra cùng một format JSON.
"""

import asyncio
import json
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách URL bài báo về nghệ sĩ Việt Nam liên quan tới ma tuý.
# Nguồn: VnExpress, VietnamNet, Thanh Niên, Tuổi Trẻ, VOV, Báo Chính Phủ.
ARTICLE_URLS = [
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://vietnamnet.vn/ngoai-nguyen-cong-tri-nhung-nghe-si-nao-tung-bi-bat-vi-ma-tuy-2424971.html",
    "https://vietnamnet.vn/sao-viet-bi-bat-ngoi-tu-mat-danh-tieng-vi-chat-cam-2513746.html",
    "https://thanhnien.vn/miu-le-va-loi-xin-loi-muon-mang-cua-loat-sao-viet-vuong-vao-ma-tuy-18526051513021689.htm",
    "https://tuoitre.vn/khoi-to-3-bi-can-trong-vu-ca-si-miu-le-su-dung-ma-tuy-o-cat-ba-20260514230349573.htm",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://baochinhphu.vn/khoi-to-bat-tam-giam-ca-si-long-nhat-son-ngoc-minh-vi-to-chuc-su-dung-ma-tuy-102260520125739676.htm",
]

# User-Agent giả lập trình duyệt để tránh bị chặn bởi một số báo điện tử.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Fallback parser: requests + html.parser (không cần Crawl4AI)
# ---------------------------------------------------------------------------
class _ArticleHTMLParser(HTMLParser):
    """Trích tiêu đề và phần text trong các thẻ <p>/<h1>/<h2> của bài báo."""

    _SKIP_TAGS = {"script", "style", "noscript", "header", "footer", "nav", "aside"}
    _TEXT_TAGS = {"p", "h1", "h2", "h3", "li"}

    def __init__(self):
        super().__init__()
        self.title = ""
        self._og_title = ""
        self._in_title = False
        self._skip_depth = 0
        self._capture_depth = 0
        self._buf = []
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            ad = dict(attrs)
            if ad.get("property") == "og:title" and ad.get("content"):
                self._og_title = ad["content"]
            if ad.get("name") in ("title", "description") and ad.get("content") and not self._og_title:
                self._og_title = self._og_title or ad["content"]
        elif tag in self._TEXT_TAGS and self._skip_depth == 0:
            self._capture_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self._TEXT_TAGS and self._capture_depth > 0:
            self._capture_depth -= 1
            text = " ".join("".join(self._buf).split())
            if text:
                self.blocks.append(text)
            self._buf = []

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._capture_depth > 0 and self._skip_depth == 0:
            self._buf.append(data)

    def best_title(self):
        return (self._og_title or self.title).strip()


def _html_to_article(url: str, html: str) -> dict:
    """Chuyển HTML thô thành dict bài báo (dùng cho đường fallback)."""
    parser = _ArticleHTMLParser()
    parser.feed(html)
    # Giữ các đoạn có độ dài hợp lý để loại bỏ menu/nút bấm rời rạc.
    paragraphs = [b for b in parser.blocks if len(b) > 25]
    content = "\n\n".join(paragraphs)
    return {
        "url": url,
        "title": parser.best_title() or "Unknown",
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
        "method": "requests+htmlparser",
    }


async def _crawl_with_requests(url: str) -> dict:
    """Fallback: tải trang bằng requests rồi parse bằng thư viện chuẩn."""
    import requests

    def _fetch():
        resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=30)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text

    html = await asyncio.to_thread(_fetch)
    return _html_to_article(url, html)


async def _crawl_with_crawl4ai(url: str) -> dict:
    """Đường chính: dùng Crawl4AI (AsyncWebCrawler)."""
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        metadata = getattr(result, "metadata", None) or {}
        return {
            "url": url,
            "title": metadata.get("title", "Unknown"),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown or "",
            "method": "crawl4ai",
        }


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str,
            "method": str,
        }
    """
    try:
        import crawl4ai  # noqa: F401
    except ImportError:
        return await _crawl_with_requests(url)
    return await _crawl_with_crawl4ai(url)


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    saved = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:  # bỏ qua bài lỗi, tiếp tục các bài còn lại
            print(f"  ✗ Lỗi: {exc}")
            continue

        if len(article.get("content_markdown", "")) < 200:
            print("  ⚠ Nội dung quá ngắn, có thể bị chặn — vẫn lưu lại để kiểm tra.")

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath} ({len(article['content_markdown'])} chars, {article['method']})")
        saved += 1

    print(f"\nHoàn tất: lưu {saved}/{len(ARTICLE_URLS)} bài báo vào {DATA_DIR}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
