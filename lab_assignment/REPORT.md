# Báo Cáo Bài Cá Nhân — RAG Pipeline v2 (Ngày 8)

**Chủ đề dữ liệu:** Pháp luật Việt Nam về ma tuý + các bài báo về nghệ sĩ liên quan tới ma tuý
**Người thực hiện:** tuyetnga09
**Ngày:** 2026-06-08
**Kết quả test:** ✅ **35/35 PASS** (`pytest tests/ -v`)

---

## Tổng Quan Kiến Trúc

```
                    ┌─────────────────────── INGESTION ───────────────────────┐
  Task 1 (legal .docx) ─┐
  Task 2 (news .json) ──┴─→ Task 3 (MarkItDown → .md) ─→ Task 4 (chunk + embed + index)
                                                              │
                                                       ChromaDB (cosine)
                    └──────────────────────────────────────────────────────────┘

                    ┌─────────────────────── RETRIEVAL ───────────────────────┐
  Query ─┬─→ Task 5 Semantic (dense)  ─┐
         ├─→ Task 6 Lexical (BM25)    ─┴─→ Task 7 RRF merge → Task 7 rerank ─┐
         │                                                                    ├─→ Task 9
         └─→ (nếu score < threshold) ─────────────→ Task 8 PageIndex fallback ┘
                                                              │
                                                       Task 10 Generation (citation)
                    └──────────────────────────────────────────────────────────┘
```

**Stack chính:** ChromaDB (vector store) · OpenAI `text-embedding-3-small` (embedding) · BM25Okapi (lexical) · Jina Reranker v2 (rerank) · PageIndex (vectorless fallback) · OpenAI `gpt-4o-mini` (generation).

> **Ghi chú môi trường:** Python 3.14.5. Vì là phiên bản rất mới, một số thư viện (Crawl4AI, torch) chưa có wheel ổn định → các task có **đường fallback** để vẫn chạy được end-to-end mà không cần cài đặt nặng.

---

## Task 1 — Thu Thập Văn Bản Pháp Luật ✅

**File:** `data/landing/legal/` (3 file `.docx`)

| File | Nội dung |
|------|----------|
| `luat-120-2025-QH15-phong-chong-ma-tuy.docx` | Luật Phòng, chống ma túy |
| `nghi-dinh-57-2022-ND-CP-danh-muc-chat-ma-tuy-tien-chat.docx` | Danh mục chất ma túy & tiền chất |
| `phap-lenh-01-2022-UBTVQH15-...-csncbb.docx` | Pháp lệnh đưa người nghiện vị thành niên vào cơ sở cai nghiện |

**Test:** `test_minimum_3_legal_files`, `test_files_not_empty`, `test_landing_legal_dir_exists` → PASS.

---

## Task 2 — Crawl Bài Báo ✅

**File code:** `src/task2_crawl_news.py` · **Output:** `data/landing/news/article_01..07.json` (7 bài, vượt yêu cầu ≥5)

- Crawl 7 bài từ **VnExpress, VietnamNet, Thanh Niên, Tuổi Trẻ, VOV, Báo Chính Phủ** về các vụ: Miu Lê, Long Nhật & Sơn Ngọc Minh, Nguyễn Công Trí, Hữu Tín, Châu Việt Cường, Chi Dân, Andrea Aybar...
- **Thiết kế:** ưu tiên **Crawl4AI** (theo đề); tự **fallback** sang `requests` + `html.parser` (thư viện chuẩn) khi Crawl4AI không cài được trên Python 3.14.
- Mỗi file JSON có metadata: `url`, `title`, `date_crawled`, `content_markdown`, `method`.

**Test:** 4/4 PASS (đủ ≥5 file, mỗi file >500 bytes, JSON có trường `url`).

---

## Task 3 — Convert Sang Markdown ✅

**File code:** `src/task3_convert_markdown.py` · **Output:** `data/standardized/legal/` (3) + `data/standardized/news/` (7) = 10 file `.md`

- **Legal (.docx):** dùng **MarkItDown** (`markitdown[docx]`) giữ cấu trúc heading/bảng; có **fallback** trích text qua `zipfile`+XML nếu thiếu dependency.
- **News (.json):** đọc `content_markdown`, gắn header metadata (`# title`, `**Source:**`, `**Crawled:**`).
- Giữ nguyên cấu trúc thư mục con `legal/` và `news/`.

**Test:** 4/4 PASS.

---

## Task 4 — Chunking & Indexing ✅

**File code:** `src/task4_chunking_indexing.py` · **Output:** ChromaDB tại `data/chroma/`, collection `drug_law_docs` (**392 chunks**)

| Hạng mục | Lựa chọn | Lý do |
|----------|----------|-------|
| Chunking | `RecursiveCharacterTextSplitter` | Tách theo đoạn→dòng→câu→từ, giữ ngữ nghĩa cho cả luật lẫn báo |
| `chunk_size` | **800** ký tự | Đủ chứa trọn 1 khoản/điều, không quá to làm loãng embedding |
| `chunk_overlap` | **120** (~15%) | Giữ ngữ cảnh ở ranh giới, tránh cắt ngang câu |
| Embedding | OpenAI `text-embedding-3-small` (**1536-d**) | Đa ngôn ngữ tốt cho tiếng Việt; gọi qua API → không cần cài torch nặng |
| Vector store | **ChromaDB** (cosine, persistent) | Đơn giản, local, không cần Docker (alternative hợp lệ theo đề) |

**Test:** 4/4 PASS.

---

## Task 5 — Semantic Search ✅

**File code:** `src/task5_semantic_search.py`

- `semantic_search(query, top_k)`: embed query bằng **cùng model Task 4** → query ChromaDB → đổi cosine distance sang similarity (`score = 1 - distance`).
- Trả về `[{content, score, metadata}]` sorted giảm dần. Cache collection + client bằng `lru_cache`.

**Test:** 4/4 PASS.

---

## Task 6 — Lexical Search (BM25) ✅

**File code:** `src/task6_lexical_search.py`

- `lexical_search(query, top_k)` dùng **BM25Okapi** (`k1=1.5, b=0.75`) trên **đúng tập chunk của Task 4** → dễ merge ở Task 9.
- Tokenize tiếng Việt: `re.findall(r"\w+", text.lower(), re.UNICODE)` (giữ dấu á/ử/đ). Index cache 1 lần. Chỉ trả về chunk `score > 0`.

**Test:** 4/4 PASS.

---

## Task 7 — Reranking ✅

**File code:** `src/task7_reranking.py` — implement cả **3 phương pháp**:

| Hàm | Cơ chế |
|-----|--------|
| `rerank_cross_encoder` (mặc định) | **Jina Reranker v2 multilingual** qua API; có **fallback offline** (token-overlap) khi lỗi/hết quota |
| `rerank_mmr` | **MMR**: `λ·sim(q,d) − (1−λ)·max sim(d, đã chọn)` — cân bằng relevance & diversity |
| `rerank_rrf` | **RRF**: `Σ 1/(k+rank)`, k=60 — gộp nhiều ranked list |

**Test:** 3/3 PASS.

---

## Task 8 — PageIndex Vectorless RAG ✅

**File code:** `src/task8_pageindex_vectorless.py` · **Output:** `data/pageindex/drug_law_corpus.pdf` + `doc_id.json` (cache)

- PageIndex **chỉ nhận PDF** → gộp 10 markdown thành 1 PDF bằng `fpdf2` + font **Arial Unicode** (tiếng Việt), `wrapmode="CHAR"` để bẻ token dài.
- Luồng: `submit_document → is_retrieval_ready (poll) → submit_query → get_retrieval (poll)`; `doc_id` được cache để khỏi upload lại.
- `_format_nodes` flatten cấu trúc `relevant_contents` (list lồng nhau) → `[{content, score, metadata, source:'pageindex'}]`.

**Test:** 2/2 PASS.

---

## Task 9 — Retrieval Pipeline Hoàn Chỉnh ✅

**File code:** `src/task9_retrieval_pipeline.py`

```
semantic (top_k*2) + lexical (top_k*2)
   → RRF merge (source="hybrid")
   → rerank cross-encoder
   → nếu best_score < SCORE_THRESHOLD (0.3) hoặc rỗng → fallback PageIndex
   → trả top_k
```

- RRF gộp theo **thứ hạng** nên không lệch do thang điểm khác nhau (cosine vs BM25).
- Fallback PageIndex có `try/except` → không crash khi API lỗi.

**Test:** 4/4 PASS (gồm `test_fallback_logic_exists` ép threshold=0.99 → kích hoạt PageIndex thật).

---

## Task 10 — Generation Có Citation ✅

**File code:** `src/task10_generation.py`

| Hạng mục | Lựa chọn | Lý do |
|----------|----------|-------|
| `top_k` | 5 | Đủ evidence, không quá dài gây lost-in-the-middle |
| `top_p` | 0.9 | Đủ đa dạng nhưng không quá random |
| `temperature` | 0.3 | RAG cần factual, ít sáng tạo |
| Model | `gpt-4o-mini` | Rẻ, nhanh, đủ tốt cho tiếng Việt |

- `reorder_for_llm` chống **"lost in the middle"**: tốt nhất ở đầu, tốt nhì ở cuối, kém ở giữa (`[1,2,3,4,5]→[1,3,5,4,2]`).
- `format_context` gắn nhãn `[Document N | Source | Type]` để LLM trích dẫn.
- `SYSTEM_PROMPT` bắt buộc citation `[Nguồn, Năm]`; thiếu evidence → *"Tôi không thể xác minh thông tin này từ nguồn hiện có."*

**Test:** 3/3 PASS. Demo trả lời có citation thật cho câu hỏi về nghệ sĩ bị bắt vì ma tuý.

---

## Kết Quả Test Tổng Hợp

```
pytest tests/ -v
================== 35 passed, 1 warning in ~1m44s ==================
```

| Task | Test | Trạng thái |
|------|------|-----------|
| 1 | 3 | ✅ |
| 2 | 4 | ✅ |
| 3 | 4 | ✅ |
| 4 | 4 | ✅ |
| 5 | 4 | ✅ |
| 6 | 4 | ✅ |
| 7 | 3 | ✅ |
| 8 | 2 | ✅ |
| 9 | 4 | ✅ |
| 10 | 3 | ✅ |
| **Tổng** | **35** | **✅ PASS** |

*1 warning vô hại: `DeprecationWarning` của chromadb trên Python 3.14 — không ảnh hưởng kết quả.*

---

## Lưu Ý / Việc Cần Làm Thêm

1. ⚠️ **Bảo mật:** `.env.example` đang chứa **API key thật** (OpenAI/Jina/PageIndex) và bị commit. Cần **rotate key** và thay bằng placeholder, đảm bảo `.env*` nằm trong `.gitignore`.
2. **Jina API** có lúc trả `403 Forbidden` (hết quota/rate-limit) → pipeline vẫn chạy nhờ fallback token-overlap.
3. **Dependencies bổ sung** đã cập nhật vào `requirements.txt`: `markitdown[docx]`, `fpdf2`.
4. **Cách chạy demo từng module:** dùng dạng module để import `src.` hoạt động, ví dụ:
   ```bash
   python3 -m src.task5_semantic_search
   python3 -m src.task9_retrieval_pipeline
   ```
