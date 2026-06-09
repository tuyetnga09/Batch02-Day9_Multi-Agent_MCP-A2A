# Lab Solution — Day09 Multi-Agent (A2A) & Tích Hợp Day08 (RAG)

**Người thực hiện:** tuyetnga09
**Ngày:** 2026-06-09
**Phạm vi:** Hoàn thành codelab Day09 (multi-agent A2A) + bài cộng điểm, và đề xuất/thiết kế việc áp dụng pipeline RAG của Day08 vào hệ multi-agent Day09.

---

## Phần A — Đã giải quyết trong buổi (Day09)

### A0. Khởi tạo môi trường
- Cài `uv`, `uv sync` (71 packages).
- Sửa `.env`: `REGISTRY_URL` từ `:3000` → **`:10000`** (sai port làm Stage 5 không discover được).
- Vì tài khoản OpenRouter hết credit (lỗi `402`), chuyển `OPENROUTER_MODEL` sang model free **`openai/gpt-oss-120b:free`** và thêm `OPENROUTER_MAX_TOKENS` để chặn việc giữ chỗ 64k token.

### A1. Stage 1–4 (các bài tập trong CODELAB.md)

| Bài | Nội dung | File |
|---|---|---|
| 1.1 | Đổi câu hỏi sang tình huống luật lao động VN | `stages/stage_1_direct_llm/main.py` |
| 1.2 | Thêm `temperature` (mặc định 0.3) | `common/llm.py` |
| 2.1 | Thêm entry knowledge base luật lao động | `exercises/exercise_2_tools.py` |
| 2.2 | Tool `check_statute_of_limitations` | `exercises/exercise_2_tools.py` |
| 3.1 | Tool `search_case_law` (tra án lệ) | `stages/stage_3_single_agent/main.py` |
| 4.1 | Implement `privacy_agent` (GDPR) | `exercises/exercise_4_multiagent.py` |
| 4.2 | Conditional routing cho privacy agent | `exercises/exercise_4_multiagent.py` |

> **Bug đã sửa:** skeleton `exercise_4` wire sai graph — `check_routing` trả `list[Send]` nhưng bị đăng ký làm node thường → `InvalidUpdateError`. Đã sửa thành dùng `add_conditional_edges("law_agent", check_routing)` (giống stage 4).

Tất cả Stage 1–4 + 2 exercise đã chạy verify ra kết quả thật.

### A2. Bài cộng điểm — Mục 1: HTML demo tương tác Agent
- **`demo.html`** (self-contained, mở bằng trình duyệt): animate luồng Stage 5
  `User → Customer → Registry → Law → routing → Tax ∥ Compliance ∥ analyze_law → aggregate`,
  có công tắc Baseline/Optimized và nút **"⏱️ Đo thật"** gọi LIVE hệ thống.
- Đã thêm **CORS + endpoint REST `POST /ask`** vào `customer_agent/__main__.py` để trình duyệt
  gọi được hệ thống thật và đo latency (dùng `app.add_route` kiểu Starlette để tránh
  dependency-injection của FastAPI làm hiểu nhầm body thành query param).

### A3. Bài cộng điểm — Mục 2: Latency & tối ưu

| | Latency | Ghi chú |
|---|---|---|
| **Baseline** | **420.0s** | LLM routing + `analyze_law` nối tiếp |
| **Optimized** | **299.3s** | keyword routing + `analyze_law` song song |
| **Giảm** | **~121s (≈29%)** | routing: ~102s → ~2ms |

Hai tối ưu trong `law_agent/graph.py`:
1. **Keyword routing** thay LLM trong `check_routing` (bỏ 1 LLM call nối tiếp).
2. **Song song hoá `analyze_law`** với Tax/Compliance (output chỉ dùng ở `aggregate`).
→ Critical path bên Law Agent giảm từ **4 tầng nối tiếp** xuống **2 tầng**.

Công cụ: **`measure_latency.py`**. Đã xác nhận đo live qua `/ask`: ~283s, response 8376 ký tự.
Chi tiết đầy đủ: xem **`BAI_CONG_DIEM.md`**.

### A4. Dọn dẹp git
- `.gitignore` đã bảo vệ `.env` (API key **không** bị commit) + thêm `.DS_Store`.
- Day09 + bonus đã được commit.

---

## Phần B — Áp dụng Day08 (RAG) vào Day09 (Multi-Agent)

### B1. Hai hệ giải quyết hai nửa của cùng một bài toán

| | **Day08 — RAG Pipeline** | **Day09 — Multi-Agent A2A** |
|---|---|---|
| Vai trò | *Trả lời ĐÚNG dựa trên tài liệu thật* | *Điều phối nhiều chuyên gia, chạy phân tán* |
| Mạnh | Retrieval hybrid (semantic + BM25 + rerank + PageIndex fallback), generation có **citation** | Orchestration, song song hoá, discovery động, A2A |
| Yếu | Chỉ 1 luồng hỏi-đáp, không đa agent | Knowledge base **đồ chơi** (keyword match), dễ "chế" thông tin |

→ **Điểm khớp:** Day09 thiếu nguồn tri thức thật; Day08 chính là *retrieval backend* lý tưởng. Gắn Day08 vào Day09 biến mỗi specialist agent từ "trả lời theo trí nhớ LLM" thành "trả lời có dẫn nguồn từ văn bản pháp luật thật".

### B2. Kiến trúc tích hợp đề xuất

Thay phần knowledge "đồ chơi" trong các agent Day09 bằng pipeline retrieval của Day08:

```
Day09 Law/Tax/Compliance Agent
        │  (thay vì search_legal_database keyword-match)
        ▼
Day08  retrieve(query)  =  Task 9 retrieval_pipeline
        │   semantic (Chroma) + lexical (BM25) → RRF → rerank
        │   → fallback PageIndex nếu score thấp
        ▼
   [{content, score, metadata, source}]
        │
        ▼
Day09 agent đưa context này vào prompt → trả lời CÓ CITATION (như Task 10)
```

**Cụ thể về code:** trong `law_agent/graph.py` (và tax/compliance), tool/nút phân tích sẽ gọi
hàm retrieval của Day08:

```python
# pseudo — tích hợp Day08 vào 1 agent Day09
from day08.src.task9_retrieval_pipeline import retrieve   # hybrid retrieval

async def analyze_law(state):
    docs = retrieve(state["question"], top_k=5)            # Day08 RAG
    context = format_context(docs)                          # gắn [Document N | Source]
    messages = [SystemMessage(content=PROMPT_BAT_BUOC_CITATION),
                HumanMessage(content=f"{context}\n\nCâu hỏi: {state['question']}")]
    result = await llm.ainvoke(messages)
    return {"law_analysis": result.content}                 # câu trả lời có dẫn nguồn
```

### B3. Lộ trình thực hiện (các bước)

1. **Đóng gói Day08 thành thư viện gọi được:** expose `retrieve(query, top_k)` (đã có sẵn ở `task9_retrieval_pipeline.py`) và `generate_with_citation()` (Task 10).
2. **Mỗi specialist agent có corpus riêng:** Day08 hiện index dữ liệu *ma tuý*; mở rộng để có collection cho *thuế*, *compliance*, *hợp đồng* → mỗi agent retrieve trên collection phù hợp.
3. **Thay `search_legal_database`** (keyword đồ chơi) bằng tool gọi `retrieve()` của Day08.
4. **Bắt buộc citation** ở bước `aggregate`: tổng hợp kèm `[Nguồn, Năm]` như SYSTEM_PROMPT của Task 10.
5. **Đo lại latency:** retrieval thêm ~1 bước I/O; cân nhắc cache embedding/kết quả để giữ latency.

### B4. Trạng thái

- ✅ **Đã phân tích & thiết kế** điểm tích hợp (tài liệu này).
- ⏳ **Chưa nối code thật** giữa 2 repo — vì Day08 là **repo GitHub riêng**
  (`github.com/tuyetnga09/Day08_RAG_pipeline_cohort2`), chỉ tình cờ nằm trong thư mục Day09.
- Khi triển khai: nên đưa Day08 vào dạng **package/submodule** thay vì copy, để giữ liên kết repo gốc.

---

## Ghi chú & cảnh báo bảo mật

1. ⚠️ **Day08:** theo `REPORT.md`, `.env.example` của Day08 từng **chứa API key thật** và bị commit → **cần rotate key** (OpenAI/Jina/PageIndex) và thay bằng placeholder.
2. **Day09:** `.env` (OpenRouter key) đã nằm trong `.gitignore` → an toàn.
3. **Model free** của Day09 chậm/bị queue → latency tuyệt đối cao; muốn nhanh & ổn định nên nạp credit hoặc đổi model trả phí.

---

## Tham chiếu

- Day09 bonus chi tiết: `BAI_CONG_DIEM.md`
- Demo tương tác: `demo.html`
- Đo latency: `measure_latency.py`
- Day08 báo cáo: `lab_assignment/Day08_RAG_pipeline_cohort2/REPORT.md`
