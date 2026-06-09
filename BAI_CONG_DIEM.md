# Bài Tập Cộng Điểm — Multi-Agent A2A

> Model dùng để đo: `openai/gpt-oss-120b:free` (OpenRouter, free tier).
> Câu hỏi test: *"If a company breaks a contract and avoids taxes, what are the legal and regulatory consequences?"*

---

## Mục 1 — File HTML demo tương tác các Agent

**File:** [`demo.html`](demo.html) — mở trực tiếp bằng trình duyệt, không cần server.

Demo mô phỏng trung thực luồng Stage 5 (theo log thật của hệ thống):

- Nhập câu hỏi → animate từng bước: **User → Customer (:10100) → Registry discover (:10000) → Law (:10101) → routing → Tax (:10102) ∥ Compliance (:10103) ∥ analyze_law → aggregate → trả lời**.
- Live log minh hoạ `trace_id` được truyền qua mọi hop A2A.
- Có công tắc **Baseline (nối tiếp)** vs **Optimized (song song)** để thấy rõ khác biệt về luồng và latency.
- Logic keyword routing trong demo **giống hệt** `law_agent/graph.py` (`_TAX_KEYWORDS`, `_COMPLIANCE_KEYWORDS`).
- **Nút “⏱️ Đo thật”**: gọi LIVE tới hệ thống thật (`POST :10100/ask`), đếm giờ và hiển thị
  **latency đo được + câu trả lời thật**. (Đã thêm CORS + endpoint REST `/ask` vào
  `customer_agent/__main__.py` để trình duyệt gọi được.)
  Ví dụ đo live đã xác nhận: **~283s**, response 8376 ký tự.

---

## Mục 2 — Latency & phương án giảm

### 2.1. Latency của hệ thống là bao nhiêu giây?

Đo bằng [`measure_latency.py`](measure_latency.py) (đo wall-clock từ lúc gửi request tới Customer Agent đến khi nhận đủ câu trả lời):

```
Baseline: 420.0s  (response 5262 ký tự)
```

→ **Latency ban đầu ≈ 420 giây / câu hỏi.**

> Latency lớn vì dùng model **free** (bị queue/rate-limit), và vì chuỗi LLM **nối tiếp** trên critical path.

### 2.2. Bottleneck (đọc từ log thật)

Critical path của 1 request gồm ~6 LLM call **nối tiếp**:

```
Customer (quyết định delegate)
  → Law: analyze_law → check_routing → [Tax ∥ Compliance] → aggregate
    → Customer (phản hồi cuối)
```

Log baseline lộ rõ điểm nghẽn:

```
15:12:51  analyze_law xong
15:14:33  check_routing xong     ← cách nhau ~102 GIÂY chỉ cho 1 LLM call định tuyến!
15:14:35  Routing decision: needs_tax=True needs_compliance=True
```

Hai vấn đề:
1. **`check_routing` gọi LLM** chỉ để trả về 2 cờ true/false → tốn nguyên 1 LLM call nối tiếp (~102s).
2. **`analyze_law` chạy nối tiếp TRƯỚC** Tax/Compliance, dù output của nó chỉ được dùng ở bước `aggregate` (Tax/Compliance không cần `law_analysis`).

### 2.3. Phương án giảm latency (đã apply trong `law_agent/graph.py`)

| # | Thay đổi | Lợi ích |
|---|---|---|
| 1 | **Keyword routing**: thay LLM trong `check_routing` bằng so khớp từ khoá | Bỏ hẳn 1 LLM call nối tiếp (~102s), routing còn ~2ms |
| 2 | **Song song hoá `analyze_law`**: dispatch `analyze_law` cùng lúc với Tax/Compliance qua `Send`, hội tụ tại `aggregate` | `analyze_law` không còn chặn critical path |

Topology bên Law Agent: từ **4 tầng nối tiếp** `analyze → route → specialists → aggregate`
xuống **2 tầng** `route(tức thời) → {analyze ∥ tax ∥ compliance} → aggregate`.

Log sau tối ưu xác nhận:
```
15:22:24.102  LawAgent executing
15:22:24.104  Routing decision (keyword): needs_tax=True needs_compliance=True   ← ~2ms
15:22:25-26   tax + compliance + analyze_law chạy SONG SONG
```

### 2.4. Kết quả demo (trước/sau)

| Chế độ | Latency | Ghi chú |
|---|---|---|
| **Baseline** (LLM routing + analyze nối tiếp) | **420.0s** | response 5262 ký tự |
| **Optimized** (keyword routing + analyze song song) | **299.3s** | response 6521 ký tự |

> **Giảm ≈ 121 giây (≈ 29%).** Đáng chú ý: routing từ ~102s → ~2ms; phần còn lại là độ trễ của các LLM call specialist/aggregate (do model free).

### 2.5. Hướng giảm thêm (chưa apply)

- **Đổi model nhanh/trả phí** (vd `claude-sonnet-4-5`, `gemini-flash`): loại bỏ độ trễ queue của free tier — đây là yếu tố chi phối lớn nhất.
- **Bỏ bớt LLM call ở Customer Agent**: câu trả lời cuối của Law Agent đã hoàn chỉnh, Customer có thể trả thẳng thay vì gọi thêm 1 LLM để "diễn đạt lại".
- **Streaming** kết quả về user để giảm *perceived latency*.
- **Cache** câu trả lời cho các câu hỏi lặp lại.

---

## Cách chạy lại

```bash
# 1. Khởi động hệ thống
./start_all.sh

# 2. Đo latency (1 lần, hoặc truyền số lần)
uv run python measure_latency.py 1

# 3. Mở demo trực quan
open demo.html
```
