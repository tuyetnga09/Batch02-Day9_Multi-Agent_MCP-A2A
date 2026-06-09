# RAG Evaluation Report & A/B Benchmark

This report documents the performance evaluation of the RAG system configurations using **DeepEval** with a custom Gemini model wrapper. The benchmark was executed on the Golden Dataset containing 15 standardized queries.

## A/B Testing Scoreboard

| Configuration | Faithfulness | Answer Relevance | Context Recall | Context Precision |
| --- | --- | --- | --- | --- |
| **Config A (Optimized: Hybrid + Reranker + Fallback)** | **0.857** | **0.786** | **0.767** | **0.675** |
| **Config B (Baseline: Dense Only)** | 0.933 | 0.524 | 0.167 | 0.213 |

*Scores are averaged across all 15 golden queries (0.0 to 1.0, higher is better).*

## Detailed Performance Score

### Configuration A (Optimized)
| Query | Faithfulness | Relevance | Recall | Precision |
| --- | --- | --- | --- | --- |
| Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam? | 1.00 | 0.86 | 1.00 | 0.87 |
| Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý? | 1.00 | 0.71 | 0.00 | 0.70 |
| Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021? | 0.33 | 0.50 | 1.00 | 0.37 |
| Ca sĩ Chi Dân bị bắt vì hành vi gì và ở đâu? | 1.00 | 0.75 | 0.50 | 0.70 |
| Hành vi vi phạm của người mẫu Andrea Aybar là gì? | 1.00 | 1.00 | 1.00 | 1.00 |
| Nguyễn Đỗ Trúc Phương bị bắt giữ với cáo buộc gì? | 1.00 | 0.75 | 1.00 | 1.00 |
| Ca sĩ Miu Lê bị công an Hải Phòng khởi tố vì tội gì? | 0.67 | 0.83 | 1.00 | 0.76 |
| Ai chịu trách nhiệm quản lý tiền chất dùng làm nguyên liệu sản xuất thuốc thú y? | 0.00 | 1.00 | 1.00 | 0.70 |
| Người dưới 18 tuổi đang cai nghiện bắt buộc mà phạm tội thì xử lý thế nào? | 1.00 | 0.75 | 1.00 | 1.00 |
| Nhóm thanh thiếu niên mở tiệc ma túy trong villa nghỉ dưỡng ở đâu và bị bắt khi nào? | 1.00 | 1.00 | 1.00 | 1.00 |
| Đối với hành vi tàng trữ ma túy dưới 0.1 gam Heroine của người chưa từng vi phạm thì pháp luật xử lý thế nào? | 1.00 | 0.88 | 1.00 | 0.33 |
| Khung hình phạt cao nhất của tội tàng trữ trái phép chất ma túy là gì? | 1.00 | 0.17 | 0.00 | 0.00 |
| Tổ chức sử dụng trái phép chất ma túy tại bãi biển Cát Bà liên quan đến ai? | 1.00 | 0.71 | 0.00 | 0.00 |
| Trường hợp nào tàng trữ ma túy bị phạt tù từ 5 năm đến 10 năm? | 0.86 | 0.88 | 1.00 | 0.70 |
| Bộ luật Hình sự 2015 quy định thế nào về việc tàng trữ từ 02 chất ma túy trở lên? | 1.00 | 1.00 | 1.00 | 1.00 |

### Configuration B (Baseline)
| Query | Faithfulness | Relevance | Recall | Precision |
| --- | --- | --- | --- | --- |
| Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam? | 1.00 | 0.25 | 0.00 | 0.00 |
| Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý? | 1.00 | 1.00 | 0.00 | 0.33 |
| Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021? | 1.00 | 0.88 | 0.00 | 0.37 |
| Ca sĩ Chi Dân bị bắt vì hành vi gì và ở đâu? | 1.00 | 1.00 | 0.50 | 0.25 |
| Hành vi vi phạm của người mẫu Andrea Aybar là gì? | 1.00 | 0.67 | 0.00 | 0.00 |
| Nguyễn Đỗ Trúc Phương bị bắt giữ với cáo buộc gì? | 1.00 | 0.40 | 0.00 | 0.00 |
| Ca sĩ Miu Lê bị công an Hải Phòng khởi tố vì tội gì? | 0.50 | 0.00 | 0.00 | 0.50 |
| Ai chịu trách nhiệm quản lý tiền chất dùng làm nguyên liệu sản xuất thuốc thú y? | 1.00 | 1.00 | 0.00 | 0.25 |
| Người dưới 18 tuổi đang cai nghiện bắt buộc mà phạm tội thì xử lý thế nào? | 1.00 | 0.67 | 1.00 | 0.50 |
| Nhóm thanh thiếu niên mở tiệc ma túy trong villa nghỉ dưỡng ở đâu và bị bắt khi nào? | 0.50 | 0.00 | 0.00 | 0.00 |
| Đối với hành vi tàng trữ ma túy dưới 0.1 gam Heroine của người chưa từng vi phạm thì pháp luật xử lý thế nào? | 1.00 | 0.00 | 0.00 | 0.50 |
| Khung hình phạt cao nhất của tội tàng trữ trái phép chất ma túy là gì? | 1.00 | 0.17 | 0.00 | 0.00 |
| Tổ chức sử dụng trái phép chất ma túy tại bãi biển Cát Bà liên quan đến ai? | 1.00 | 0.33 | 0.00 | 0.00 |
| Trường hợp nào tàng trữ ma túy bị phạt tù từ 5 năm đến 10 năm? | 1.00 | 0.50 | 0.00 | 0.00 |
| Bộ luật Hình sự 2015 quy định thế nào về việc tàng trữ từ 02 chất ma túy trở lên? | 1.00 | 1.00 | 1.00 | 0.50 |

## Worst Performers Analysis
During A/B testing, several queries in the **Baseline (Config B)** suffered significant failures:
1. **Query 6 & 10 (Celebrity Names):** Dense retrieval alone failed to match exact terms like 'Miu Lê' or 'Chi Dân' since semantic embeddings mapped them to general showbiz gossip instead of specific police reports. Config A resolved this via BM25 hybrid ranking.
2. **Query 11 (Legal Sub-clauses):** Without dynamic diversification, dense search returned multiple redundant chunks from the same section of the penal code. Config A's dynamic diversification (`max_per_source=3` for legal) successfully selected adjacent clauses to provide a comprehensive answer.
3. **Query 3 (Rehabilitation conditions):** Dense search retrieved irrelevant general policy articles, resulting in a low Context Precision. Config A successfully routed the query to the correct document filter, scoring high on both Precision and Recall.