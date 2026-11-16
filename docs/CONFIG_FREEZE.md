# Cấu hình chốt trước khi chạy test

Ghi ngày 2026-11-16, trước khi chạy bất kỳ câu test nào. Test chỉ chạy một lần, sau file này.

## Cấu hình được chọn: `agent-dev-hybrid-b2500`

Hybrid retrieval (dense e5 + BM25, RRF), gpt-4o-mini, top_k=3, context budget 2500 ký tự, tối đa 3 lượt tìm, **không dùng filter**.

## So sánh trên dev (20 câu có đáp án + 2 câu thiếu bằng chứng, chấm tay)

| Run | Grounded TB /2 | Từ chối nhầm | Từ chối đúng | Citation đúng | Chi phí |
|---|---:|---:|---:|---:|---:|
| b800 (baseline) | 1.05 | 9/20 | 2/2 | 11/11 | 0,0051 USD |
| **b2500 (chọn)** | **1.95** | 0/20 | 2/2 | 24/25 | 0,0053 USD |
| b2500 + filter | 1.95 | 0/20 | 2/2 | 23/24 | 0,0053 USD |

## Tiêu chí chọn (công bố trước khi chạy test)

1. Chất lượng câu trả lời có căn cứ (grounded answer trung bình) cao nhất trên dev.
2. Ít từ chối nhầm câu có đáp án nhất, và vẫn từ chối đúng câu thiếu bằng chứng.
3. Khi hai cấu hình bằng nhau về chất lượng, chọn cấu hình đơn giản hơn, không phụ thuộc đầu vào bổ sung (filter do người dùng cung cấp).
4. Chi phí và thời gian chênh nhau không đáng kể (dưới 10%) nên không phải yếu tố quyết định.

`b2500` và `b2500 + filter` bằng nhau về chất lượng nên áp dụng tiêu chí 3: chọn `b2500`.

## Lệnh chạy test

```
python src/agent.py --split test --method hybrid --budget 800  --allow-test --output results/agent-test-hybrid-b800
python src/agent.py --split test --method hybrid --budget 2500 --allow-test --output results/agent-test-hybrid-b2500
```

Không chỉnh prompt, retriever hay cấu hình sau khi xem kết quả test.
