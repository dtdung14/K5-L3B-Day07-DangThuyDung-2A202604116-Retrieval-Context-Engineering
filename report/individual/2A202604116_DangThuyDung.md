# Báo cáo Day 7: Agentic RAG trên *Psychology 2e*

Đặng Thùy Dung · 2A202604116 · Lớp L3B

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** *Psychology 2e* (OpenStax, giấy phép CC BY), https://openstax.org/details/books/psychology-2e. Dùng Chương 1–2 (51 trang nội dung, 158.144 ký tự), dữ liệu Day 6. Kho có **279 chunk** (bảng `chunks_b`, tách theo đoạn văn, tối đa 800 ký tự, trung bình 565 ký tự). Dữ liệu xuất ở `data/day6/chunks.json`.
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16 + pgvector 0.8 (Docker), tìm exact bằng cosine. Embedding `intfloat/multilingual-e5-base`, **768 chiều**, tiền tố `query:`/`passage:`, vector đã chuẩn hóa. Query embedding dùng cùng model với index nên tương thích.
- **Agent model, cách chạy và phân chia dev/test:** `gpt-4o-mini`, `temperature=0`, tool calling (`search`, `answer`, `abstain`). Benchmark ở `data/questions.json`:
  - **Dev:** 20 câu có đáp án + 2 câu thiếu bằng chứng. Gồm 15 câu dev Day 6 và 5 câu test cũ của Day 6 (đã dùng để chọn cấu hình ở Day 6 nên chuyển sang dev, không còn độc lập).
  - **Test:** 5 câu có đáp án (q21–q25) + 2 câu thiếu bằng chứng (q28–q29), viết mới (sau khi chuyển test cũ sang dev) từ các trang chưa có câu hỏi nào. Test chạy **một lần** sau khi chốt cấu hình ([docs/CONFIG_FREEZE.md](../../docs/CONFIG_FREEZE.md)).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có đáp án: gold gồm `source_id`, trang, đoạn evidence nguyên văn nằm trọn trong một chunk. Script xác nhận cả 25 câu có evidence khớp nguyên văn trong chunk đúng trang.
  - Câu thiếu bằng chứng (q26 Bandura/Bobo, q27 Rorschach, q28 SSRI, q29 Zimbardo): chủ đề gần tâm lý học nhưng đã tìm các từ khóa (Bandura, Bobo, Rorschach, inkblot, SSRI, serotonin, Prozac, Zimbardo, Stanford) trong 279 chunk và trong PDF gốc 755 trang bằng `pypdf`: **0 kết quả ở mọi trang Chương 1–2 (trang 19–74)**. Các chủ đề này có xuất hiện ở chương sau của cuốn sách (ví dụ Bandura/Bobo trang 216–222, Rorschach trang 400+, SSRI trang 95/625, Zimbardo trang 418+), nên câu thiếu bằng chứng so với **phần được index (Chương 1–2)**, không phải so với cả cuốn sách.
  - Chấm theo [BENCHMARK.md](../../BENCHMARK.md): grounded answer 0/1/2, citation hỗ trợ, abstention, context 0/1/2. Hit/Context Evidence tự động dựa trên việc evidence nguyên văn nằm trong chunk (đúng trang) hoặc trong context đóng gói.

## Retrieval và thử nghiệm

Ba phương pháp khác nhau về cách tìm/xếp hạng ([src/retrievers.py](../../src/retrievers.py)):

1. **Dense:** query embedding e5 + pgvector (kho Day 6).
2. **BM25:** lexical, Okapi (k1=1.5, b=0.75), tự cài trên `chunks.json`.
3. **Hybrid:** Reciprocal Rank Fusion (k=60) của dense và BM25, mỗi bên lấy 20 ứng viên, vì xếp hạng theo thứ hạng nên không cần chuẩn hóa score.

Retrieval dùng query gốc, top-3, cùng corpus, index và nhãn. Context Evidence Rate (máy) = evidence nguyên văn nằm trong context top-3 đóng gói theo budget (gồm cả header nguồn). Thời gian là ms/câu, chi phí retrieval 0 USD (chạy local).

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense, top-3, không filter, không rerank, budget 800 | Dense trên vector store Day 6 là điểm xuất phát | dev | 15/20 (75%) | 17/20 (85%) | 12/20 (60%) | 55 ms / $0 |
| Phương pháp 2 | BM25, top-3, không filter, budget 800 | Câu hỏi nhiều tên riêng nên lexical khớp từ khóa tốt | dev | 17/20 (85%) | 19/20 (95%) | 12/20 (60%) | 0,7 ms / $0 |
| Phương pháp 3 | Hybrid (RRF), top-3, không filter, budget 800 | Hai bảng xếp hạng bù lỗi cho nhau | dev | 15/20 (75%) | 20/20 (100%) | 13/20 (65%) | 50 ms / $0 |
| Cải thiện 1 | Hybrid, top-3, **budget 2500** | Budget 800 chỉ chứa ~1 chunk nên evidence ở hạng 2–3 bị cắt; tăng budget giữ được evidence | dev | 15/20 (75%) | 20/20 (100%) | 20/20 (100%) | 55 ms / $0 |
| Cải thiện 2 | Hybrid, top-3, budget 2500, **+ filter `section`** (do người dùng cung cấp cho 5 câu) | Lọc theo chương loại bớt nhiễu khác chương | dev | 16/20 (80%) | 20/20 (100%) | 20/20 (100%) | 87 ms / $0 |
| Cấu hình đã chọn | Hybrid, top-3, budget 2500, không filter | Chốt trên dev ([CONFIG_FREEZE](../../docs/CONFIG_FREEZE.md)) | test | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 67 ms / $0 |
| Baseline | Hybrid, top-3, budget 800 (baseline của agent) | Đối chứng trên test | test | 5/5 (100%) | 5/5 (100%) | 3/5 (60%) | 60 ms / $0 |

Các biến thể có filter của dense/BM25 (dev, budget 800) nằm ở [results/retrieval-dev/summary.md](../../results/retrieval-dev/summary.md); budget 2500 ở `results/retrieval-dev-b2500/`; test ở `results/retrieval-test-b800/` và `results/retrieval-test-b2500/`.

Hit@1/Hit@3 không phụ thuộc budget nên giống nhau giữa hai dòng hybrid; chỉ Context Evidence thay đổi. Phương pháp hybrid cũng là baseline của các thí nghiệm agent vì Hit@3 cao nhất (20/20) và không bỏ sót câu nào ở top-3; cải thiện 1 và 2 được so với baseline này. Cải thiện 1 là thay đổi cấu hình (budget), không phải phương pháp mới; cải thiện 2 (filter) cũng là cấu hình. Test chỉ có 5 câu nên Hit 5/5 không phân biệt được các phương pháp.

**Hai lỗi dev và cách xử lý:**

1. **q01 (Wundt, Leipzig).** *Hiện tượng:* agent trả lời "Germany, 1879" thay vì "University of Leipzig", trích chunk trang 22 không chứa 1879. *Nguyên nhân:* retrieval với câu hỏi gốc có tìm ra evidence trang 21 trong top-3 (hybrid Hit@3 20/20), nhưng agent tự viết lại query thành "Wilhelm Wundt psychology laboratory location and date", nên evidence không vào context cuối; model trả lời từ chú thích hình ở trang 22. *Thử:* tăng budget (b2500) không sửa được lỗi này; filter cũng không. *Kết luận:* lỗi do query viết lại, cần đo riêng; chưa sửa, ghi nhận là giới hạn.
2. **Từ chối nhầm 9/20 câu ở baseline (q01, q04, q05, q09, q10, q11, q14, q15, q16).** *Nguyên nhân:* đoạn đóng gói theo thứ tự xếp hạng, nhưng chunk trung bình 565 ký tự + header, nên budget 800 ký tự chỉ vừa khoảng một chunk; chunk chứa evidence bị bỏ khi không vừa, model từ chối đúng vì context thiếu bằng chứng (cả 9 câu đều có context mức 0). *Thử:* budget 2500 (cải thiện 1). *Kết quả:* từ chối nhầm 9/20 xuống 0/20, điểm trung bình 1,05 lên 1,95.

**Lý do chọn cấu hình:** theo tiêu chí công bố trước khi chạy test (grounded answer cao nhất, ít từ chối nhầm, hòa thì chọn đơn giản hơn không phụ thuộc đầu vào bổ sung): b2500 và b2500+filter bằng nhau về chất lượng (1,95), nên chọn b2500 không filter. Chi phí chênh dưới 10%.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Tool: `search(query)`, `answer(answer, citations)`, `abstain(reason)`; bước đầu bắt buộc `search`. Khung điều phối là `run_agent` của starter, `OpenAIController` trong [src/agent.py](../../src/agent.py).
  - Tối đa 3 lượt tìm. Dừng khi model gọi `answer`/`abstain`, hoặc hết lượt (tự từ chối). Câu trả lời không có `[chunk-id]` hợp lệ trong context cũng bị chuyển thành từ chối.
  - Context: gộp ứng viên các lượt tìm, loại trùng, đóng gói theo thứ tự xếp hạng đến hết budget (ký tự, gồm header nguồn), chunk không vừa bị bỏ. Nội dung truy xuất được bọc trong `<retrieved_context>` và prompt nói rõ là dữ liệu, không phải chỉ dẫn.
  - Lỗi API được ghi `status: "error"` trong trace, không thay bằng kết quả giả. Cả 5 lần chạy đều **0 lỗi hệ thống**.
- **Phân bố số lượt tìm (số câu):** b800 dev: 1 lượt: 11, 3 lượt: 11; b2500 dev: 1 lượt: 20, 3 lượt: 2; test b800: 1: 4, 3: 3; test b2500: 1: 5, 3: 2. Các câu 3 lượt là câu từ chối.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| dev, b800 (baseline) | 22/22 | 66 LLM, 44 search | 9/1/10, TB 1,05 | 11/11 | 11/11 | 2/2 | 9/20 |
| dev, b2500 | 22/22 | 48 LLM, 26 search | 0/1/19, TB 1,95 | 24/25 | 20/20 | 2/2 | 0/20 |
| dev, b2500 + filter | 22/22 | 48 LLM, 26 search | 0/1/19, TB 1,95 | 23/24 | 20/20 | 2/2 | 0/20 |
| test, b800 (baseline) | 7/7 | 20 LLM, 13 search | 2/0/3, TB 1,2 | 3/3 | 3/3 | 2/2 | 2/5 |
| test, b2500 (chọn) | 7/7 | 18 LLM, 11 search | 0/0/5, TB 2,0 | 5/5 | 5/5 | 2/2 | 0/5 |

Context do người chấm (0/1/2): dev b800 = 9/0/11, dev b2500 = 1/0/19, test b800 = 2/0/3, test b2500 = 0/0/5.

Ví dụ trace và nhãn (đủ trace ở thư mục `results/`):

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q06 (dev, b2500) | Có đáp án (filter Chương 2 ở run filter) | 1 search, hybrid top-3, context có chú thích hình p50 | "Inez Beverly Prosser ... [86742f43]" và Clark [b94a1d91] | Answer 2, cả hai citation đúng; evidence gốc không nguyên văn trong context nhưng chú thích p50 đủ hỗ trợ |
| q01 (dev, b2500) | Có đáp án | 1 search, query tự viết lại; evidence p21 không vào context | "Germany in 1879 [199c85b7]" (p22) | Answer 1 (thiếu Leipzig), citation sai: chunk không chứa 1879; context 0 |
| q01 (dev, b800) | Có đáp án | 3 search, context chỉ có chunk 99 ký tự | "Không đủ bằng chứng" | Answer 0 (từ chối nhầm), context 0 do budget |
| q20 (dev, b800) | Có đáp án | 1 search, chunk p68 "five percent or less" | "ngưỡng 5% ... false positive [c14d83c7]" | Answer 1: đúng 5% nhưng thiếu điều kiện "nếu hai nhóm không khác nhau"; citation đúng |
| q26 (dev) | Thiếu bằng chứng (Bandura) | 3 search, không đoạn liên quan | Từ chối | Abstention đúng, không thêm khẳng định |
| q21 (test, b800) | Có đáp án | 3 search, evidence p24 bị cắt khỏi context | Từ chối | Answer 0, context 0 |
| q21 (test, b2500) | Có đáp án | 1 search, evidence p24 trong context | "John B. Watson ... behaviorism [2d541957]" | Answer 2, citation đúng, context 2 |
| q25 (test, b2500) | Có đáp án | 1 search, chunk p73 | "1972, báo chí ... bắt đầu 1932 [6ab2d154]" | Answer 2; chi tiết 1932 có trong context (chunk p72) nhưng không được trích, citation đủ cho ý chính |
| q28 (test) | Thiếu bằng chứng (SSRI) | 3 search, không đoạn liên quan | Từ chối | Abstention đúng |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:** `results/agent-{dev,test}-hybrid-{b800,b2500,b2500-filter}/` gồm `traces.json`, `review.md`, `reviews.json`, `metrics.json`, `summary.json`. Lệnh chạy ở [README.md](../../README.md). Tổng chi phí: dev b800 $0,00505 (TB 3,8 s/câu), dev b2500 $0,00534 (3,4 s), dev b2500+filter $0,00532 (3,4 s), test b800 $0,00156 (5,3 s), test b2500 $0,00200 (3,4 s). Giá token giả định 0,15 và 0,60 USD/triệu token (input/output) cho gpt-4o-mini; cần đối chiếu bảng giá hiện hành.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - *Trước/sau:* tăng budget 800 lên 2500 là thay đổi duy nhất mang lại cải thiện (dev 1,05 lên 1,95; test 1,2 lên 2,0), chi phí tăng khoảng 5–28% (dev 5%, test 28%, do context dài hơn) nhưng thời gian trung bình giảm vì baseline phải tìm lặp nhiều lần. Filter không thay đổi chất lượng; run khác nhau ở một số câu không có filter (q02, q10) cho thấy model dao động giữa các lần chạy dù `temperature=0`, nên chênh lệch nhỏ chưa phải bằng chứng.
  - *Đánh đổi:* context dài hơn cho nhiều bằng chứng hơn nhưng tăng token, và có thể cho model thêm chi tiết không cần thiết (như 1932 ở q25).
  - *Giới hạn:* (1) test chỉ có 5 câu có đáp án và 2 câu thiếu bằng chứng, 5/5 không cho thấy hiệu năng tổng quát; (2) câu hỏi viết từ chính các chunk nên dễ hơn câu hỏi tự nhiên và đều là câu đơn đoạn, chưa có multi-hop; (3) nhãn được chấm với hỗ trợ của AI rồi đối chiếu lại: nhãn dev và test đều đã được tôi kiểm tra lại; (4) câu thiếu bằng chứng được kiểm bằng từ khóa chứ chưa đọc từng đoạn gần chủ đề (ví dụ đoạn Skinner/Watson nói về học tập) để loại trừ đáp án gián tiếp, và chỉ đúng với phạm vi index Chương 1–2; (5) chưa thử rerank, query transformation hay multi-hop có kiểm soát, và lỗi q01 do query viết lại chưa được sửa; (6) dữ liệu mới gồm Chương 1–2 (51 trang), nhỏ hơn tài liệu thực tế.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
