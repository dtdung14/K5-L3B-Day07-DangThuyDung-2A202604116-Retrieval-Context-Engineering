# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.

## Cấu trúc repo

| Thư mục / file | Nội dung |
| --- | --- |
| [starter/](starter/README.md) | Code mẫu: baseline, vòng lặp agent, metrics, chuyển đổi dữ liệu Day 6 |
| [data/](data/README.md) | Ví dụ chunks, câu hỏi benchmark và agent cases |
| [src/](src/README.md) | Nơi đặt code của bạn |
| [docs/](docs/README.md) | Tài liệu bổ sung |
| [tests/](tests/) | Unit test (mock) cho starter |
| [report/individual/](report/individual/README.md) | Báo cáo cá nhân |
| [assets/](assets/) | Sơ đồ minh họa |

## Tài liệu

| File | Mục đích |
| --- | --- |
| [GUIDE.md](GUIDE.md) | Hướng dẫn làm bài và gợi ý phương pháp retrieval |
| [PSEUDOCODE.md](PSEUDOCODE.md) | Pseudocode tham khảo |
| [BENCHMARK.md](BENCHMARK.md) | Định dạng và yêu cầu benchmark |
| [RUBRIC.md](RUBRIC.md) | Tiêu chí chấm điểm |
| [RULES.md](RULES.md) | Quy tắc làm bài |
| [ADVANCED.md](ADVANCED.md) | Thử thách nâng cao multimodal (+15 điểm) |
| [CHECKPOINTS.md](CHECKPOINTS.md) | Mốc thời gian gợi ý |
| [SUBMISSION.md](SUBMISSION.md) | Cách nộp bài |
| [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) | Mẫu báo cáo |
| [AGENT_SETUP.md](AGENT_SETUP.md) | Dùng AI hỗ trợ lập trình (tùy chọn) |

## Chạy test

```bash
python -m unittest discover -s tests -v
```

---

# Bài làm: Agentic RAG trên *Psychology 2e* (Chương 1–2)

Agent dùng `gpt-4o-mini` với tool calling để tìm bằng chứng trong kho Day 6, trả lời có citation `[chunk-id]` hoặc từ chối khi thiếu bằng chứng. Cấu hình đã chốt: **hybrid retrieval, top-3, context budget 2500 ký tự, không filter** ([lý do chọn](docs/CONFIG_FREEZE.md)).

## Dữ liệu và mô hình

| Mục | Giá trị |
|---|---|
| Tài liệu | *Psychology 2e*, OpenStax (CC BY), Chương 1–2, 51 trang nội dung |
| Chunk | 279 đoạn (tách theo đoạn văn, tối đa 800 ký tự), bảng `chunks_b` của Day 6 |
| Embedding | `intfloat/multilingual-e5-base`, 768 chiều, tiền tố `query:`/`passage:`, cosine |
| Vector store | PostgreSQL 16 + pgvector 0.8 (Docker, tìm exact) |
| Model agent | `gpt-4o-mini`, `temperature=0` |
| Benchmark | [data/questions.json](data/questions.json): dev 20 câu có đáp án + 2 thiếu bằng chứng; test 5 + 2 |

Năm câu test cũ của Day 6 đã dùng để chọn cấu hình nên được chuyển sang dev; test hiện tại (q21–q25, q28–q29) là câu mới viết sau đó.

## Ba phương pháp retrieval ([src/retrievers.py](src/retrievers.py))

- **Dense:** query embedding e5 + pgvector (kho Day 6).
- **BM25:** lexical, tự cài, chạy trên `data/day6/chunks.json`.
- **Hybrid:** Reciprocal Rank Fusion (k=60) của hai bảng xếp hạng trên.

## Kết quả

**Retrieval trên dev** (20 câu có đáp án, query gốc, top-3, [chi tiết](results/retrieval-dev/summary.md)):

| Phương pháp | Hit@1 | Evidence Hit@3 | MRR |
|---|---:|---:|---:|
| Dense | 15/20 | 17/20 | 0.800 |
| BM25 | 17/20 | 19/20 | 0.900 |
| Hybrid | 15/20 | 20/20 | 0.867 |
| Dense + filter | 16/20 | 18/20 | 0.850 |
| BM25 + filter | 18/20 | 19/20 | 0.925 |
| Hybrid + filter | 16/20 | 20/20 | 0.900 |

**Agent (hybrid), chấm tay** (thang 0–2; chi phí cả lần chạy):

| Split | Cấu hình | Điểm TB /2 | Từ chối nhầm | Từ chối đúng | Citation đúng | Chi phí |
|---|---|---:|---:|---:|---:|---:|
| dev | b800 (baseline) | 1.05 | 9/20 | 2/2 | 11/11 | $0.0051 |
| dev | b2500 (chọn) | 1.95 | 0/20 | 2/2 | 24/25 | $0.0053 |
| dev | b2500 + filter | 1.95 | 0/20 | 2/2 | 23/24 | $0.0053 |
| test | b800 (baseline) | 1.2 | 2/5 | 2/2 | 3/3 | $0.0016 |
| test | **b2500 (chọn)** | **2.0** | 0/5 | 2/2 | 5/5 | $0.0020 |

Nhận xét chính: budget 800 ký tự chỉ chứa khoảng một chunk nên evidence bị cắt khỏi context và agent từ chối nhầm; tăng lên 2500 loại bỏ lỗi này. Filter `section` không cải thiện thêm. Lỗi còn lại ở dev: q01 (trả lời "Germany" thay vì "Leipzig", citation không chứa năm 1879).

**Giới hạn:** test chỉ có 5 câu có đáp án nên kết quả chưa đủ để suy ra hiệu năng tổng quát; câu hỏi được viết từ chính các chunk nên dễ hơn câu hỏi tự nhiên; nhãn do người chấm (có hỗ trợ AI) nên cần đối chiếu lại; model dao động nhẹ giữa các lần chạy dù `temperature=0`. Chưa làm thử thách nâng cao 100 trang.

## Cách chạy lại

Cần Python 3.10+ (đã chạy trên 3.10), Docker Desktop, và dữ liệu Day 6 đã nạp vào pgvector.

```bash
pip install psycopg[binary] pgvector sentence-transformers openai python-dotenv numpy
cp .env.example .env            # điền OPENAI_API_KEY; DATABASE_URL trỏ tới pgvector Day 6
# (Windows PowerShell: Copy-Item .env.example .env)

# 1. (tùy chọn) xuất lại dữ liệu từ Day 6; kết quả đã có sẵn trong data/day6/
python src/export_day6.py --day6 "<đường dẫn thư mục starter Day 6>" --output data/day6

# 2. So sánh retrieval trên dev
python src/evaluate_retrieval.py --split dev --output results/retrieval-dev

# 3. Chạy agent trên dev (baseline và cấu hình cải thiện)
python src/agent.py --split dev --method hybrid --output results/agent-dev-hybrid-b800
python src/agent.py --split dev --method hybrid --budget 2500 --output results/agent-dev-hybrid-b2500
python src/agent.py --split dev --method hybrid --budget 2500 --use-filter --output results/agent-dev-hybrid-b2500-filter

# 4. Chấm tay: đọc review.md, điền reviews.json, rồi tổng hợp
python src/review.py export  --run results/agent-dev-hybrid-b2500
python src/review.py metrics --run results/agent-dev-hybrid-b2500

# 5. Test, chỉ chạy một lần sau khi chốt cấu hình (cần --allow-test)
python src/agent.py --split test --allow-test --method hybrid --budget 2500 --output results/agent-test-hybrid-b2500
```

Thử một câu: `python src/agent.py --question "..." --method hybrid --budget 2500`.

Test code (mock, không đo chất lượng): `python -m unittest discover -s tests -v`. Trên Windows, `test_baseline` có thể lỗi encoding (cp1258) từ starter; đặt `PYTHONUTF8=1` nếu gặp.

## Cấu trúc bài làm

| Đường dẫn | Nội dung |
|---|---|
| [src/retrievers.py](src/retrievers.py) | Dense, BM25, Hybrid (RRF) |
| [src/agent.py](src/agent.py) | Agent OpenAI tool calling, trace, chi phí |
| [src/evaluate_retrieval.py](src/evaluate_retrieval.py) | Hit@1, Hit@3, MRR, Context Evidence |
| [src/review.py](src/review.py) | Xuất bảng chấm tay và tổng hợp nhãn |
| [src/export_day6.py](src/export_day6.py) | Xuất chunk và benchmark từ Day 6 |
| [data/](data/) | `questions.json` và `day6/` (chunks, câu hỏi gốc) |
| [results/](results/) | Trace, `review.md`, nhãn `reviews.json`, `metrics.json` của từng lần chạy |
| [docs/CONFIG_FREEZE.md](docs/CONFIG_FREEZE.md) | Cấu hình và tiêu chí chọn, ghi trước khi chạy test |
