# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đức Khang  
**MSSV:** 2A202600588  
**Nhóm:** F2  
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
High cosine similarity nghĩa là hai vector embedding có hướng gần nhau trong không gian nhúng. Với text embeddings, điều này thường cho thấy hai câu cùng chủ đề hoặc gần nhau về nghĩa, dù cách diễn đạt khác nhau.

**Ví dụ HIGH similarity:**

- Sentence A: "List comprehension trong Python tạo list mới từ một iterable."
- Sentence B: "Có thể dùng cú pháp comprehension thay cho vòng lặp for để tạo list."
- Lý do: Hai câu đều nói về cách tạo list bằng list comprehension trong Python.

**Ví dụ LOW similarity:**

- Sentence A: "Class trong Python định nghĩa khuôn mẫu để tạo object."
- Sentence B: "Lệnh pip install tải package từ PyPI về máy."
- Lý do: Một câu nói về OOP, câu còn lại nói về package management. Hai chủ đề này không gần nhau trong bộ tài liệu.

**Tại sao cosine similarity phù hợp hơn Euclidean distance cho text embeddings?**  
Cosine similarity đo góc giữa hai vector, nên tập trung vào hướng biểu diễn nghĩa. Euclidean distance bị ảnh hưởng nhiều hơn bởi độ dài vector; với text, hai câu cùng nghĩa nhưng độ dài khác nhau vẫn nên được xem là gần nhau.

### Chunking Math (Ex 1.2)

Document dài 10,000 ký tự, `chunk_size=500`, `overlap=50`.

```text
step = chunk_size - overlap = 500 - 50 = 450
num_chunks = ceil((doc_length - overlap) / step)
           = ceil((10000 - 50) / 450)
           = ceil(9950 / 450)
           = 23 chunks
```

Nếu `overlap=100`:

```text
step = 500 - 100 = 400
num_chunks = ceil((10000 - 100) / 400)
           = ceil(9900 / 400)
           = 25 chunks
```

Overlap tăng từ 50 lên 100 làm số chunk tăng từ 23 lên 25. Đổi lại, mỗi chunk giữ thêm context từ chunk trước, giảm rủi ro cắt mất ý ở ranh giới.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Python Official Documentation for Building Data/RAG Applications.

Nhóm chọn domain này vì lab dùng Python để xây dựng chunking, vector store, metadata filtering và RAG agent. Bộ tài liệu lấy từ Python Official Documentation nên nguồn rõ ràng, nội dung ổn định và có cấu trúc heading/section phù hợp để thử nhiều chunking strategies.

Nguồn dữ liệu được crawl bằng Firecrawl local để kiểm tra scrape success, sau đó dùng `requests + BeautifulSoup + markdownify` để lấy phần main content sạch hơn. Metadata inventory được lưu tại:

```text
data/python_official_docs_metadata.json
```

Firecrawl result:

```text
8 / 8 documents firecrawl_success = true
```

### Data Inventory


| #   | Tên tài liệu                | Topic            | Nguồn                                                                                                            | Số ký tự | Metadata chính                          |
| --- | --------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------- | -------- | --------------------------------------- |
| 1   | python_data_structures.md   | data_structures  | [https://docs.python.org/3/tutorial/datastructures.html](https://docs.python.org/3/tutorial/datastructures.html) | 25,910   | source_url, topic, difficulty, keywords |
| 2   | python_modules.md           | modules          | [https://docs.python.org/3/tutorial/modules.html](https://docs.python.org/3/tutorial/modules.html)               | 26,216   | source_url, topic, difficulty, keywords |
| 3   | python_errors_exceptions.md | errors           | [https://docs.python.org/3/tutorial/errors.html](https://docs.python.org/3/tutorial/errors.html)                 | 25,666   | source_url, topic, difficulty, keywords |
| 4   | python_classes.md           | oop              | [https://docs.python.org/3/tutorial/classes.html](https://docs.python.org/3/tutorial/classes.html)               | 37,950   | source_url, topic, difficulty, keywords |
| 5   | python_stdlib.md            | standard_library | [https://docs.python.org/3/tutorial/stdlib.html](https://docs.python.org/3/tutorial/stdlib.html)                 | 15,100   | source_url, topic, difficulty, keywords |
| 6   | python_venv.md              | environment      | [https://docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html)                     | 7,908    | source_url, topic, difficulty, keywords |
| 7   | python_input_output.md      | io               | [https://docs.python.org/3/tutorial/inputoutput.html](https://docs.python.org/3/tutorial/inputoutput.html)       | 22,341   | source_url, topic, difficulty, keywords |
| 8   | python_argparse.md          | cli              | [https://docs.python.org/3/library/argparse.html](https://docs.python.org/3/library/argparse.html)               | 95,464   | source_url, topic, difficulty, keywords |


### Metadata Schema


| Trường metadata | Kiểu         | Ví dụ giá trị                                                                                | Tại sao hữu ích cho retrieval?                |
| --------------- | ------------ | -------------------------------------------------------------------------------------------- | --------------------------------------------- |
| source_url      | string       | [https://docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html) | Trace câu trả lời về nguồn chính thức.        |
| doc_id          | string       | python_venv                                                                                  | Gom chunk theo tài liệu gốc.                  |
| doc_title       | string       | Virtual Environments and Packages                                                            | Hiển thị nguồn dễ đọc trong report/log.       |
| topic           | string       | environment                                                                                  | Dùng cho metadata filtering theo chủ đề.      |
| difficulty      | string       | beginner                                                                                     | Filter câu hỏi beginner/intermediate.         |
| doc_type        | string       | official_documentation                                                                       | Phân biệt tài liệu official với note tự viết. |
| keywords        | list[string] | venv, pip, dependency                                                                        | Hỗ trợ đọc hiểu nhanh nội dung tài liệu.      |
| chunk_index     | int          | 2                                                                                            | Xác định vị trí chunk trong document.         |


Metadata không chỉ dùng để trang trí. Ví dụ query “Virtual environment và pip giúp quản lý dependency conflict như thế nào?” có thể filter trước bằng `topic=environment`, sau đó search trong nhóm chunk nhỏ hơn.

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Strategy Của Tôi

**Strategy:** `RecursiveChunker(chunk_size=700)`

Tôi chọn RecursiveChunker vì bộ Python docs có heading, paragraph, code block và danh sách bullet. Fixed-size dễ cắt ngang câu hoặc cắt ngang code. Sentence chunking giữ câu tốt nhưng có thể tạo chunk quá ngắn hoặc quá dài với documentation có nhiều code. RecursiveChunker ưu tiên separator lớn trước (`\n\n`, `\n`, `.` , space), nên thường giữ section nhỏ và paragraph cùng nhau.

Code sử dụng trong `chat.py`:

```python
chunker = RecursiveChunker(chunk_size=700)
```

Khi chạy với 8 tài liệu Python docs, RecursiveChunker tạo:

```text
474 chunks
avg_length = 525.0 chars
min_length = 23 chars
max_length = 699 chars
```

### So Sánh 4 Strategies Trong Nhóm F2

Nhóm F2 có 4 thành viên. Mỗi người phụ trách một hướng chunking và so sánh trên cùng dataset Python Official Documentation, cùng 5 benchmark queries, cùng embedding backend `ollama:qwen3-embedding:0.6b`. Cách chạy này giữ phần đánh giá công bằng: khác nhau chủ yếu nằm ở strategy, không phải do đổi data hoặc đổi câu hỏi.

| Thứ tự | Thành viên       | MSV        | Strategy                                     | Chunk count | Avg length | Top-1 relevant | Top-3 relevant | Avg top-1 score |
| ------ | ---------------- | ---------- | -------------------------------------------- | ----------- | ---------- | -------------- | -------------- | --------------- |
| 1      | Lê Quốc Anh      | 2A202600824 | FixedSizeChunker(chunk_size=700, overlap=50) | 391         | 692.4      | 5/5            | 5/5            | 0.6733          |
| 2      | Lý Hải Long      | 2A202600568 | SentenceChunker(max_sentences_per_chunk=3)   | 478         | 521.4      | 5/5            | 5/5            | 0.6732          |
| 3      | Nguyễn Đức Khang | 2A202600588 | RecursiveChunker(chunk_size=700)             | 474         | 525.0      | 5/5            | 5/5            | 0.6833          |
| 4      | Nguyễn Đức Mạnh  | 2A202600945 | HeaderAwareChunker(chunk_size=700)           | 519         | 479.3      | 5/5            | 5/5            | 0.6783          |


**Kết luận strategy:**  
Cả 4 strategies đều retrieve đúng tài liệu trong top-3 cho 5/5 queries. RecursiveChunker có average top-1 score cao nhất trong lần chạy này: `0.6833`. Chênh lệch không lớn, nhưng RecursiveChunker cân bằng tốt giữa coherence và số lượng chunk. Fixed-size ít chunk hơn nhưng có rủi ro cắt ngang ý. Header-aware bám heading tốt nhưng tạo nhiều chunk hơn.

### Failure Case / Risk

Không có query nào fail top-3 trong benchmark chính. Failure risk nằm ở câu hỏi tiếng Việt vì tài liệu gốc là tiếng Anh. Nếu embedding model không mạnh về cross-lingual retrieval, câu hỏi tiếng Việt thuần có thể kém ổn định. Tôi giảm rủi ro bằng query song ngữ nhẹ, ví dụ giữ các keyword `Python`, `list comprehension`, `Module Search Path`, `try-except-finally`, `Virtual environment`, `pip`.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`**  
Tôi tách câu bằng regex `(?<=[.!?])(?:\s+|\n+)`, sau đó group theo `max_sentences_per_chunk`. Cách này giữ ranh giới câu và xử lý được text rỗng, whitespace và tham số nhỏ hơn 1.

**`RecursiveChunker.chunk` / `_split`**  
Thuật toán thử separator theo thứ tự từ lớn đến nhỏ. Nếu một đoạn vẫn dài hơn `chunk_size`, nó recurse với separator tiếp theo. Khi không còn separator phù hợp, nó fallback sang cắt theo ký tự. Cách này hợp với Markdown docs vì thường có heading, paragraph và line break rõ.

### EmbeddingStore

**`add_documents` + `search`**  
Mỗi `Document` được lưu thành record gồm `id`, `doc_id`, `content`, `metadata` và `embedding`. Khi search, store embed query, tính dot product với từng record, sort giảm dần theo score và trả về top-k.

**`search_with_filter` + `delete_document`**  
`search_with_filter` lọc metadata trước khi tính similarity. `delete_document` xóa tất cả chunk có cùng `doc_id`. Hai hàm này giúp vector store không chỉ search toàn cục mà còn hỗ trợ truy vấn theo phạm vi tài liệu.

### KnowledgeBaseAgent

`KnowledgeBaseAgent.answer()` chạy RAG theo 3 bước:

```text
1. store.search(question, top_k)
2. build prompt từ top-k chunks
3. gọi llm_fn(prompt) để sinh answer
```

Trong phần benchmark, tôi dùng `chat.py` với:

```text
Embedding backend: ollama:qwen3-embedding:0.6b
Chat backend: ollama-chat:qwen3.5:0.8b
Top-k: 3
Chat log: logs/python_docs_benchmark.jsonl
```

### Test Results

```text
.venv/bin/python -m pytest tests/ -v
42 tests collected
42 passed
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Embedding backend dùng để đo actual score: `OllamaEmbedder(qwen3-embedding:0.6b)`.

| Pair | Sentence A                                                     | Sentence B                                                               | Dự đoán | Actual Score | Đúng? |
| ---- | -------------------------------------------------------------- | ------------------------------------------------------------------------ | ------- | ------------ | ----- |
| 1    | List comprehension trong Python tạo list mới từ một iterable.  | Có thể dùng cú pháp comprehension thay cho vòng lặp for để tạo list.     | high    | 0.7331       | Đúng  |
| 2    | Khối try-except bắt lỗi xảy ra trong lúc chương trình chạy.    | Exception handling giúp Python xử lý lỗi runtime an toàn hơn.            | high    | 0.6353       | Đúng  |
| 3    | Virtual environment tách dependency cho từng project Python.   | pip cài đặt và quản lý package trong môi trường Python.                  | high    | 0.6211       | Đúng  |
| 4    | Class trong Python định nghĩa khuôn mẫu để tạo object.         | Lệnh pip install tải package từ PyPI về máy.                             | low     | 0.3374       | Đúng  |
| 5    | Module Python giúp tổ chức code thành nhiều file riêng.        | Môi trường ảo giúp tránh xung đột phiên bản dependency.                  | low     | 0.3540       | Đúng  |

Pair 4 và Pair 5 có score khoảng `0.33–0.35`, cao hơn mức “khác hoàn toàn” mà tôi kỳ vọng. Lý do có thể là các câu vẫn cùng nằm trong domain Python. Vì vậy similarity score chỉ nên là tín hiệu ban đầu; khi làm RAG vẫn cần kiểm tra top-k chunk và source document.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers

Tôi dùng đúng 5 câu hỏi benchmark đã chọn:


| #   | Query                                                                           | Gold Answer                                                                                                                                                |
| --- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Python list comprehension khác gì so với for loop khi tạo list mới?             | List comprehension tạo list mới bằng cú pháp ngắn gọn gồm expression và for clause; for loop làm cùng việc nhưng dài hơn và cần append thủ công.           |
| 2   | Python module import hoạt động như thế nào, và Module Search Path ảnh hưởng gì? | `import` nạp definitions từ module; Module Search Path quyết định Python tìm module ở thư mục script, PYTHONPATH và thư mục cài đặt chuẩn theo thứ tự nào. |
| 3   | try-except-finally / exception handling trong Python xử lý lỗi ra sao?          | `try` chạy code có thể lỗi, `except` bắt exception phù hợp, `finally` chạy cleanup dù có lỗi hay không; `raise` dùng để phát sinh hoặc phát lại exception. |
| 4   | Class, instance, attribute, method trong Python OOP khác nhau thế nào?          | Class định nghĩa kiểu object; instance là object tạo từ class; attribute là dữ liệu gắn với object/class; method là function thuộc class.                  |
| 5   | Virtual environment và pip giúp quản lý dependency conflict như thế nào?        | Virtual environment tách package theo từng project; pip cài, nâng cấp và quản lý package trong môi trường đó.                                              |


### Retrieval Results Với Strategy Của Tôi

Strategy: `RecursiveChunker(chunk_size=700)`  
Embedding backend: `ollama:qwen3-embedding:0.6b`  
Evaluation log: `logs/python_docs_retrieval_eval.json`


| #   | Expected doc             | Top-1 retrieved doc      | Score  | Top-3 relevant? |
| --- | ------------------------ | ------------------------ | ------ | --------------- |
| 1   | python_data_structures   | python_data_structures   | 0.7106 | Có              |
| 2   | python_modules           | python_modules           | 0.6490 | Có              |
| 3   | python_errors_exceptions | python_errors_exceptions | 0.7162 | Có              |
| 4   | python_classes           | python_classes           | 0.6460 | Có              |
| 5   | python_venv              | python_venv              | 0.6948 | Có              |


**Top-1 relevant:** 5 / 5  
**Top-3 relevant:** 5 / 5

### Chat.py Results

Tôi chạy `chat.py` với đúng 5 câu hỏi trên. Full log nằm ở:

```text
logs/python_docs_benchmark.jsonl
```


| #   | Query                                                                           | Chat answer summary                                                                         | Relevant?                                         |
| --- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| 1   | Python list comprehension khác gì so với for loop khi tạo list mới?             | Trả lời list comprehension ngắn gọn hơn và đưa ví dụ `squares = [x**2 for x in range(10)]`. | Có                                                |
| 2   | Python module import hoạt động như thế nào, và Module Search Path ảnh hưởng gì? | Trả lời về cách Python tìm module và ảnh hưởng của search path.                             | Có, nhưng answer hơi ngắn.                        |
| 3   | try-except-finally / exception handling trong Python xử lý lỗi ra sao?          | Trả lời `try/except/finally`, re-raise và cleanup.                                          | Có                                                |
| 4   | Class, instance, attribute, method trong Python OOP khác nhau thế nào?          | Trả lời đủ 4 khái niệm, nhưng câu cuối bị model cắt ở phần tóm tắt.                         | Có, cần kiểm soát `num_predict` nếu demo dài hơn. |
| 5   | Virtual environment và pip giúp quản lý dependency conflict như thế nào?        | Trả lời virtual environment tách dependency theo ứng dụng và version.                       | Có                                                |


Chat model trả lời được bằng tiếng Việt dù tài liệu nguồn là tiếng Anh. Tuy nhiên retrieval ổn định vì query giữ keyword tiếng Anh như `list comprehension`, `Module Search Path`, `try-except-finally`, `Virtual environment`, `pip`.

---

## 7. What I Learned (5 điểm — Demo)

### Điều học được từ phần so sánh nhóm

Chunking strategy không chỉ là chia text nhỏ ra. Với Python documentation, chunk phải giữ được section title, code snippet và paragraph giải thích. Nếu chunk bị cắt ngang code hoặc tách title khỏi nội dung, retrieval vẫn có thể đúng doc nhưng answer dễ thiếu chi tiết.

RecursiveChunker là lựa chọn tốt cho bộ docs này vì nó giữ cấu trúc tự nhiên của Markdown mà không tạo quá nhiều chunk nhỏ. Kết quả benchmark của tôi đạt `Top-3 relevant = 5/5`, và average top-1 score cao nhất trong 4 strategies được so sánh.

### Điều học được từ việc chạy chat.py

Retrieval đúng chưa đảm bảo answer hoàn hảo. Query 4 retrieve đúng `python_classes`, nhưng chat model `qwen3.5:0.8b` trả lời hơi dài và bị cắt ở cuối. Vấn đề này không nằm ở vector store mà nằm ở generation config (`num_predict=256`) và kích thước model nhỏ. Nếu demo chính thức, tôi sẽ tăng `num_predict` hoặc yêu cầu answer ngắn hơn.

### Nếu làm lại, tôi sẽ thay đổi gì?

Tôi sẽ giữ RecursiveChunker nhưng thêm section heading vào mỗi chunk rõ hơn. Với tài liệu official docs, heading như `5.1.3 List Comprehensions` hoặc `12.2 Creating Virtual Environments` rất quan trọng. Nếu chunk luôn giữ heading, answer dễ trích đúng phần hơn và log dễ kiểm chứng hơn.

---

## Tự Đánh Giá


| Tiêu chí               | Điểm tự đánh giá | Bằng chứng                                                      |
| ---------------------- | ---------------- | --------------------------------------------------------------- |
| Warm-up                | 5 / 5            | Đã trả lời cosine similarity và chunking math.                  |
| Document selection     | 10 / 10          | 8 official docs, Firecrawl success 8/8, metadata đầy đủ.        |
| Chunking strategy      | 15 / 15          | So sánh 4 strategies, RecursiveChunker đạt top-3 5/5.           |
| My approach            | 10 / 10          | Giải thích chunking, store, filter, delete, RAG agent.          |
| Similarity predictions | 5 / 5            | 5 pairs có prediction và actual score.                          |
| Results                | 10 / 10          | 5 benchmark queries, gold answers, chat.py log, retrieval eval. |
| Core implementation    | 30 / 30          | `42 passed`.                                                    |
| Retrieval quality      | 10 / 10          | Top-3 relevant = 5/5.                                           |
| Demo                   | 5 / 5            | Có kết quả so sánh và failure/risk analysis.                    |
| **Tổng**               | **100 / 100**    | Core 60 + Group 40 theo SCORING.md.                             |


---

## Evidence Files

```text
data/python_official_docs/
data/python_official_docs_metadata.json
scripts/scrape_python_docs.py
scripts/evaluate_python_docs.py
logs/python_docs_benchmark.jsonl
logs/python_docs_retrieval_eval.json
```
