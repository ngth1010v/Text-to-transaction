# Transaction Tagger

Gán nhãn BIO cho từng từ trong một câu mô tả giao dịch (tiếng Việt hoặc tiếng Anh).

```
hôm qua 5k đ gửi xe uit momo
B-DATE I-DATE B-MONEY I-MONEY B-ITEM I-ITEM I-ITEM B-ACCOUNT
```

Bộ nhãn: `O`, `MONEY`, `DATE`, `ACCOUNT`, `ITEM`, `TYPE`. Chi tiết quy ước ở [CLAUDE.md](CLAUDE.md).

## Cài đặt

Dùng [uv](https://docs.astral.sh/uv/), không dùng `pip`. Thư viện khai báo trong `pyproject.toml`, phiên bản chốt trong `uv.lock`, Python 3.14 (`.python-version`). `torch` lấy bản CPU.

```
uv sync
```

Thêm thư viện:

```
uv add <tên>          # thư viện chạy
uv add --dev <tên>    # chỉ dùng khi phát triển (pytest...)
```

## Embedding (fastText)

Model dùng embedding từ có sẵn, đóng băng khi train, cùng một nguồn cho mọi ngôn ngữ trong `LANGS`.

- Nguồn: [fastText Common Crawl](https://fasttext.cc/docs/en/crawl-vectors.html) (`cc.<lang>.300.vec.gz`) của Facebook AI Research.
- Giấy phép: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Được nhúng vào app (kể cả thương mại) nhưng phải ghi nguồn; bảng embedding đã lọc trong app là bản dẫn xuất nên cũng phải ghi nguồn này.
- `get_embeddings.py` chỉ tải phần đầu file (mặc định 100 000 dòng/ngôn ngữ, khoảng 67 MB mỗi ngôn ngữ, không tải cả file 1.2 GB), giảm 300 → 100 chiều bằng PCA, rồi ghi `data/embeddings/<lang>.txt` (khoảng 85 MB/ngôn ngữ, không commit).

```
uv run python src/get_embeddings.py [--lines 100000] [--dim 100]
```

Mạng chậm thì mất khoảng 25 phút; nên chạy ở nền. Cần chạy trước khi train.

Trích dẫn: E. Grave, P. Bojanowski, P. Gupta, A. Joulin, T. Mikolov, *Learning Word Vectors for 157 Languages*, LREC 2018.

## Cách chạy

TODO: bổ sung dần theo từng module.

- Tải embedding: `uv run python src/get_embeddings.py` (M5)
- Sinh dữ liệu: `uv run python src/gen_data.py` (M2)
- Kiểm tra dữ liệu: `uv run python src/check_data.py <file.jsonl>` (M1)
- Baseline rule-based trên tập test thật: `uv run python src/baseline.py` (M3)
- Train: `uv run python src/train.py` (M5)
- Xuất model: `uv run python src/export.py` (M7)
- Test: `uv run pytest`

## Kết quả

F1 theo cụm trên `data/real_test/` (232 câu: 118 vi, 114 en).

| Loại | Baseline (all) | Baseline (vi) | Baseline (en) |
|---|---|---|---|
| MONEY | 0.979 | 0.975 | 0.983 |
| DATE | 0.861 | 0.860 | 0.863 |
| ACCOUNT | 0.919 | 0.918 | 0.921 |
| ITEM | 0.830 | 0.770 | 0.894 |
| TYPE | 0.918 | 0.833 | 0.990 |
| **Micro** | **0.902** | **0.873** | **0.932** |

Baseline (`src/baseline.py`) tra từ điển lấy từ `data/raw/` cộng vài luật (số tiền, từ gợi ý account, từ nối). Từ điển không chia split nên điểm bị thổi lên: cụm có trong từ điển gần như luôn đúng. Recall trên cụm **không có** trong từ điển:

- ACCOUNT: 5/16.
- DATE: 1/23.

Model cần vượt mạnh ở hai chỗ này: account tự đặt và cách nói ngày chưa gặp.

## Tích hợp Kotlin

TODO (M7): định dạng file xuất, cách nạp trọng số, cách kiểm tra khớp bằng test vector.
