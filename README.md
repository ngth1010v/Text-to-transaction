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

## Cách chạy

TODO: bổ sung dần theo từng module.

- Sinh dữ liệu: `uv run python src/gen_data.py` (M2)
- Kiểm tra dữ liệu: `uv run python src/check_data.py <file.jsonl>` (M1)
- Train: `uv run python src/train.py` (M5)
- Xuất model: `uv run python src/export.py` (M7)
- Test: `uv run pytest`

## Tích hợp Kotlin

TODO (M7): định dạng file xuất, cách nạp trọng số, cách kiểm tra khớp bằng test vector.
