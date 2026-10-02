# Transaction Tagger

Gán nhãn BIO cho từng từ trong một câu mô tả giao dịch (tiếng Việt hoặc tiếng Anh).

```
hôm qua 5k đ gửi xe uit momo
B-DATE I-DATE B-MONEY I-MONEY B-ITEM I-ITEM I-ITEM B-ACCOUNT
```

Bộ nhãn: `O`, `MONEY`, `DATE`, `ACCOUNT`, `ITEM`, `TYPE`. Chi tiết quy ước ở [CLAUDE.md](CLAUDE.md).

## Cài đặt

```
pip install numpy torch pytest
```

## Cách chạy

TODO: bổ sung dần theo từng module.

- Sinh dữ liệu: `python src/gen_data.py` (M2)
- Kiểm tra dữ liệu: `python src/check_data.py <file.jsonl>` (M1)
- Train: `python src/train.py` (M5)
- Xuất model: `python src/export.py` (M7)
- Test: `pytest`

## Tích hợp Kotlin

TODO (M7): định dạng file xuất, cách nạp trọng số, cách kiểm tra khớp bằng test vector.
