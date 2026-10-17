# PROCESS

**Status:** DONE
**Started:** 2026-10-17
**Last updated:** 2026-10-17

## Task
"các embedding hiện tại hơi quá lớn (gần 1GB cho cả vi/en), hãy giúp tôi tìm các embedding nhẹ hơn (max = 500mb cho cả vn/en), sau đó update vào plan"

## Plan
1. [x] Đo kích thước nguồn hiện tại + tìm thay thế
2. [x] Đo độ phủ + PCA trên dữ liệu thật
3. [x] Cập nhật CLAUDE.md (mục 4, 8, 9, 11, 13)
4. [x] Cập nhật Jira TTTM-27 (mô tả + 2.5h), TTTM-34, TTTM-36, TTTM-6 epic, due date 28/29/30/33

Quyết định: fastText cc.<lang>.300.vec.gz cho vi + en, tải 100k dòng đầu (~67 MB/lang), PCA 300->100, script mới src/get_embeddings.py.
Lý do: PhoW2V cấm phân phối (không nhúng vào app được); glove.6B.zip = 862 MB.

## Done
- Số đo: en 50k dòng = 33.4 MB tải; PCA 100d giữ 57% phương sai (vi); độ phủ gộp top20k: train 0.894, real vi 0.895, real en 0.835.

## Not done yet
- Commit CLAUDE.md (chưa commit, chờ người dùng).

## Notes for resuming
- Scratchpad có cc.vi.200k.vec (47k dòng), cc.en.50k.vec, stream_head.py, coverage.py.
