# CLAUDE.md — Transaction Tagger

Đọc file này đầu mỗi phiên làm việc.

## 1. Mục tiêu

Nhận một câu ngắn mô tả giao dịch (tiếng Việt hoặc tiếng Anh, có thể không dấu, có lỗi do nhận dạng giọng nói) và gán nhãn cho từng từ.

Ví dụ: `hôm qua 5k đ gửi xe uit momo`

- Đầu vào: một chuỗi văn bản.
- Đầu ra: `tokens` và `tags`, cùng độ dài, định dạng BIO.
- Một model duy nhất gán nhãn cả câu trong một lượt. Không tách thành nhiều bước (tìm tiền trước, rồi tìm cái khác).
- Model cuối phải đủ nhỏ để chạy offline trên Android (Kotlin).
- Repo này chỉ lo phần Python: dữ liệu, train, đánh giá, xuất model.
- Hiện hỗ trợ `vi` và `en`. Thiết kế để thêm ngôn ngữ mới chỉ cần thêm dữ liệu rồi train lại, không sửa code (xem mục 13).

Ngoài phạm vi repo:

- Quy đổi số tiền (`5k` → 5000): code thường ở phía app làm, không phải model.
- Gợi ý category: xem mục 2.

## 2. Bộ nhãn và quy ước BIO

| Nhãn | Ý nghĩa | Ví dụ |
|---|---|---|
| `O` | Không thuộc cụm nào | `cho`, `nha` |
| `B-MONEY` / `I-MONEY` | Số tiền | `5k đ` |
| `B-DATE` / `I-DATE` | Thời gian | `hôm qua` |
| `B-ACCOUNT` / `I-ACCOUNT` | Tài khoản / ví | `momo` |
| `B-ITEM` / `I-ITEM` | Nội dung giao dịch | `gửi xe uit` |
| `B-TYPE` / `I-TYPE` | Loại: chi / thu / chuyển khoản | `chuyển khoản` |

Tổng cộng 11 nhãn. Thứ tự nhãn (ánh xạ nhãn ↔ số) cố định trong `src/schema.py`; không đổi thứ tự sau khi đã xuất model.

Bộ nhãn dùng chung cho mọi ngôn ngữ. Không thêm nhãn riêng cho ngôn ngữ nào.

Quy ước BIO:

- `B-X` mở đầu một cụm loại X. `I-X` là từ tiếp theo của cùng cụm đó.
- `I-X` chỉ hợp lệ khi từ ngay trước là `B-X` hoặc `I-X` (cùng loại X).
- Hai cụm cùng loại đứng sát nhau: cụm thứ hai vẫn bắt đầu bằng `B-X`.

Không có nhãn CATEGORY: người dùng tự tạo category nên tập category khác nhau ở mỗi người, model không thể học một bộ nhãn cố định. Bước gợi ý category nằm ngoài repo này.

## 3. Định dạng dữ liệu (JSONL)

Mỗi dòng là một object JSON:

```json
{"id": "syn-000001", "lang": "vi", "source": "synthetic", "tokens": ["hôm", "qua", "5k", "đ", "gửi", "xe", "uit", "momo"], "tags": ["B-DATE", "I-DATE", "B-MONEY", "I-MONEY", "B-ITEM", "I-ITEM", "I-ITEM", "B-ACCOUNT"]}
```

- `id`: chuỗi duy nhất trong toàn bộ dữ liệu.
- `lang`: mã ISO 639-1, phải thuộc `LANGS` trong `src/schema.py` (hiện là `["vi", "en"]`). Đây là chỗ duy nhất khai báo danh sách ngôn ngữ; code khác không viết cứng `vi`/`en`.
- `source`: `synthetic` hoặc `real`.
- `tokens`, `tags`: hai danh sách cùng độ dài.

Thư mục dữ liệu:

- `data/raw/<lang>/`: mảnh gốc và template của từng ngôn ngữ, mỗi loại một file: `money.txt`, `date.txt`, `account.txt`, `item.txt`, `type.txt`, `templates.txt`. `gen_data.py` đọc mọi thư mục có tên trong `LANGS`.
- `data/synth/`: dữ liệu sinh ra (train / val / test), gộp mọi ngôn ngữ. Không commit, sinh lại bằng `gen_data.py`.
- `data/real_test/<lang>.jsonl`: tập test thật gán tay, mỗi ngôn ngữ một file (tổng 100-200 câu). Có commit.

Nhiễu trong `gen_data.py`: mỗi hàm nhiễu khai báo áp dụng cho ngôn ngữ nào. Ví dụ: bỏ dấu chỉ dùng cho `vi`.

## 4. Quy ước kỹ thuật

- Tối đa 100 từ/câu, 16 ký tự/từ. Dài hơn thì cắt. 100 cũng là số vị trí của positional embedding, không đổi sau khi đã xuất model.
- `<PAD>` = 0, `<UNK>` = 1 (cho cả từ điển từ và từ điển ký tự).
- Nhãn ở vị trí đệm = -100 (bỏ qua khi tính loss và metric).
- Seed cố định cho `random`, `numpy`, `torch`. Cùng seed phải cho cùng kết quả.
- Tensor đầu vào model: `word_ids` (100), `char_ids` (100 × 16), `tag_ids` (100). Mask đệm suy ra từ `word_ids != 0`, không lưu riêng.
- Attention phải chặn vị trí đệm (gán điểm attention = -inf trước softmax). Không dùng causal mask: mỗi từ được nhìn cả hai phía.
- Một model chung cho mọi ngôn ngữ, không có đầu vào `lang` (câu có thể trộn vi-en). Từ điển từ và từ điển ký tự xây từ tập train gộp mọi ngôn ngữ.

## 5. Chuẩn hóa văn bản

`src/normalize.py` phải khớp 100% với bản viết lại bằng Kotlin.

- Các bước: hạ chữ thường, gom khoảng trắng, tách từ bằng khoảng trắng.
- Chỉ dùng thao tác có tương đương rõ ràng trong Kotlin. Không dùng regex phức tạp hay hành vi riêng của Python.
- Không rẽ nhánh theo ngôn ngữ: mọi câu đi qua cùng các bước.
- Chỉ hỗ trợ ngôn ngữ tách từ bằng khoảng trắng. Tiếng Trung, Nhật, Thái nằm ngoài phạm vi vì cần bộ tách từ riêng.
- Test vector phải có chữ hoa ngoài ASCII của mọi ngôn ngữ trong `LANGS`, để bắt chỗ lệch giữa `str.lower()` và `lowercase()` của Kotlin.
- Mọi thay đổi ở `normalize.py` bắt buộc phải sinh lại test vector trong `export/`.

## 6. Ràng buộc thư viện

Được dùng:

- `numpy`
- `torch`: `nn.Embedding`, `nn.Linear`, `nn.LayerNorm`, `nn.Dropout`, `nn.Conv1d`, optimizer
- `pytest`

Phải tự viết: multi-head self-attention, khối encoder, vòng train, hàm loss, metric, data loader, F1 theo cụm (entity-level).

Không dùng `nn.MultiheadAttention`, `nn.TransformerEncoder(Layer)`, `F.scaled_dot_product_attention`: phía Kotlin phải viết lại từng phép tính, nên Python cũng viết tay để hai bên khớp. Hàm kích hoạt FFN dùng ReLU (dễ viết lại hơn GELU).

Không dùng: HuggingFace/transformers, spaCy, NLTK, seqeval, scikit-learn, PyTorch Lightning, Trainer API.

Ngoại lệ: scikit-learn chỉ được dùng khi người dùng chủ động yêu cầu làm baseline.

Quản lý môi trường bằng `uv`, không dùng `pip` hay `uv pip`:

- Thư viện khai báo trong `pyproject.toml`, phiên bản chốt trong `uv.lock` (cả hai đều commit). Python 3.14 (`.python-version`).
- Thêm thư viện: `uv add <tên>`; thư viện chỉ dùng khi phát triển (pytest): `uv add --dev <tên>`. Cài môi trường: `uv sync`. Chạy: `uv run ...`.
- `[tool.uv] package = false`: script chạy thẳng từ `src/`, không build package.
- `torch` lấy từ index CPU của PyTorch (`[tool.uv.sources]`), không cần CUDA.

## 7. Nguyên tắc code

- Đơn giản nhất có thể: thư mục phẳng, mỗi file một việc.
- Ưu tiên hàm thay vì class. Chỉ model mới dùng class.
- Không framework cấu hình, không cấu trúc package phức tạp.
- Script chỉ dùng `argparse` tối thiểu.
- Comment và tài liệu bằng tiếng Việt.

Chế độ vừa học vừa làm: mục đích của repo là để học. Phần cốt lõi (self-attention, backward, vòng train) phải có comment ngắn giải thích ý nghĩa từng bước.

## 8. Cấu trúc repo

```
CLAUDE.md
README.md
pyproject.toml     # thư viện + cấu hình uv
uv.lock            # phiên bản chốt (uv tự sinh)
src/
  schema.py        # bộ nhãn, ánh xạ nhãn <-> số, danh sách ngôn ngữ LANGS
  normalize.py     # chuẩn hóa + tách từ (phải dễ viết lại y hệt bằng Kotlin)
  gen_data.py      # sinh dữ liệu template + nhiễu
  check_data.py    # kiểm tra JSONL
  metrics.py       # F1 theo cụm
  baseline.py      # baseline rule-based
  attention_numpy.py # self-attention thuần NumPy
  model_torch.py   # Transformer encoder nhỏ (+ char-CNN tùy chọn)
  train.py         # vòng train tự viết
  export.py        # xuất model + test vector
data/
  raw/<lang>/      # mảnh gốc + template theo ngôn ngữ
  synth/           # dữ liệu sinh (không commit)
  real_test/       # <lang>.jsonl, test thật gán tay
tests/
export/            # model, vocab, test vector (đầu ra, không commit)
```

## 9. Lịch làm việc

- 2-3 giờ/ngày, thứ Hai đến thứ Bảy, nghỉ Chủ nhật. Tổng khoảng 33 giờ.
- Bắt đầu thứ Hai 05/10/2026, kết thúc thứ Bảy 17/10/2026.

| Module | Nội dung | Giờ | Deadline |
|---|---|---|---|
| M0 | Khởi tạo repo, CLAUDE.md, README khung, thư mục, git init | 2 | T2 05/10 |
| M1 | Schema nhãn, chuẩn hóa văn bản, định dạng JSONL, script kiểm tra dữ liệu (độ dài tokens = tags, BIO hợp lệ) | 5 | T4 07/10 |
| M2 | Bộ sinh dữ liệu từ template (ghép mảnh theo nhiều thứ tự), nhiễu (bỏ dấu, viết tắt, lỗi gõ, từ đệm), chia train/val/test theo mảnh gốc, tập test thật gán tay (100-200 câu) | 8 | T6 09/10 |
| M3 | Metric F1 theo cụm tự viết + baseline rule-based làm mốc so sánh | 3 | T7 10/10 |
| M4 | Self-attention thuần NumPy (1 head, có mask đệm, residual, positional embedding): forward, backward, kiểm tra gradient, train thử trên dữ liệu mini | 6 | T3 13/10 |
| M5 | Transformer encoder nhỏ (có thể thêm char-CNN) bằng PyTorch, attention tự viết: tensor hóa dữ liệu (100 từ × 16 ký tự), vòng train tự viết, early stopping, chấm điểm bằng metric ở M3 | 6 | T5 15/10 |
| M6 | Phân tích lỗi trên tập test thật, bổ sung kiểu câu bị sai vào bộ sinh, train lại | 2.5 | T6 16/10 |
| M7 | Xuất model (trọng số .npz hoặc JSON + từ điển từ/ký tự + danh sách nhãn + cấu hình), 100 test vector cho Kotlin, README hướng dẫn tích hợp | 3 | T7 17/10 |

Khi bị trễ, cắt theo thứ tự:

1. Bỏ baseline ở M3.
2. Bỏ char-CNN.
3. Bỏ ONNX (vốn đã là phần tùy chọn).

Không được cắt: tập test thật, metric F1, test vector.

## 10. Quy tắc đánh giá

- Chỉ dùng tập test thật (`data/real_test/`) để kết luận chất lượng model. Điểm trên dữ liệu sinh chỉ để theo dõi quá trình train.
- Không để cùng một mảnh gốc xuất hiện ở cả train lẫn test. Chia train/val/test theo mảnh gốc, không chia theo câu đã ghép. Chia riêng trong từng ngôn ngữ.
- Metric chính: F1 theo cụm, báo riêng cho từng loại nhãn, từng ngôn ngữ, và trung bình chung.

## 11. Jira

- Site: https://ai-engineer-learning.atlassian.net
- Mã project: `TTTM` (tên `Text-to-transaction-map`, team-managed, board Scrum id 34). Key đề xuất ban đầu là `TTAG`; project được tạo thủ công với key `TTTM`.
- Sprint 1 "Dữ liệu và nền tảng" (id 37): 05/10 → 10/10/2026 (M0-M3).
- Sprint 2 "Model và xuất" (id 38): 12/10 → 17/10/2026 (M4-M7).
- 04/10/2026: đổi kiến trúc từ BiLSTM sang Transformer encoder (giới hạn 100 từ/câu). Đã sửa epic M4, M5 và các task liên quan.
- 05/10/2026: thiết kế mở rộng đa ngôn ngữ (`LANGS`, dữ liệu theo thư mục ngôn ngữ, F1 theo ngôn ngữ). Đã sửa mô tả TTTM-2, 3, 12, 13, 14, 16, 17, 18, 20, 27, 31, 34, 35, 36.
- Epic: `TTTM-1` (M0) đến `TTTM-8` (M7). Task: `TTTM-9` đến `TTTM-37`.
- Ước lượng giờ ghi ở trường "Story point estimate" (1 điểm = 1 giờ) và trong mô tả. Project không có trường time tracking.
- Quy ước đặt tên issue:
  - Epic: `M<số>: <tên module>`, ví dụ `M1: Schema và chuẩn hóa`.
  - Task: `[M<số>] <việc cần làm>`, ví dụ `[M1] Viết check_data.py`.
  - Task cuối mỗi module: `[M<số>] Ghi chú học được` (15 phút).
- Tên issue và mô tả bằng tiếng Việt. Mỗi task có due date, ước lượng giờ, tiêu chí hoàn thành đo được.

## 12. Cách làm việc với Claude

- Đọc `CLAUDE.md` đầu mỗi phiên.
- Khi viết phần cốt lõi (self-attention, backward, vòng train): thêm comment ngắn từng bước trong code và giải thích ngắn bằng tiếng Việt trong câu trả lời.
- Ưu tiên giải pháp đơn giản. Không thêm thư viện, class, hay lớp cấu hình khi chưa cần.
- Không dùng thư viện bị cấm ở mục 6, kể cả khi tiện hơn.
- Sửa `normalize.py` thì phải sinh lại test vector.
- Commit message bằng tiếng Anh, có tiền tố module + task Jira (`[M0][TTTM-9] [Chore] ...`), theo skill `.claude/skills/tttm-commit-message/`.

## 13. Thêm ngôn ngữ mới

Chỉ áp dụng cho ngôn ngữ tách từ bằng khoảng trắng. Không sửa code ngoài bước 1 và bước 3.

1. Thêm mã ngôn ngữ vào `LANGS` trong `src/schema.py`.
2. Tạo `data/raw/<lang>/` với đủ 6 file mảnh gốc và template.
3. Khai báo hàm nhiễu nào áp dụng cho ngôn ngữ mới; thêm hàm nhiễu riêng nếu cần.
4. Gán tay `data/real_test/<lang>.jsonl` (ít nhất 30 câu), chạy `check_data.py`.
5. Sinh lại dữ liệu, train lại từ đầu (từ điển đổi nên embedding đổi).
6. Xuất lại model và test vector; kiểm tra F1 của ngôn ngữ mới trên tập test thật.
