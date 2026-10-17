"""Vòng train tự viết, early stopping, chấm điểm bằng metrics.py."""

from pathlib import Path

import numpy as np

MAX_WORDS = 100
MAX_CHARS = 16
PAD = 0
UNK = 1
IGNORE = -100
EMB_DIR = Path("data/embeddings")


def load_vectors(path, top_n=20000, train_words=frozenset()):
    """Đọc file embedding dạng text (word2vec hoặc GloVe) -> dict từ -> np.array float32.

    Giữ `top_n` từ đầu file (file xếp theo tần suất giảm dần) cộng mọi từ có trong `train_words`.
    """
    vectors = {}
    dim = None
    missing = set(train_words)  # từ train chưa tìm thấy, hết thì dừng đọc sớm

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip().split(" ")

            # Header word2vec: "số_từ số_chiều" (chỉ 2 số). GloVe không có dòng này.
            if line_no == 1 and len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                continue
            if len(parts) < 2:  # dòng rỗng / hỏng
                continue

            word = parts[0].lower()  # giống normalize()
            if word in vectors:  # trùng sau khi hạ thường: giữ bản xuất hiện trước (tần suất cao hơn)
                continue

            in_top = len(vectors) < top_n
            if not in_top and word not in missing:
                if not missing:
                    break  # đã đủ top_n và đủ mọi từ train
                continue

            vec = np.asarray(parts[1:], dtype=np.float32)
            if dim is None:
                dim = vec.shape[0]
            elif vec.shape[0] != dim:
                raise ValueError(f"{path}:{line_no}: từ {word!r} có {vec.shape[0]} chiều, mong đợi {dim}")

            vectors[word] = vec
            missing.discard(word)

    return vectors
