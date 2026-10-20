"""Vòng train tự viết, early stopping, chấm điểm bằng metrics.py."""

from pathlib import Path
import numpy as np
import torch

import schema
from collections import Counter

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



def build_vocab(rows, top_n=20000, seed=90):

    counter = Counter(t for r in rows for t in r["tokens"])

    vectors = {}
    for lang in schema.LANGS:
        for w, v in load_vectors(EMB_DIR / f"{lang}.txt", top_n, set(counter)).items():
            vectors.setdefault(w, v)


    all_words = list(dict.fromkeys(list(vectors) + list(counter)))  
    words = ["<PAD>", "<UNK>"] + sorted(all_words, key=lambda w: -counter[w])
    word2id = {w: i for i, w in enumerate(words)}


    mat = np.stack(list(vectors.values()))    
    dim, std = mat.shape[1], mat.std()
    rng = np.random.default_rng(seed)


    emb = np.zeros((len(words), dim), dtype=np.float32)  
    emb[UNK] = mat.mean(axis=0)
    for w in words[2:]:                            
        emb[word2id[w]] = vectors[w] if w in vectors else rng.normal(0, std, dim)


    for lang in schema.LANGS:
        toks = [t for r in rows if r["lang"] == lang for t in r["tokens"]]
        print(f"{lang}: {sum(t in vectors for t in toks) / len(toks):.3f}")


    chars = sorted({c for w in counter for c in w})
    char2id = {c: i for i, c in enumerate(chars, start=2)}
    char2id["<PAD>"], char2id["<UNK>"] = PAD, UNK


    return word2id, char2id, emb


def encode(row, word2id, char2id):
    """Một dòng JSONL -> (word_ids, char_ids, tag_ids), độ dài cố định."""
    tokens = row["tokens"][:MAX_WORDS]  # cắt câu dài
    n_pad = MAX_WORDS - len(tokens)

    word_ids = [word2id.get(t, UNK) for t in tokens] + [PAD] * n_pad

    char_ids = []
    for t in tokens:
        ids = [char2id.get(c, UNK) for c in t[:MAX_CHARS]]  # cắt từ dài
        char_ids.append(ids + [PAD] * (MAX_CHARS - len(ids)))
    char_ids += [[PAD] * MAX_CHARS for _ in range(n_pad)]  # vị trí đệm: 16 số 0

    # nhãn đệm = -100 (0 là nhãn thật "O")
    tag_ids = [schema.TAG2ID[t] for t in row["tags"][:MAX_WORDS]] + [IGNORE] * n_pad
    return word_ids, char_ids, tag_ids


def encode_all(rows, word2id, char2id):
    """Mọi dòng -> 3 tensor long (N, 100), (N, 100, 16), (N, 100) và list lang."""
    enc = [encode(r, word2id, char2id) for r in rows]
    word_ids = torch.tensor([e[0] for e in enc], dtype=torch.long)
    char_ids = torch.tensor([e[1] for e in enc], dtype=torch.long)
    tag_ids = torch.tensor([e[2] for e in enc], dtype=torch.long)
    return word_ids, char_ids, tag_ids, [r["lang"] for r in rows]


def iterate_batches(n, batch_size, shuffle=False, generator=None):
    """Trả lần lượt các tensor chỉ số; batch cuối có thể nhỏ hơn."""
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]
