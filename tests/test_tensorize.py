import numpy as np
import pytest

from train import load_vectors


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_vectors(tmp_path):
    w2v = write(tmp_path, "vi.txt", "4 2\nCà 1 2\ncà 9 9\nphê 3 4\nxe 5 6\n")
    glove = write(tmp_path, "en.txt", "coffee 1 2\ntea 3 4\nmomo 5 6\n")

    v = load_vectors(w2v, top_n=10)
    assert list(v) == ["cà", "phê", "xe"]  # bỏ header, hạ thường, giữ bản đầu khi trùng
    assert np.array_equal(v["cà"], [1, 2])

    # top_n=1 nhưng "momo" có trong train -> vẫn giữ
    g = load_vectors(glove, top_n=1, train_words={"momo"})
    assert list(g) == ["coffee", "momo"]


def test_load_vectors_bad_dim(tmp_path):
    bad = write(tmp_path, "bad.txt", "a 1 2\nb 1 2 3\n")
    with pytest.raises(ValueError):
        load_vectors(bad)


# ---- build_vocab / encode ----

import torch

import train
from get_embeddings import pca
from schema import TAG2ID


@pytest.fixture
def vocab(tmp_path, monkeypatch):
    """Embedding giả 4 chiều cho vi + en; train có 3 câu."""
    write(tmp_path, "vi.txt", "3 4\ncà 1 0 0 0\nphê 0 1 0 0\ntrà 0 0 1 0\n")
    write(tmp_path, "en.txt", "tea 0 0 0 1\ncoffee 1 1 0 0\nmomo 2 2 2 2\n")
    monkeypatch.setattr(train, "EMB_DIR", tmp_path)
    rows = [
        {"lang": "vi", "tokens": ["cà", "phê", "30k", "momo"], "tags": ["B-ITEM", "I-ITEM", "B-MONEY", "B-ACCOUNT"]},
        {"lang": "vi", "tokens": ["cà", "30k"], "tags": ["B-ITEM", "B-MONEY"]},
        {"lang": "en", "tokens": ["tea", "30k"], "tags": ["B-ITEM", "B-MONEY"]},
    ]
    return rows, *train.build_vocab(rows, top_n=10, seed=90)


def test_pca():
    x = np.random.default_rng(0).normal(size=(50, 8)) * np.arange(1, 9)
    y = pca(x, 3)
    assert y.shape == (50, 3)
    cov = np.cov(y.T)
    assert np.allclose(cov - np.diag(np.diag(cov)), 0, atol=1e-8)  # các cột trực giao
    assert np.all(np.diff(np.diag(cov)) < 0)  # phương sai giảm dần


def test_shapes(vocab):
    rows, w2i, c2i, emb = vocab
    w, c, t, langs = train.encode_all(rows, w2i, c2i)
    assert w.shape == (3, 100) and c.shape == (3, 100, 16) and t.shape == (3, 100)
    assert w.dtype == c.dtype == t.dtype == torch.long
    assert langs == ["vi", "vi", "en"]
    assert emb.shape == (len(w2i), 4)


def test_special_ids(vocab):
    rows, w2i, c2i, emb = vocab
    assert w2i["<PAD>"] == 0 and w2i["<UNK>"] == 1
    assert c2i["<PAD>"] == 0 and c2i["<UNK>"] == 1
    assert np.all(emb[0] == 0)
    w, c, t, _ = train.encode_all(rows, w2i, c2i)
    assert t[1, 2:].eq(-100).all() and t[1, 0] == TAG2ID["B-ITEM"]
    assert (w == 0).sum() == 3 * 100 - 8  # mask đệm suy ra từ word_ids != 0
    assert c[1, 2:].eq(0).all()


def test_pretrained_row(vocab):
    _, w2i, _, emb = vocab
    assert np.array_equal(emb[w2i["phê"]], [0, 1, 0, 0])
    assert np.array_equal(emb[w2i["tea"]], [0, 0, 0, 1])


def test_unseen_but_pretrained(vocab):
    _, w2i, _, _ = vocab
    assert w2i["trà"] > 1  # có trong embedding, không có trong train


def test_unk(vocab):
    rows, w2i, c2i, _ = vocab
    w, c, _, _ = train.encode_all([{"lang": "vi", "tokens": ["zzz"], "tags": ["O"]}], w2i, c2i)
    assert w[0, 0] == 1 and c[0, 0, 0] == 1  # từ lạ và ký tự lạ -> UNK


def test_truncate(vocab):
    _, w2i, c2i, _ = vocab
    row = {"lang": "vi", "tokens": ["cà"] * 120, "tags": ["O"] * 120}
    w, _, t = train.encode(row, w2i, c2i)
    assert len(w) == len(t) == 100

    row = {"lang": "vi", "tokens": ["c" * 20], "tags": ["O"]}
    _, c, _ = train.encode(row, w2i, {"c": 2})
    assert len(c[0]) == 16 and all(i == 2 for i in c[0])  # chỉ 16 ký tự đầu


def test_deterministic(vocab):
    rows, w2i, c2i, emb = vocab
    w2, c2, e2 = train.build_vocab(rows, top_n=10, seed=90)
    assert w2 == w2i and c2 == c2i and np.array_equal(e2, emb)


def test_iterate_batches():
    g = torch.Generator().manual_seed(0)
    batches = list(train.iterate_batches(10, 4, shuffle=True, generator=g))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(torch.cat(batches).tolist()) == list(range(10))
    assert torch.cat(list(train.iterate_batches(5, 2))).tolist() == [0, 1, 2, 3, 4]
