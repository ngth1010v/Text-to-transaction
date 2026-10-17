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
