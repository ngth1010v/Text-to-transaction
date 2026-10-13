import json
from pathlib import Path

import attention_numpy as an
import numpy as np


def test_shape():
    params = an.init_params(100)

    word_ids = np.array([1,2,3,4,5])
    word_ids = an.pad_words(word_ids)
    logits, cache = an.forward(params, word_ids)

    assert logits.shape == (100,11)



def test_initial_loss():
    params = an.init_params(100)

    word_ids = np.array([1,2,3,4,5])
    word_ids = an.pad_words(word_ids)
    logits, cache = an.forward(params, word_ids)

    tag_ids = np.array([1,2,3,4,5])
    tag_ids = an.pad_tags(tag_ids)
    loss, probs = an.loss_fn(logits, tag_ids)
    
    assert np.abs(loss - 2.4) < 0.2



def test_ignore_pad():
    params = an.init_params(100)

    word_ids = an.pad_words(np.array([1, 2, 3, 4, 5]))
    logits, cache = an.forward(params, word_ids)

    tags = [1, 2, 3, 4, 5]
    loss, probs = an.loss_fn(logits, an.pad_tags(np.array(tags)))

    # tính tay chỉ trên 5 từ thật
    expected = -np.mean(np.log(probs[np.arange(5), tags]))

    assert np.allclose(loss, expected)



def test_mask():
    params = an.init_params(100)

    word_ids = an.pad_words(np.array([1, 2, 3, 4, 5]))
    before, cache = an.forward(params, word_ids)
    A = cache["A"]

    # không từ nào chú ý vào cột pad
    assert np.all(A[:, 5:] == 0)
    # mỗi hàng vẫn là phân phối xác suất
    assert np.allclose(A.sum(axis=1), 1)

    # đổi embedding của <PAD> (id 0): 5 hàng thật không được đổi
    params["E"][0] += 5.0
    after, _ = an.forward(params, word_ids)
    assert np.allclose(before[:5], after[:5])



def test_gradient():
    # std lớn để gradient không quá nhỏ, tránh nhiễu số học làm sai lệch giả
    params = an.init_params(20, std=0.5)

    # 7 từ, có từ lặp (id 3 và 5) để bắt lỗi cộng dồn gradient của embedding
    word_ids = an.pad_words(np.array([3, 5, 7, 5, 9, 2, 3]))
    tag_ids = an.pad_tags(np.array([1, 2, 0, 4, 5, 6, 1]))

    errs = an.grad_check(params, word_ids, tag_ids)

    # phải có gradient của đủ 8 tham số: embedding từ, vị trí, Wq, Wk, Wv, Wo, lớp ra
    assert set(errs) == {"E", "P", "Wq", "Wk", "Wv", "Wo", "Wout", "Bout"}
    for key, err in errs.items():
        assert err < 1e-5, f"{key}: sai lệch {err}"


# ---- train thử trên 20 câu mini (TTTM-25) ----

def _load_mini():
    path = Path(__file__).parent / "data" / "mini_train.jsonl"
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    words = sorted({w for r in rows for w in r["tokens"]})
    word2id = {w: i for i, w in enumerate(words, start=1)}
    return rows, word2id


def test_overfit():
    # model đúng phải học thuộc được 20 câu: loss < 0.1 và đoán đúng 100% nhãn
    rows, word2id = _load_mini()
    params, history = an.train_mini(rows, word2id, epochs=300, lr=0.1)

    assert history[-1] < 0.1
    assert an.accuracy(params, rows, word2id) == 1.0


def test_same_seed():
    # cùng seed (không có yếu tố ngẫu nhiên khác) thì đường loss phải giống hệt
    rows, word2id = _load_mini()
    _, h1 = an.train_mini(rows, word2id, epochs=5, seed=0)
    _, h2 = an.train_mini(rows, word2id, epochs=5, seed=0)

    assert h1 == h2
