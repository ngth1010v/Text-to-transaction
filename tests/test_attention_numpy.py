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

    # buổi 1: Wout, Bout, Wo. Buổi 2 sẽ có đủ E, P, Wq, Wk, Wv
    assert set(errs) >= {"Wout", "Bout", "Wo"}
    for key, err in errs.items():
        assert err < 1e-5, f"{key}: sai lệch {err}"
