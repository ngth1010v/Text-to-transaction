"""Self-attention thuần NumPy: forward, backward, kiểm tra gradient."""

import numpy as np
import math



def _softmax_by_row(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=1, keepdims=True)   # ổn định số học
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


    

def pad_words(ids, length=100):
    X = np.zeros(length, dtype=int)
    X[:len(ids)] = ids
    return X

def pad_tags(ids, length=100):
    X = np.full(length, -100, dtype=int)
    X[:len(ids)] = ids
    return X





def init_params(vocab_size, d=32, n_tags=11, max_len=100, seed=0, std=0.02):
    rng = np.random.default_rng(seed)

    return {
        "d": d,

        "E" : rng.normal(0, std, (vocab_size,d)), # Embedding
        "P" : rng.normal(0, std, (max_len,d)),    # Position

        "Wq" : rng.normal(0, std, (d,d)),
        "Wk" : rng.normal(0, std, (d,d)),
        "Wv" : rng.normal(0, std, (d,d)),
        "Wo" : rng.normal(0, std, (d,d)),
        
        "Wout" : rng.normal(0, std, (d,n_tags)),
        "Bout" : np.zeros(n_tags)        
    }



def forward(params, word_ids):

    # đánh dấu các pad
    mask = word_ids != 0
    # Áp dụng embedding (word + position)                         
    x = params["E"][word_ids] + params["P"]

    # Đẩy sang Queue (tôi tìm gì) - Key (tôi là ai) - Value (nếu bạn hỏi tôi -> tôi sẽ đưa cho bạn cái gì)
    Q = x @ params["Wq"]
    K = x @ params["Wk"]
    V = x @ params["Wv"]

    # Lấy mối quan hệ giữa các từ: Sij = <tôi tìm gì>j với <tôi là ai>i (chia d để làm số nhỏ lại)
    S = Q @ K.T / math.sqrt(params["d"])
    
    # Đẩy -inf vào các pad để softmax tự = 0 
    S[:, ~mask] = - np.inf

    # Đưa về xác xuất
    A = _softmax_by_row(S)

    # Lấy trung bình có trọng số
    ctx = A @ V

    # Thêm wieght vào attention, thêm x để giữ data gốc (embedding + position)
    h = x + ctx @ params["Wo"]

    # Áp wieght và bias để ra bảng <word/tag> 
    logits = h @ params["Wout"] + params["Bout"]

    return logits, {
        "word_ids": word_ids,

        "mask": mask,
        "x": x,

        "Q": Q,
        "K": K,
        "V": V,
        "S": S,

        "A": A,
        "h": h,
        "ctx": ctx
    }


def backward(params, cache, probs, tag_ids):
    grads = {}

    # đạo hàm của: loss = -mean(log probs[nhãn đúng])  (softmax + cross-entropy)
    valid = tag_ids != -100
    n = valid.sum()
    rows = np.where(valid)[0]
    dlogits = probs.copy()
    dlogits[rows, tag_ids[rows]] -= 1
    dlogits[~valid] = 0
    dlogits /= n

    # đạo hàm của: logits = h @ Wout + Bout
    grads["Wout"] = cache["h"].T @ dlogits
    grads["Bout"] = dlogits.sum(axis=0)
    dh = dlogits @ params["Wout"].T

    # đạo hàm của: h = x + ctx @ Wo
    dx = dh.copy()                        # nhánh tắt
    grads["Wo"] = cache["ctx"].T @ dh
    dctx = dh @ params["Wo"].T

    return grads


def grad_check(params, word_ids, tag_ids, eps=1e-5, n_samples=20, seed=0):
    rng = np.random.default_rng(seed)
    mask = word_ids != 0

    def loss_now():
        logits, _ = forward(params, word_ids)
        return loss_fn(logits, tag_ids)[0]

    # gradient bằng backward (a)
    logits, cache = forward(params, word_ids)
    loss, probs = loss_fn(logits, tag_ids)
    grads = backward(params, cache, probs, tag_ids)

    worst = {}
    for key, g in grads.items():
        W = params[key]

        # chỉ chọn ô có gradient khác 0
        if key == "E":
            rows = np.unique(word_ids[mask])
        elif key == "P":
            rows = np.where(mask)[0]
        else:
            rows = np.arange(W.shape[0])

        worst[key] = 0.0
        for _ in range(n_samples):
            idx = (rng.choice(rows),) + tuple(rng.integers(0, s) for s in W.shape[1:])

            old = W[idx]
            W[idx] = old + eps
            loss_plus = loss_now()
            W[idx] = old - eps
            loss_minus = loss_now()
            W[idx] = old                      # trả lại giá trị cũ

            num = (loss_plus - loss_minus) / (2 * eps)
            ana = g[idx]
            rel = abs(ana - num) / max(abs(ana) + abs(num), 1e-12)
            worst[key] = max(worst[key], rel)

    return worst



def loss_fn(logits, tag_ids):
    probs = _softmax_by_row(logits)
    valid = tag_ids != -100
    loss = - np.mean(np.log( probs [valid, tag_ids[valid]]))

    return loss, probs



if __name__ == "__main__":
    params = init_params(100)

    word_ids = np.arange(1,51, dtype=int)
    word_ids = pad_words(word_ids)
    logits, cache = forward(params, word_ids)

    tag_ids = np.random.randint(0, 11, 50, dtype=int)
    tag_ids = pad_tags(tag_ids)
    loss, probs = loss_fn(logits, tag_ids)


    backward(params, cache, probs, tag_ids)

