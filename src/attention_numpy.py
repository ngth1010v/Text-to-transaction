"""Self-attention thuần NumPy: forward, backward, kiểm tra gradient."""

import numpy as np
import math
import json
from pathlib import Path
import schema


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

    # đạo hàm của: ctx = A @ V
    dA = dctx @ cache["V"].T
    dV = cache["A"].T @ dctx

    # đạo hàm của: A = softmax(S) theo hàng
    # vị trí bị mask có A = 0 nên dS = 0 tự động, không cần xử lý riêng
    dS = cache["A"] * (dA - (dA * cache["A"]).sum(axis=1, keepdims=True))

    # đạo hàm của: S = Q @ K.T / sqrt(d)
    dQ = dS @ cache["K"] / np.sqrt(params["d"])
    dK = dS.T @ cache["Q"] / np.sqrt(params["d"])

    # đạo hàm của: Q, K, V = x @ Wq, x @ Wk, x @ Wv
    grads["Wq"] = cache["x"].T @ dQ
    grads["Wk"] = cache["x"].T @ dK
    grads["Wv"] = cache["x"].T @ dV
    # x được dùng ở 3 nhánh này và nhánh tắt residual nên cộng dồn vào dx
    dx += dQ @ params["Wq"].T + dK @ params["Wk"].T + dV @ params["Wv"].T

    # đạo hàm của: x = E[word_ids] + P
    grads["P"] = dx
    grads["E"] = np.zeros_like(params["E"])
    # np.add.at vì một từ có thể lặp lại trong câu, cần cộng dồn thay vì ghi đè
    np.add.at(grads["E"], cache["word_ids"], dx)

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



def train_mini(rows, word2id, epochs=300, lr=0.1, d=32, seed=0):

    vocab_size = len(word2id) + 1 # +1 cho pad
    params = init_params(vocab_size, d=d, seed=seed, std=0.1) # std==0.5 làm số bị nổ lên inf

    loss_history = []
    for _ in range (epochs):

        total_loss = 0
        for line in rows:

            word_ids = [word2id[w] for w in line["tokens"]]
            word_ids = pad_words(word_ids)            
            logits, cache = forward(params, word_ids)

            tag_ids = [schema.TAG2ID[w] for w in line["tags"]]
            tag_ids = pad_tags(tag_ids)
            loss, probs = loss_fn(logits, tag_ids)

            grads = backward(params, cache, probs, tag_ids)

            for name in grads:
                params[name] -= lr * grads[name]

            total_loss += loss

        avg_loss = total_loss / len(rows)
        loss_history.append(avg_loss)
        if (len(loss_history) % 20 == 0):
            print(avg_loss)

        if avg_loss < 0.1:
            break
        
    return params, loss_history


def accuracy(params, rows, word2id):
    dung = tong = 0
    for r in rows:
        word_ids = pad_words([word2id[w] for w in r["tokens"]])
        tag_ids = pad_tags([schema.TAG2ID[t] for t in r["tags"]])
        logits, _ = forward(params, word_ids)
        valid = tag_ids != -100
        dung += (logits.argmax(-1)[valid] == tag_ids[valid]).sum()
        tong += valid.sum()
    return dung / tong
            


if __name__ == "__main__":

    train_path = Path(__file__).resolve().parent.parent / "data" / "synth" / "train.jsonl"

    limit = 20
    with open(train_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) >= limit:
            break

    word_set = set()
    for line in rows:
        for word in line["tokens"]:
            word_set.add(word)
    word_set = sorted(word_set)
    word2id = {}
    for id, word in enumerate(word_set, start=1): 
        word2id[word] = id


    params, history = train_mini(rows, word2id)
    print(accuracy(params, rows, word2id))






    # params = init_params(100)

    # word_ids = np.arange(1,51, dtype=int)
    # word_ids = pad_words(word_ids)
    # logits, cache = forward(params, word_ids)

    # tag_ids = np.random.randint(0, 11, 50, dtype=int)
    # tag_ids = pad_tags(tag_ids)
    # loss, probs = loss_fn(logits, tag_ids)


    # backward(params, cache, probs, tag_ids)

