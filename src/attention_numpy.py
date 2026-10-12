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





def init_params(vocab_size, d=32, n_tags=11, max_len=100, seed=0):
    rng = np.random.default_rng(seed)

    return {
        "d": d,

        "E" : rng.normal(0, 0.02, (vocab_size,d)), # Embedding
        "P" : rng.normal(0, 0.02, (max_len,d)),    # Position

        "Wq" : rng.normal(0, 0.02, (d,d)),
        "Wk" : rng.normal(0, 0.02, (d,d)),
        "Wv" : rng.normal(0, 0.02, (d,d)),
        "Wo" : rng.normal(0, 0.02, (d,d)),
        
        "Wout" : rng.normal(0, 0.02, (d,n_tags)),
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

    print(logits[60:])

    tag_ids = np.random.randint(0, 11, 50, dtype=int)
    tag_ids = pad_tags(tag_ids)
    loss, probs = loss_fn(logits, tag_ids)

