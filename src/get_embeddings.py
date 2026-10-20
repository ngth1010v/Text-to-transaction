import gzip
import argparse
import numpy as np
import urllib.request
import os
import schema


""" Lấy embedding từ web với dim = 300"""
def read_head(lang, lines):

    url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.vec.gz"
    resp = urllib.request.urlopen(url, timeout=60)
    f = gzip.open(resp, "rt", encoding="utf-8")

    next(f)


    words = []
    rows = []
    for i, line in enumerate(f):
        if i >= lines: break

        parts = line.rstrip().split(" ")
        if len(parts) == 301:
            words.append(parts[0])
            rows.append(parts[1:])

    # resp.length = số byte còn lại chưa đọc; tổng - còn lại = đã tải
    total = int(resp.headers["Content-Length"])
    print(f"Đã tải ~{(total - resp.length) / 1e6:.1f} MB")
    resp.close()

    return words, rows



""" Pca thủ công (x.shape = (N, 300))"""
def pca(x, dim):

    if (dim == 300):
        return x

    N = x.shape[0]                    # số mẫu (từ), không phải số chiều
    x = x - x.mean(axis=0)            # trừ trung bình để căn giữa dữ liệu
    cov = x.T @ x / (N - 1)           # ma trận hiệp phương sai (300, 300)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1][:dim]
    w = vecs[:, idx]
    return x @ w # shape = (N, dim)



""" Ghi ra định dạng word2vec text: dòng đầu "N dim", mỗi dòng "từ v1 ... vdim" """
def save_txt(path, words, x):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(words)} {x.shape[1]}\n")
        for w, v in zip(words, x):
            f.write(w + " " + " ".join(f"{a:.5f}" for a in v) + "\n")



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, default=100_000, help="Số từ sẽ lấy")
    parser.add_argument("--dim", type=int, default=100, help="Số chiều của embedding")    
    args = parser.parse_args()


    for lang in schema.LANGS:
        print(f"Loading {lang}...")

        words, rows = read_head(lang, args.lines)
        x = np.array(rows, dtype=np.float32)
        x = pca(x, args.dim)

        os.makedirs("data/embeddings", exist_ok=True)
        path = f"data/embeddings/{lang}.txt"
        save_txt(path, words, x)
        print(f"Đã ghi {path}: {os.path.getsize(path) / 1e6:.1f} MB")


    print("Load embedding done.")

    


if __name__ == "__main__":
    main()

