import gzip
import argparse
import numpy as np
import urllib.request




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

    return words, rows



""" Pca thủ công (x.shape = (N, 300))"""
def pca(x, dim):

    if (dim == 300):
        return x

    N = x.shape[1]
    x = x - x.mean(axis=0)
    cov = x.T @ x / (N - 1)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1][:dim]
    w = vecs[:, idx]
    return x @ w # shape = (N, dim)





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, default=100_000, help="Số từ sẽ lấy")
    parser.add_argument("--dim", type=int, default=100, help="Số chiều của embedding")
    parser.add_argument("--lang", type=str, default="en", help="Ngôn ngữ mục tiêu")
    args = parser.parse_args()


    words, rows = read_head(args.lang, args.lines)
    x = np.array(rows, dtype=np.float32)
    print(x.shape)
    x = pca(x, args.dim)

    print(x.shape)


    


if __name__ == "__main__":
    main()

