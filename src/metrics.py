"""F1 theo cụm (entity-level), tự viết. Không dùng seqeval.

Một cụm đúng khi khớp cả loại lẫn vị trí đầu-cuối.
Nhãn có thể là chuỗi ("B-MONEY") hoặc số (id trong TAGS). Vị trí gold = -100 bị bỏ qua.
"""

from schema import TAGS

IGNORE = -100
# Các loại cụm lấy từ TAGS, không viết cứng: ["MONEY", "DATE", ...]
TYPES = [t[2:] for t in TAGS if t.startswith("B-")]


def extract_spans(tags):
    """Chuỗi nhãn BIO (list chuỗi) -> tập (loại, start, end), end không tính.

    I-X mồ côi (không đứng sau B-X/I-X cùng loại) được coi là mở cụm mới (kiểu conlleval).
    """
    spans = set()
    kind, start = None, 0  # cụm đang mở
    for i, tag in enumerate(tags):
        if tag == "O":
            prefix, t = "O", None
        else:
            prefix, t = tag[0], tag[2:]
        # I-X nối tiếp cụm đang mở cùng loại thì không làm gì
        if prefix == "I" and t == kind:
            continue
        # còn lại: đóng cụm đang mở, rồi mở cụm mới nếu là B-X hoặc I-X mồ côi
        if kind is not None:
            spans.add((kind, start, i))
        kind, start = (t, i) if prefix in ("B", "I") else (None, i)
    if kind is not None:
        spans.add((kind, start, len(tags)))
    return spans


def prf(tp, fp, fn):
    """Precision, recall, F1. Mẫu số bằng 0 thì trả 0."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def _to_str(tag):
    return TAGS[tag] if isinstance(tag, int) else tag


def _clean(gold, pred):
    """Bỏ vị trí gold = -100 ở cả gold lẫn pred, đổi id -> chuỗi nhãn."""
    keep = [i for i, g in enumerate(gold) if g != IGNORE]
    return [_to_str(gold[i]) for i in keep], [_to_str(pred[i]) for i in keep]


def score(golds, preds):
    """golds, preds: list các câu (mỗi câu là list nhãn). Trả P/R/F1 từng loại + micro."""
    count = {t: [0, 0, 0] for t in TYPES}  # tp, fp, fn
    for gold, pred in zip(golds, preds):
        g, p = _clean(gold, pred)
        gs, ps = extract_spans(g), extract_spans(p)
        for kind, _, _ in gs & ps:
            count[kind][0] += 1
        for kind, _, _ in ps - gs:
            count[kind][1] += 1
        for kind, _, _ in gs - ps:
            count[kind][2] += 1

    def row(tp, fp, fn):
        p, r, f1 = prf(tp, fp, fn)
        return {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    per_type = {t: row(*c) for t, c in count.items()}
    # micro: cộng tp/fp/fn của mọi loại rồi mới tính
    micro = row(*(sum(c[i] for c in count.values()) for i in range(3)))
    return {"per_type": per_type, "micro": micro}


def score_by_lang(langs, golds, preds):
    """langs: list `lang` song song với golds/preds. Danh sách ngôn ngữ lấy từ dữ liệu."""
    result = {}
    for lang in sorted(set(langs)):
        idx = [i for i, l in enumerate(langs) if l == lang]
        result[lang] = score([golds[i] for i in idx], [preds[i] for i in idx])
    return result


def print_report(result, title=""):
    if title:
        print(title)
    print(f"{'type':<10}{'P':>8}{'R':>8}{'F1':>8}{'tp':>6}{'fp':>6}{'fn':>6}")
    rows = list(result["per_type"].items()) + [("MICRO", result["micro"])]
    for name, s in rows:
        print(f"{name:<10}{s['p']:>8.3f}{s['r']:>8.3f}{s['f1']:>8.3f}{s['tp']:>6}{s['fp']:>6}{s['fn']:>6}")
