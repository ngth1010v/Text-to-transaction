"""Baseline rule-based làm mốc so sánh cho model.

Không học gì cả: chỉ tra từ điển (lấy từ data/raw) và vài luật đơn giản.
Model ở M5 phải vượt được điểm F1 của file này thì mới đáng dùng.

Gán nhãn qua 4 lượt, lượt sau chỉ đụng vào từ mà lượt trước chưa gán (None):

1. Tra từ điển DATE / TYPE / ACCOUNT, ưu tiên cụm dài nhất (longest match).
2. Luật MONEY: từ có chữ số / số đếm bằng chữ, nuốt thêm đơn vị theo sau (`5k đ`, `12 triệu 500`).
3. Luật ACCOUNT: từ chưa gán đứng ngay sau từ gợi ý (`qua`, `từ`, `via`...) -> ACCOUNT.
4. Từ nối / từ đệm ở ranh giới cụm -> O. Mọi từ còn lại, liền nhau -> một cụm ITEM.

Chạy: uv run python src/baseline.py
Lưu ý: từ điển lấy từ toàn bộ data/raw (không chia split) nên có thể trùng chuỗi với
data/real_test. Với baseline thì chấp nhận được: nếu có lệch thì lệch có lợi cho baseline.
"""

import json
from pathlib import Path

import gen_data
import metrics
from normalize import normalize
from schema import LANGS

REAL_TEST_DIR = Path(__file__).resolve().parent.parent / "data" / "real_test"

# ====================================================================================
# Luật MONEY: dùng chung mọi ngôn ngữ (không rẽ nhánh theo lang)
# ====================================================================================
# Đơn vị tiền / bội số đứng sau con số: "5 dollars", "3 triệu", "20 bucks", "35k đ"
UNITS = {
    "k", "đ", "đồng", "dong", "vnd", "vnđ", "nghìn", "ngàn", "nghin", "ngan", "triệu", "trieu",
    "tr", "củ", "rưỡi", "chục", "trăm", "mươi",
    "dollar", "dollars", "bucks", "usd", "cents", "euros", "pounds", "yen", "grand",
}
# Số đếm viết bằng chữ: chỉ thành MONEY khi đi cùng đơn vị ("năm mươi nghìn", "twenty dollars")
NUMWORDS = {
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "twenty", "thirty", "forty", "fifty", "hundred", "thousand",
}

# Từ gợi ý đứng ngay trước ACCOUNT (xem mục 2 trong CLAUDE.md)
ACCOUNT_CUES = {"qua", "từ", "vào", "bằng", "sang", "via", "from", "with", "using", "by"}


def has_digit(word):
    return any(c.isdigit() for c in word)


# ====================================================================================
# Từ điển: đọc từ data/raw/<lang>/, gộp mọi ngôn ngữ (câu có thể trộn vi-en)
# ====================================================================================
def strip_accents(words):
    """Bản bỏ dấu của cụm, để khớp được cả câu gõ không dấu ("hom qua")."""
    return gen_data.noise_strip_accents(words, None, "vi", None)[0]


def build_lexicon():
    """Trả (lexicon, stopwords).

    lexicon: dict tuple-từ -> loại, ví dụ ("hôm", "qua") -> "DATE".
    stopwords: tập từ nối / từ đệm, sẽ gán O nếu đứng ở ranh giới cụm.
    """
    lexicon = {}
    stopwords = set(ACCOUNT_CUES)

    def add(phrase, kind):
        words = tuple(normalize(phrase))
        # setdefault: cụm đã có loại thì giữ loại cũ -> thứ tự add quyết định ưu tiên khi trùng
        for w in (words, tuple(strip_accents(list(words)))):
            if w:
                lexicon.setdefault(w, kind)

    for lang in LANGS:
        raw = gen_data.load_lang(lang)

        # Các từ xuất hiện trong ITEM: dùng để loại tên account 1 từ quá chung chung
        # ("tiền", "lương" vừa là account vừa là nội dung -> tra từ điển sẽ gán sai ITEM thành ACCOUNT)
        item_words = {w for phrase in raw["item"] for w in normalize(phrase)}

        # Thứ tự ưu tiên khi một cụm có ở nhiều loại: DATE > TYPE > ACCOUNT
        for phrase in raw["date"]:
            add(phrase, "DATE")
        for phrase in raw["type"]:
            add(phrase, "TYPE")
        for name in raw["account"]:
            # Ghép prefix/suffix giống gen_data: "ví momo", "momo wallet", "thẻ tín dụng"...
            for p in raw["account_prefix"]:
                add(f"{p} {name}", "ACCOUNT")
            for s in raw["account_suffix"]:
                add(f"{name} {s}", "ACCOUNT")
            words = normalize(name)
            if len(words) > 1 or words[0] not in item_words:
                add(name, "ACCOUNT")

        # Chữ cố định trong template (không phải {SLOT}) là từ nối: "cho", "on", "hết", "i"...
        for t in raw["templates"]:
            stopwords |= {w for w in normalize(t) if not w.startswith("{")}
        stopwords |= set(gen_data.FILLERS[lang])

    return lexicon, stopwords


LEXICON, STOPWORDS = build_lexicon()
MAX_LEN = max(len(k) for k in LEXICON)  # cụm dài nhất trong từ điển, giới hạn vòng tìm


# ====================================================================================
# Gán nhãn
# ====================================================================================
def mark(tags, start, end, kind):
    """Gán cụm [start, end) là `kind` theo BIO: từ đầu B-, các từ sau I-."""
    tags[start] = f"B-{kind}"
    for i in range(start + 1, end):
        tags[i] = f"I-{kind}"


def tag_lexicon(tokens, tags):
    """Lượt 1: quét trái sang phải, tại mỗi vị trí thử cụm dài nhất trước rồi ngắn dần."""
    i = 0
    while i < len(tokens):
        for n in range(min(MAX_LEN, len(tokens) - i), 0, -1):
            kind = LEXICON.get(tuple(tokens[i:i + n]))
            if kind:
                mark(tags, i, i + n, kind)
                i += n
                break
        else:  # không cụm nào khớp ở vị trí i -> sang từ kế
            i += 1


def tag_money(tokens, tags):
    """Lượt 2: tìm cụm MONEY trong các từ chưa gán.

    Một cụm ứng viên = dãy từ liền nhau, mỗi từ là số (có chữ số), số bằng chữ, hoặc đơn vị.
    Chỉ nhận nếu cụm có chữ số hoặc có đơn vị: "two" đứng một mình (trong "two days ago")
    không phải tiền, nhưng "twenty dollars" thì phải.
    Từ có "/" và chữ số ("15/9") là ngày, không phải tiền.
    """
    def is_money_word(w):
        return has_digit(w) or w in UNITS or w in NUMWORDS

    i = 0
    while i < len(tokens):
        w = tokens[i]
        # đơn vị không được đứng đầu cụm ("k" một mình không phải tiền)
        if tags[i] is not None or w in UNITS or not is_money_word(w):
            i += 1
            continue
        if "/" in w and has_digit(w):
            mark(tags, i, i + 1, "DATE")
            i += 1
            continue
        # kéo dài cụm sang phải khi từ kế vẫn chưa gán và vẫn là từ chỉ tiền
        j = i + 1
        while j < len(tokens) and tags[j] is None and is_money_word(tokens[j]) and "/" not in tokens[j]:
            j += 1
        span = tokens[i:j]
        if any(has_digit(x) or x in UNITS for x in span):
            mark(tags, i, j, "MONEY")
        i = j


def tag_account_cue(tokens, tags):
    """Lượt 3: dãy từ chưa gán ngay sau từ gợi ý ("qua", "via"...) là ACCOUNT.

    Đây là luật giúp nhận account tự đặt không có trong từ điển ("bằng heo đất").
    Từ gợi ý phải chưa gán: "qua" trong "hôm qua" đã thành DATE ở lượt 1 nên không bị tính.
    """
    for i, w in enumerate(tokens[:-1]):
        if tags[i] is None and w in ACCOUNT_CUES and tags[i + 1] is None:
            j = i + 1
            while j < len(tokens) and tags[j] is None and tokens[j] not in STOPWORDS:
                j += 1
            if j > i + 1:
                mark(tags, i + 1, j, "ACCOUNT")


def tag_rest(tokens, tags):
    """Lượt 4: từ nối -> O, phần còn lại -> ITEM.

    Từ nối chỉ là O khi ở ranh giới cụm. Nếu kẹp giữa hai từ ITEM thì nó thuộc ITEM:
    "dinner with friends" là một ITEM, nhưng "spent 60 on shoes" thì "on" là O.
    """
    n = len(tokens)

    def is_item(k):  # vị trí k là từ thường chưa gán (ứng viên ITEM)
        return 0 <= k < n and tags[k] is None and tokens[k] not in STOPWORDS

    stop = [tags[i] is None and tokens[i] in STOPWORDS and not (is_item(i - 1) and is_item(i + 1))
            for i in range(n)]
    for i in range(n):
        if stop[i]:
            tags[i] = "O"

    # gom từng dãy None liền nhau thành một cụm ITEM
    i = 0
    while i < n:
        if tags[i] is not None:
            i += 1
            continue
        j = i
        while j < n and tags[j] is None:
            j += 1
        mark(tags, i, j, "ITEM")
        i = j


def predict(tokens):
    """tokens (đã chuẩn hóa) -> list nhãn BIO cùng độ dài."""
    tags = [None] * len(tokens)  # None = chưa gán
    tag_lexicon(tokens, tags)
    tag_money(tokens, tags)
    tag_account_cue(tokens, tags)
    tag_rest(tokens, tags)
    return tags


# ====================================================================================
# Chấm điểm trên tập test thật
# ====================================================================================
def load_real_test():
    rows = []
    for lang in LANGS:
        with open(REAL_TEST_DIR / f"{lang}.jsonl", encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def main():
    rows = load_real_test()
    golds = [r["tags"] for r in rows]
    preds = [predict(r["tokens"]) for r in rows]
    langs = [r["lang"] for r in rows]

    metrics.print_report(metrics.score(golds, preds), f"ALL ({len(rows)} câu)")
    for lang, result in metrics.score_by_lang(langs, golds, preds).items():
        print()
        metrics.print_report(result, f"{lang} ({langs.count(lang)} câu)")


if __name__ == "__main__":
    main()
