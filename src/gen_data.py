"""Sinh dữ liệu từ template + nhiễu, chia train/val/test theo mảnh gốc."""
"""FLOW: load_lang -> split_pieces -> mix_lang -> gen_account -> điền template -> nhiễu -> JSONL"""

import argparse
import json
import math
import random
import unicodedata
from pathlib import Path

import normalize
import schema




RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

""" Hàm dùng để load toàn bộ data trong 1 lang_code ra thành 1 dict """
def load_lang(lang_code):

    if not isinstance(lang_code, str):
        raise ValueError("lang_code must be string")


    lang_dir = RAW_DIR / lang_code
    if not lang_dir.exists():
        raise FileNotFoundError(f"lang_dir not found: {lang_dir}")


    paths = {}
    for raw_type in schema.RAW_TOKENS:
        path = lang_dir / f"{raw_type}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        paths[raw_type] = path


    lang = {}
    for key, path in paths.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        # bỏ dòng rỗng và dòng trùng (giữ thứ tự), để cùng một mảnh không bị chia vào 2 split
        lang[key] = list(dict.fromkeys(line.strip() for line in lines if line.strip()))


    return lang



""" Hàm dùng tách 1 mảng thành dict: train / val / test """
SPLIT_RATIO = [0.8, 0.1, 0.1] # train / val / test
def split_pieces(pieces, rng):
    n = len(pieces)
    if n < 3:
        raise ValueError(f"Need atleast 3 pieces for split, currently: {n}")

    shuffled = rng.sample(pieces, k=n)

    n_val  = max(1, round(n * SPLIT_RATIO[1]))
    n_test = max(1, round(n * SPLIT_RATIO[2]))
    n_train = n - n_val - n_test   # phần còn lại, luôn >= 1 khi n >= 3

    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]

    return {"train": train, "val": val, "test": test}



""" Load mọi ngôn ngữ và chia train/val/test.
Một chuỗi chỉ thuộc đúng 1 split, kể cả khi nó xuất hiện ở nhiều ngôn ngữ hoặc nhiều loại
(vd `visa` có ở vi lẫn en, `lương` có thể vừa là account vừa là item). """
SPLITS = ["train", "val", "test"]
PASS_FULL = ("account_prefix", "account_suffix", "templates")  # ít, không phải nội dung giao dịch: dùng đủ ở cả 3 tập
def split_all(rng):
    langs = {s: {lang_code: {} for lang_code in schema.LANGS} for s in SPLITS}
    assigned = {}  # chuỗi -> split

    for lang_code in schema.LANGS:
        for key, data in load_lang(lang_code).items():

            if key in PASS_FULL:
                for s in SPLITS:
                    langs[s][lang_code][key] = data
                continue

            free = [p for p in data if p not in assigned]
            if len(free) >= 3:
                splited = split_pieces(free, rng)
            else:
                splited = {"train": free, "val": [], "test": []}
            for s in SPLITS:
                for p in splited[s]:
                    assigned[p] = s

            for s in SPLITS:
                pool = [p for p in data if assigned[p] == s]
                if not pool:
                    raise ValueError(f"{lang_code}/{key}: split '{s}' rỗng, cần thêm mảnh gốc")
                langs[s][lang_code][key] = pool

    return langs







""" Hàm để trộn 1 list main và nhiều list sub thành 1 mảng lớn """
def shuffle_merge_list(main, subs, n, rng, main_ratio=0.9):
    """Trộn 1 list main và nhiều list sub thành 1 mảng gồm n phần tử.

    - main chiếm main_ratio, phần còn lại chia đều cho các sub.
    - Mỗi nguồn được lấy xoay vòng trên bản đã xáo, nên phủ đều phần tử
      và chỉ lặp lại khi nguồn có ít phần tử hơn số lượng cần lấy.
    """
    if not isinstance(main, list):
        raise ValueError("main must be a list")
    if not isinstance(subs, list):
        raise ValueError("subs must be a list")
    for sub in subs:
        if not isinstance(sub, list):
            raise ValueError("sub must be a list")
    if not 0 <= main_ratio <= 1:
        raise ValueError("main_ratio must be in [0, 1]")
    if n < 0:
        raise ValueError("n must be >= 0")

    # bỏ các sub rỗng; nếu không còn sub nào thì main nhận toàn bộ
    subs = [s for s in subs if s]
    if not subs:
        main_ratio = 1.0
    if main_ratio > 0 and not main:
        raise ValueError("main is empty but main_ratio > 0")

    # số lượng từng nguồn: tính bằng phép trừ để tổng luôn đúng bằng n
    n_main = round(n * main_ratio)
    n_rest = n - n_main
    n_each, extra = divmod(n_rest, len(subs)) if subs else (0, 0)
    counts = [n_main] + [n_each + (1 if i < extra else 0) for i in range(len(subs))]

    def _take(source, count, rng):
        """Lấy count phần tử: xáo source, lặp vòng (mỗi vòng xáo lại) nếu thiếu."""
        if count == 0:
            return []
        result = []
        while len(result) < count:
            pool = source[:]
            rng.shuffle(pool)
            result.extend(pool[:count - len(result)])
        return result

    out = []
    for source, count in zip([main] + subs, counts):
        out.extend(_take(source, count, rng))

    rng.shuffle(out)
    return out









""" Hàm để gom: [prefix] + (85% account hoặc 15% item) + [suffix] -> account"""
ACCOUNT_RATIO = 0.85
EMPTY_RATIO = 0.3   # xác suất prefix / suffix bị bỏ trống (tính riêng từng bên)
ACCOUNT_REPEAT = 3  # số lượt đi qua mỗi mảnh account / item
def gen_account(account_prefix, account_suffix, account, item, rng, foreign=frozenset()):
    """Trả (danh sách cụm account, tập cụm có chứa tên mượn từ ngôn ngữ khác theo `foreign`)."""

    n = ACCOUNT_REPEAT * (len(account) + len(item))

    out = []
    foreign_out = set()
    i_account = 0
    i_account_item = 0
    for _ in range(n):

        prefix = "" if rng.random() < EMPTY_RATIO else rng.choice(account_prefix)
        suffix = "" if rng.random() < EMPTY_RATIO else rng.choice(account_suffix)

        if rng.random() < ACCOUNT_RATIO:
            name = account[i_account]
            i_account = (i_account + 1) % len(account)

        else:
            name = item[i_account_item]
            i_account_item = (i_account_item + 1) % len(item)
            # item không có prefix/suffix thì chỉ là item thường, không phải account
            if not prefix and not suffix:
                prefix = rng.choice(account_prefix)

        # tên đã tự mang prefix/suffix (`ví chính`, `main wallet`) thì không thêm lần nữa
        if any(name.startswith(p + " ") for p in account_prefix):
            prefix = ""
        if any(name.endswith(" " + x) for x in account_suffix):
            suffix = ""

        text = " ".join(part for part in (prefix, name, suffix) if part)
        out.append(text)
        if name in foreign:
            foreign_out.add(text)


    return out, foreign_out



""" Trộn 1 số ngôn ngữ này vào ngôn ngữ khác (chỉ trong cùng split, nên không rò mảnh gốc).
Mỗi pool lấy MIX_MAIN_RATIO từ ngôn ngữ của nó; mỗi câu có nhiều slot nên tỉ lệ câu trộn
cao hơn tỉ lệ mảnh trộn. Chỉnh để ~10-20% câu là câu trộn ngôn ngữ. """
MIX_MAIN_RATIO = 0.94
def mix_lang(langs, rng):

    for type in ["train", "val", "test"]:

        for raw_type in schema.RAW_TOKENS:

            # Không trộn template; prefix/suffix giữ đúng ngữ pháp từng ngôn ngữ
            if raw_type in ["templates", "account_prefix", "account_suffix"]:
                continue
            
            for lang_code in schema.LANGS: 
                main = langs[type][lang_code][raw_type]
                subs = []
                for sub_lang_code in schema.LANGS: 
                    if sub_lang_code == lang_code:
                        continue
                    subs.append(langs[type][sub_lang_code][raw_type])

                main_ratio = MIX_MAIN_RATIO
                n = math.ceil(len(main) / main_ratio)
                langs[type][lang_code][raw_type] = shuffle_merge_list(main, subs, n, rng, main_ratio)




# ====================================================================================
# Điền template -> (tokens, tags)
# ====================================================================================
SLOTS = {"{MONEY}": "MONEY", "{DATE}": "DATE", "{ACCOUNT}": "ACCOUNT", "{ITEM}": "ITEM", "{TYPE}": "TYPE"}
SLOT_POOL = {"MONEY": "money", "DATE": "date", "ACCOUNT": "account", "ITEM": "item", "TYPE": "type"}


def fill_template(template, pools, rng, foreign=frozenset()):
    """Mỗi slot lấy ngẫu nhiên 1 mảnh (đã tách từ): từ đầu B-X, các từ sau I-X.
    Từ thường trong template là O. Không normalize cả template vì sẽ hạ chữ hoa tên slot.
    Trả thêm `mixed`: True nếu có mảnh thuộc `foreign` (tập tuple từ mượn ngôn ngữ khác)."""
    tokens, tags, mixed = [], [], False
    for word in template.split():
        if word in SLOTS:
            slot = SLOTS[word]
            piece = rng.choice(pools[SLOT_POOL[slot]])
            mixed = mixed or tuple(piece) in foreign
            tokens.extend(piece)
            tags.extend([f"B-{slot}"] + [f"I-{slot}"] * (len(piece) - 1))
        else:
            words = normalize.normalize(word)
            tokens.extend(words)
            tags.extend(["O"] * len(words))
    return tokens, tags, mixed


# ====================================================================================
# Nhiễu: mỗi hàm nhận (tokens, tags, lang, rng) và trả (tokens, tags)
# Chỉ đổi từ 1-1 hoặc chèn O ở ranh giới cụm nên độ dài khớp và BIO luôn hợp lệ.
# ====================================================================================
def noise_strip_accents(tokens, tags, lang, rng):
    """Bỏ dấu tiếng Việt (đ -> d)."""
    def strip(w):
        w = w.replace("đ", "d").replace("Đ", "D")
        w = unicodedata.normalize("NFD", w)
        w = "".join(c for c in w if unicodedata.category(c) != "Mn")
        return unicodedata.normalize("NFC", w)
    return [strip(w) for w in tokens], tags


ABBR = {
    "vi": {"không": "ko", "được": "đc", "với": "vs", "triệu": "tr", "nghìn": "k", "ngàn": "k"},
    "en": {"please": "pls", "with": "w/", "and": "n", "thousand": "k", "dollars": "usd",
           "tomorrow": "tmrw", "yesterday": "yday"},
}


def noise_abbreviate(tokens, tags, lang, rng):
    """Viết tắt từ phổ biến (1 từ -> 1 từ, tag giữ nguyên)."""
    abbr = ABBR[lang]
    return [abbr.get(w, w) for w in tokens], tags


def noise_typo(tokens, tags, lang, rng):
    """Gõ sai 1 từ (hoán đổi / bỏ / lặp ký tự). Bỏ qua từ có chữ số và từ ngắn."""
    ids = [i for i, w in enumerate(tokens) if len(w) >= 4 and not any(c.isdigit() for c in w)]
    if not ids:
        return tokens, tags
    i = rng.choice(ids)
    w, k = tokens[i], rng.randrange(len(tokens[i]) - 1)
    op = rng.choice(["swap", "drop", "dup"])
    if op == "swap":
        w = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    elif op == "drop":
        w = w[:k] + w[k + 1:]
    else:
        w = w[:k + 1] + w[k] + w[k + 1:]
    return tokens[:i] + [w] + tokens[i + 1:], tags


FILLERS = {
    "vi": ["ừm", "à", "ơi", "nhé", "nha", "đó", "luôn"],
    "en": ["um", "uh", "so", "like", "ok", "please", "btw"],
}


def noise_filler(tokens, tags, lang, rng):
    """Chèn từ đệm (O) vào ranh giới cụm, không chèn giữa cụm."""
    ids = [i for i in range(len(tokens) + 1) if i == len(tokens) or not tags[i].startswith("I-")]
    i = rng.choice(ids)
    return tokens[:i] + [rng.choice(FILLERS[lang])] + tokens[i:], tags[:i] + ["O"] + tags[i:]


# (hàm, xác suất áp dụng cho mỗi câu, ngôn ngữ áp dụng)
NOISES = [
    (noise_strip_accents, 0.3, ["vi"]),
    (noise_abbreviate,    0.3, ["vi", "en"]),
    (noise_typo,          0.2, ["vi", "en"]),
    (noise_filler,        0.2, ["vi", "en"]),
]


def add_noise(tokens, tags, lang, rng):
    for fn, prob, applies_to in NOISES:
        if lang in applies_to and rng.random() < prob:
            tokens, tags = fn(tokens, tags, lang, rng)
    return tokens, tags


# ====================================================================================
# Sinh câu cho 1 split
# ====================================================================================
def gen_split(split_langs, foreign, n_per_lang, rng, start_id):
    """Trả (danh sách dòng JSONL, id kế tiếp, số câu trộn ngôn ngữ).
    Mỗi ngôn ngữ n_per_lang câu, không trùng nhau."""
    rows, seen, next_id, n_mixed = [], set(), start_id, 0
    for lang_code in schema.LANGS:
        raw = split_langs[lang_code]
        pools = {k: [normalize.normalize(p) for p in raw[k]] for k in SLOT_POOL.values()}
        foreign_t = {tuple(normalize.normalize(p)) for p in foreign[lang_code]}

        made, attempts = 0, 0
        while made < n_per_lang and attempts < 20 * n_per_lang:
            attempts += 1
            tokens, tags, mixed = fill_template(rng.choice(raw["templates"]), pools, rng, foreign_t)
            tokens, tags = add_noise(tokens, tags, lang_code, rng)
            key = (lang_code, tuple(tokens))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"id": f"syn-{next_id:06d}", "lang": lang_code, "source": "synthetic",
                         "tokens": tokens, "tags": tags})
            next_id += 1
            made += 1
            n_mixed += mixed
    rng.shuffle(rows)
    return rows, next_id, n_mixed


def gen_data(out_dir, n, seed):
    """n = tổng số câu (chia đều cho các ngôn ngữ, rồi chia train/val/test theo SPLIT_RATIO)."""

    rng = random.Random(seed)

    """ Load lang và chia train/val/test """
    langs = split_all(rng)

    # chuỗi thuộc ngôn ngữ của chính nó (trước khi trộn), để biết mảnh nào là mảnh mượn
    own = {s: {l: {p for k, v in langs[s][l].items() if k not in PASS_FULL for p in v}
               for l in schema.LANGS} for s in SPLITS}


    """ Trộn ngôn ngữ """
    mix_lang(langs, rng)

    foreign = {s: {l: {p for k, v in langs[s][l].items() if k not in PASS_FULL for p in v} - own[s][l]
                   for l in schema.LANGS} for s in SPLITS}


    """ Mix account ([prefix] + (account/item) + [suffix]); cả cụm là ACCOUNT """
    for type in SPLITS:
        for lang_code in schema.LANGS:
            _prefix = langs[type][lang_code]["account_prefix"]
            _suffix = langs[type][lang_code]["account_suffix"]
            _account = langs[type][lang_code]["account"]
            _item = langs[type][lang_code]["item"]
            langs[type][lang_code]["account"], foreign_acc = gen_account(
                _prefix, _suffix, _account, _item, rng, foreign[type][lang_code])
            foreign[type][lang_code] |= foreign_acc


    """ Điền template + nhiễu + ghi JSONL """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_lang = n // len(schema.LANGS)
    n_val = max(1, round(per_lang * SPLIT_RATIO[1]))
    n_test = max(1, round(per_lang * SPLIT_RATIO[2]))
    sizes = {"train": per_lang - n_val - n_test, "val": n_val, "test": n_test}

    next_id, counts, n_mixed, n_total = 1, {}, 0, 0
    for split in SPLITS:
        rows, next_id, mixed = gen_split(langs[split], foreign[split], sizes[split], rng, next_id)
        n_mixed += mixed
        n_total += len(rows)
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = len(rows)

    counts["mixed_ratio"] = n_mixed / n_total if n_total else 0.0
    return counts


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Sinh dữ liệu synthetic")
    parser.add_argument("--out", default=str(root / "data" / "synth"), help="thư mục ghi train/val/test.jsonl")
    parser.add_argument("--n", type=int, default=10000, help="tổng số câu")
    parser.add_argument("--seed", type=int, default=90)
    args = parser.parse_args()
    print(gen_data(args.out, args.n, args.seed))