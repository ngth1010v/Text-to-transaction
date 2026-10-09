"""Sinh dữ liệu từ template + nhiễu, chia train/val/test theo mảnh gốc."""
"""FLOW: load_lang -> gen_account -> split_pieces"""

import schema
from pathlib import Path
import random




""" Hàm dùng để load toàn bộ data trong 1 lang_code ra thành 1 dict """
def load_lang(lang_code):
    
    if not isinstance(lang_code, str):
        raise ValueError("lang_code must be string")


    lang_dir = Path(__file__).resolve().parent / ".." / "data" / "raw" / lang_code
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
        lang[key] = [line.strip() for line in lines if line.strip()]


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







""" Hàm để trộn 1 list main và nhiều list sub thành 1 mảng lớn """
def shuffle_merge_list(main, subs, n, rng, main_ratio=0.8):
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









""" Hàm để gom: account_prefix + (60% account hoặc 40% item) -> account"""
ACCOUNT_RATIO = 0.6
def gen_account(account_prefix, account, item, rng):

    n = len(account_prefix) * (len(account) + len(item))

    out = []
    i_account_prefix = 0
    i_account = 0
    i_account_item = 0
    for _ in range(n):

        prefix = account_prefix[i_account_prefix]
        i_account_prefix = (i_account_prefix + 1) % len(account_prefix)

        if rng.random() < ACCOUNT_RATIO:
            ac = account[i_account]
            i_account = (i_account + 1) % len(account)
            out.append(prefix + ' ' + ac)

        else:
            at = item[i_account_item]
            i_account_item = (i_account_item + 1) % len(item)
            out.append(prefix + ' ' + at)


    return out



""" Trộn 1 số ngôn ngữ này vào ngôn ngữ khác """
def mix_lang(langs, rng):

    for type in ["train", "val", "test"]:
        lang = langs[type]

        for raw_type in schema.RAW_TOKENS:

            # Không trộn template
            if raw_type in ["template"]:
                continue
            
            for lang_code in schema.LANGS: 
                main = lang[type][lang_code][raw_type]
                subs = []
                for sub_lang_code in schema.LANGS: 
                    if sub_lang_code == lang_code:
                        continue
                    subs.append(lang[type][sub_lang_code][raw_type])

                lang[type][lang_code][raw_type] = shuffle_merge_list(main, subs, )




def gen_data(out_dir, n, seed):

    """ Load lang và chia train/val/test """
    langs = {
        "train": {},
        "val": {},
        "test": {},
    }
    for lang_code in schema.LANGS:

        langs["train"][lang_code] = {}
        langs["val"][lang_code]   = {}
        langs["test"][lang_code]  = {}
        
        lang = load_lang(lang_code)        
        for key, data in lang.items():

            # Truyền full prefix do tập này thường rất ít (~5)
            if (key == "account_prefix"):
                langs["train"][lang_code][key] = data
                langs["val"][lang_code][key] = data
                langs["test"][lang_code][key] = data
                continue

            splited = split_pieces(data)
            langs["train"][lang_code][key] = splited
            langs["val"][lang_code][key] = splited
            langs["test"][lang_code][key] = splited



    



if __name__ == "__main__":
    vi = load_lang("vi")
    print(vi)