import json
import random
import shutil

import pytest

import check_data
import gen_data
import schema

NON_CONTENT = ("account_prefix", "account_suffix", "templates")


def _read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def test_gen_data_valid_and_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    gen_data.gen_data(a, 400, 1)
    gen_data.gen_data(b, 400, 1)

    ids = set()
    for split in ["train", "val", "test"]:
        assert (a / f"{split}.jsonl").read_text(encoding="utf-8") == (b / f"{split}.jsonl").read_text(encoding="utf-8")
        for row in _read(a / f"{split}.jsonl"):
            assert check_data.check_line(row) == ""
            assert row["id"] not in ids
            ids.add(row["id"])


def test_fill_template_account_is_one_span():
    pools = {"account": [["ví", "momo", "chính"]], "money": [["5k"]], "item": [["cà", "phê"]],
             "date": [["hôm", "qua"]], "type": [["chi"]]}
    tokens, tags, mixed = gen_data.fill_template("{ITEM} {MONEY} qua {ACCOUNT}", pools, random.Random(0))
    assert tokens == ["cà", "phê", "5k", "qua", "ví", "momo", "chính"]
    assert tags == ["B-ITEM", "I-ITEM", "B-MONEY", "O", "B-ACCOUNT", "I-ACCOUNT", "I-ACCOUNT"]
    assert mixed is False


def test_fill_template_marks_mixed():
    pools = {"account": [["momo"]], "money": [["5k"]], "item": [["rent"]], "date": [["hôm", "qua"]], "type": [["chi"]]}
    _, _, mixed = gen_data.fill_template("{ITEM} {MONEY}", pools, random.Random(0), {("rent",)})
    assert mixed is True


def test_noise_keeps_length_and_bio():
    tokens = ["chi", "50k", "ăn", "sáng", "qua", "ví", "momo"]
    tags = ["B-TYPE", "B-MONEY", "B-ITEM", "I-ITEM", "O", "B-ACCOUNT", "I-ACCOUNT"]
    for seed in range(200):
        t, g = gen_data.add_noise(tokens, tags, "vi", random.Random(seed))
        assert len(t) == len(g)
        row = {"id": "x", "lang": "vi", "source": "synthetic", "tokens": t, "tags": g}
        assert check_data.check_line(row) == ""


def test_load_lang_has_no_duplicates():
    for lang_code in schema.LANGS:
        for key, pieces in gen_data.load_lang(lang_code).items():
            assert len(pieces) == len(set(pieces)), f"{lang_code}/{key} có mảnh trùng"


@pytest.mark.parametrize("seed", [0, 1, 2, 90, 123])
def test_no_piece_in_two_splits(seed):
    """Không mảnh gốc nào (ITEM, ACCOUNT, ...) nằm ở 2 split, kể cả sau khi trộn ngôn ngữ."""
    rng = random.Random(seed)
    langs = gen_data.split_all(rng)
    gen_data.mix_lang(langs, rng)

    owner = {}
    for split in gen_data.SPLITS:
        for lang_code in schema.LANGS:
            for key, pieces in langs[split][lang_code].items():
                if key in NON_CONTENT:
                    continue
                for p in pieces:
                    assert owner.setdefault(p, split) == split, f"'{p}' có ở {owner[p]} và {split}"


def test_mixed_sentence_ratio(tmp_path):
    counts = gen_data.gen_data(tmp_path, 4000, 90)
    assert 0.10 <= counts["mixed_ratio"] <= 0.20


def test_account_names_not_hardcoded_prefix():
    """Prefix/suffix lấy từ data/raw, mỗi ngôn ngữ có đủ để model thấy nhiều dạng tên."""
    for lang_code in schema.LANGS:
        lang = gen_data.load_lang(lang_code)
        assert len(lang["account_prefix"]) >= 3
        assert len(lang["account_suffix"]) >= 3


def test_new_language_without_code_change(tmp_path, monkeypatch):
    """Thêm `xx` vào LANGS + thư mục data/raw/xx là sinh được câu lang=xx."""
    raw = tmp_path / "raw"
    shutil.copytree(gen_data.RAW_DIR, raw)
    shutil.copytree(raw / "vi", raw / "xx")
    monkeypatch.setattr(gen_data, "RAW_DIR", raw)
    monkeypatch.setattr(schema, "LANGS", ["vi", "en", "xx"])

    out = tmp_path / "out"
    gen_data.gen_data(out, 600, 1)

    rows = _read(out / "train.jsonl")
    assert any(r["lang"] == "xx" for r in rows)
    assert all(check_data.check_line(r) == "" for r in rows)
