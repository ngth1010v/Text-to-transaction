from metrics import extract_spans, prf, score, score_by_lang
from schema import TAG2ID


def test_extract_spans_basic():
    tags = ["B-DATE", "I-DATE", "O", "B-MONEY", "B-MONEY", "I-MONEY"]
    assert extract_spans(tags) == {("DATE", 0, 2), ("MONEY", 3, 4), ("MONEY", 4, 6)}


def test_exact_match():
    g = [["B-DATE", "I-DATE", "B-MONEY"]]
    s = score(g, g)
    assert s["micro"]["f1"] == 1.0
    assert s["per_type"]["DATE"]["f1"] == 1.0


def test_boundary_mismatch():
    # gold dài 3, pred dài 2: sai cả tp lẫn biên -> fp=1, fn=1
    g = [["B-ITEM", "I-ITEM", "I-ITEM"]]
    p = [["B-ITEM", "I-ITEM", "O"]]
    m = score(g, p)["micro"]
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 1)
    assert m["f1"] == 0.0


def test_wrong_type():
    # cùng vị trí nhưng sai loại: ACCOUNT bị thiếu, ITEM bị thừa
    s = score([["B-ACCOUNT"]], [["B-ITEM"]])["per_type"]
    assert (s["ACCOUNT"]["fn"], s["ITEM"]["fp"]) == (1, 1)
    assert s["ACCOUNT"]["tp"] == 0


def test_no_spans():
    # không có cụm nào: không chia cho 0
    m = score([["O", "O"]], [["O", "O"]])["micro"]
    assert (m["tp"], m["fp"], m["fn"]) == (0, 0, 0)
    assert m["f1"] == 0.0
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


def test_orphan_i():
    # I-X mồ côi mở cụm mới; I-X sau B-Y khác loại cũng vậy
    assert extract_spans(["O", "I-ITEM", "I-ITEM"]) == {("ITEM", 1, 3)}
    assert extract_spans(["B-DATE", "I-ITEM"]) == {("DATE", 0, 1), ("ITEM", 1, 2)}


def test_ignore_minus_100():
    # gold -100 ở vị trí đệm; pred có rác ở đó thì không được tính
    g = [[TAG2ID["B-MONEY"], -100, -100]]
    p = [[TAG2ID["B-MONEY"], TAG2ID["I-MONEY"], TAG2ID["I-MONEY"]]]
    m = score(g, p)["micro"]
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0)


def test_hand_calculated_mix():
    # gold: DATE(0,2) MONEY(2,3); pred: DATE(0,2) ITEM(2,3) -> tp=1 fp=1 fn=1
    g = [["B-DATE", "I-DATE", "B-MONEY"]]
    p = [["B-DATE", "I-DATE", "B-ITEM"]]
    r = score(g, p)
    assert r["micro"]["p"] == r["micro"]["r"] == r["micro"]["f1"] == 0.5
    assert r["per_type"]["DATE"]["f1"] == 1.0
    assert r["per_type"]["MONEY"]["f1"] == 0.0


def test_score_by_lang():
    langs = ["vi", "en"]
    golds = [["B-MONEY"], ["B-MONEY"]]
    preds = [["B-MONEY"], ["O"]]
    r = score_by_lang(langs, golds, preds)
    assert set(r) == {"vi", "en"}
    assert r["vi"]["micro"]["f1"] == 1.0
    assert r["en"]["micro"]["f1"] == 0.0
