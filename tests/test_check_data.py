import check_data
import subprocess
import sys

def test_check_valid_line():
    assert (
        check_data.check_line({
            "id": "real-vi-0001",
            "lang": "vi",
            "source": "real",
            "tokens": ["hôm", "qua", "5k", "đ", "gửi", "xe", "uit", "momo"],
            "tags": ["B-DATE","I-DATE","B-MONEY","I-MONEY","B-ITEM","I-ITEM","I-ITEM","B-ACCOUNT",],
        })
        == ""
    )


def test_check_tag_token_mismatch_line():
    assert (
        check_data.check_line({
            "id": "real-vi-0001",
            "lang": "vi",
            "source": "real",
            "tokens": ["hôm", "qua", "5k", "đ", "gửi", "xe", "uit", "momo"],
            "tags": ["B-DATE","I-DATE","B-MONEY","I-MONEY","B-ITEM","I-ITEM","I-ITEM",],
        })
        == "'tags' len must match with 'tokens' len"
    )


def test_check_invalid_tag_line():
    assert (
        "is not a valid tag" in 
        check_data.check_line({
            "id": "real-vi-0001",
            "lang": "vi",
            "source": "real",
            "tokens": ["hôm", "qua", "5k", "đ", "gửi", "xe", "uit", "momo"],
            "tags": ["B-TIME","I-TIME","B-MONEY","I-MONEY","B-ITEM","I-ITEM","I-ITEM","B-ACCOUNT"],
        })
    )


def test_check_invalid_BIO_line():
    assert (
        "Invalid BIO" in 
        check_data.check_line({
            "id": "real-vi-0001",
            "lang": "vi",
            "source": "real",
            "tokens": ["hôm", "qua", "5k", "đ", "gửi", "xe", "uit", "momo"],
            "tags": ["I-DATE","I-DATE","B-MONEY","I-MONEY","B-ITEM","I-ITEM","I-ITEM","B-ACCOUNT",],
        })
    )

def test_check_invalid_BIO_type_line():
    assert (
        "Invalid BIO" in 
        check_data.check_line({
            "id": "real-vi-0001",
            "lang": "vi",
            "source": "real",
            "tokens": ["hôm", "qua", "5k", "đ", "gửi", "xe", "uit", "momo"],
            "tags": ["B-DATE","I-MONEY","B-MONEY","I-MONEY","B-ITEM","I-ITEM","I-ITEM","B-ACCOUNT",],
        })
    )


def test_check_empty_token_line():
    assert (
        check_data.check_line({
            "id": "real-vi-0001",
            "lang": "vi",
            "source": "real",
            "tokens": [],
            "tags": [],
        })
        == "'tokens' and 'tags' must not empty"
    )


def test_check_invalid_lang_line():
    assert (
        "is not a valid lang" in 
        check_data.check_line({
            "id": "real-vi-0001",
            "lang": "fr",
            "source": "real",
            "tokens": ["hôm", "qua", "5k", "đ", "gửi", "xe", "uit", "momo"],
            "tags": ["B-DATE","I-DATE","B-MONEY","I-MONEY","B-ITEM","I-ITEM","I-ITEM","B-ACCOUNT",],
        })
    )


def test_check_missing_field_line():
    assert (
        "missing field" in 
        check_data.check_line({
            "id": "real-vi-0001",
            "lang": "vi",
            "tokens": ["hôm", "qua", "5k", "đ", "gửi", "xe", "uit", "momo"],
            "tags": ["B-DATE","I-DATE","B-MONEY","I-MONEY","B-ITEM","I-ITEM","I-ITEM","B-ACCOUNT",],
        })
    )


def test_check_invalid_field_line():
    assert (
        "is not a valid field" in 
        check_data.check_line({
            "id": "real-vi-0001",
            "unknown": "abc",
            "lang": "vi",
            "source": "real",
            "tokens": ["hôm", "qua", "5k", "đ", "gửi", "xe", "uit", "momo"],
            "tags": ["B-DATE","I-DATE","B-MONEY","I-MONEY","B-ITEM","I-ITEM","I-ITEM","B-ACCOUNT",],
        })
    )


def test_check_lines():
    assert check_data.check_lines("tests/data/real_test/vi.jsonl", 100) == []


def test_check_inlines():
    errors = check_data.check_lines("tests/data/real_test/vi_invalid.jsonl", 100)
    assert "missing field" in errors[0]["err"] and errors[0]["line"] == 1
    assert "duplicate is detected" in errors[1]["err"] and errors[1]["line"] == 6
    assert "'tags' len must match with 'tokens' len" in errors[2]["err"] and errors[2]["line"] == 10


def test_cli_exit_code():
    bad = subprocess.run([sys.executable, "src/check_data.py", "tests/data/real_test/vi_invalid.jsonl"], capture_output=True)
    ok = subprocess.run([sys.executable, "src/check_data.py", "tests/data/real_test/vi.jsonl"], capture_output=True)
    assert bad.returncode == 1
    assert ok.returncode == 0
    assert "line 6" in bad.stdout.decode("utf-8")