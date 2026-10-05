import pytest

from normalize import normalize


def test_invalid_input():
    with pytest.raises(TypeError):
        normalize((1, 23, 4))

    with pytest.raises(TypeError):
        normalize(1)


def test_output():
    assert isinstance(normalize("xin chào"), list)
    for w in normalize("An trua 10k momo"):
        assert isinstance(w, str)


def test_accented_char():
    assert normalize("xin chào") == ["xin", "chào"]


def test_unaccented_char():
    assert normalize("hello world") == ["hello", "world"]
    assert normalize("An trua 10k momo") == ["an", "trua", "10k", "momo"]


def test_uppercase():
    assert normalize("HEllO woRLd") == ["hello", "world"]


def test_whitespace():
    assert normalize("  hello world   ") == ["hello", "world"]
    assert normalize("hello    world") == ["hello", "world"]


def test_tab():
    assert normalize("\t\thello world\t") == ["hello", "world"]
    assert normalize("\t\thello \t\t world\t") == ["hello", "world"]


def test_unicode_whitespace():
    assert normalize("\u00a0 hello \u3000 world  ") == ["hello", "world"]
    assert normalize("  \t ") == []


def test_empty():
    assert normalize("") == []


def test_currency():
    assert normalize("100$ 2500₫ 10$ 20k Đ 100đ") == [
        "100$",
        "2500₫",
        "10$",
        "20k",
        "đ",
        "100đ",
    ]


def test_consistency():
    assert normalize("100$ 2500₫ 10$") == normalize("100$ 2500₫ 10$")
    assert normalize("  hello world   ") == normalize("  hello world   ")


def test_complex_vie_sentence():
    test_result = normalize("2 tiếng trước mua 20k rau bằng NGÂN hàng ở quynh mắc")
    result = [
        "2",
        "tiếng",
        "trước",
        "mua",
        "20k",
        "rau",
        "bằng",
        "ngân",
        "hàng",
        "ở",
        "quynh",
        "mắc",
    ]
    assert test_result == result


def test_complex_eng_sentence():
    test_result = normalize("2$ for Metro Ticket in cash 2 hours ago")
    result = ["2$", "for", "metro", "ticket", "in", "cash", "2", "hours", "ago"]
    assert test_result == result
