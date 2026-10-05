from schema import tag_to_code


def test_tag_to_code():
    assert tag_to_code("B-MONEY") == 1