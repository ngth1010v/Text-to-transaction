import schema


def test_tag():
    for t in schema.TAGS:
        assert schema.TAGS[schema.TAG2ID[t]] == t


def test_langs():
    """Check is list"""
    assert isinstance(schema.LANGS, list)

    """Check is duplicate"""
    assert len(schema.LANGS) == len(set(schema.LANGS))

    """Check is all string"""
    for l in schema.LANGS:
        assert isinstance(l, str)
