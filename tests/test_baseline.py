from baseline import predict


def test_example_sentence():
    tokens = ["hôm", "qua", "5k", "đ", "gửi", "xe", "uit", "momo"]
    assert predict(tokens) == ["B-DATE", "I-DATE", "B-MONEY", "I-MONEY",
                               "B-ITEM", "I-ITEM", "I-ITEM", "B-ACCOUNT"]


def test_cue_word_and_stopword():
    # "on" ở ranh giới -> O; "heo đất" sau "via" -> ACCOUNT nhờ từ gợi ý
    tokens = ["spent", "twenty", "dollars", "on", "snacks", "via", "heo", "đất"]
    assert predict(tokens) == ["B-TYPE", "B-MONEY", "I-MONEY", "O",
                               "B-ITEM", "O", "B-ACCOUNT", "I-ACCOUNT"]


def test_number_word_alone_is_not_money():
    assert predict(["two", "days", "ago"]) == ["B-DATE", "I-DATE", "I-DATE"]
