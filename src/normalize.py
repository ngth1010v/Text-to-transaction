"""Chuyển từ text -> list[str], chỉ hỗ trợ các ngôn ngữ có từ được cách bằng dấu cách (vie,eng,...), không support trung,hàn,nhật,..."""


def normalize(text: str):

    if not isinstance(text, str):
        raise TypeError("text phải là string")

    text = text.lower()  # kotlin: .lowercase()
    text = text.strip()  # kotlin: .trim()
    words = text.split()  # kotlin: .split(Regex("(?U)\\s+")).filter { it.isNotEmpty() }

    return words
