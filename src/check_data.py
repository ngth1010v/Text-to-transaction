"""Kiểm tra file JSONL: độ dài tokens = tags, BIO hợp lệ."""

import schema
import json
import argparse
import sys

# ===================================================================================
# Check line (dict), return "" if valid
# ===================================================================================
def check_line(line: dict):

    if not isinstance(line, dict):
        return "Line must be a dict"

    line_fields = line.keys()
    for f in line_fields:
        if f not in schema.FIELDS:
            return f"'{f}' is not a valid field"
    for f in schema.FIELDS:
        if f not in line_fields:
            return f"missing field '{f}'"

    if len(line["tags"]) != len(line["tokens"]):
        return "'tags' len must match with 'tokens' len"

    if len(line["tokens"]) == 0:
        return "'tokens' and 'tags' must not empty"

    if line["lang"] not in schema.LANGS:
        return f"'{line['lang']}' is not a valid lang"


    for t in line["tags"]:
        if t not in schema.TAGS:
            return f"'{t}' is not a valid tag"

    for i in range(len(line["tags"]) - 1):
        tag = line["tags"][i]
        next_tag = line["tags"][i + 1]
        if next_tag[0] == "I":
            if tag[0] not in ["B", "I"]:
                return "Invalid BIO"
            if tag.split('-')[1] != next_tag.split('-')[1]:
                return "Invalid BIO"
    if line["tags"][0][0] == "I":
        return "Invalid BIO"

    return ""


# ===================================================================================
# Check lines from files, return [] if valid
# ===================================================================================
def check_lines(path: str, error_limit: int):

    line_ids = set()
    errors = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                l = json.loads(line)
            except json.JSONDecodeError as e:
                err = f"Invalid JSON: {e}"
            else:
                err = check_line(l)
                if err == "":
                    line_id = l["id"]
                    if line_id in line_ids:
                        err = f"line_id duplicate is detected: {line_id}"
                    else:
                        line_ids.add(line_id)

            if err != "":
                errors.append({"line": line_no, "err": err})
                if len(errors) >= error_limit:
                    break

    return errors


sys.stdout.reconfigure(encoding="utf-8")
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Kiểm tra file JSONL")
    parser.add_argument("path", help="đường dẫn file .jsonl")
    parser.add_argument("--limit", type=int, default=100, help="dừng sau N lỗi")
    args = parser.parse_args()

    errors = check_lines(args.path, args.limit)

    for e in errors:
        print(f"line {e['line']}: {e['err']}")

    if errors:
        print(f"{len(errors)} lỗi")
        sys.exit(1)
    print("OK")