---
name: tttm-commit-message
description: Commit message format for the Text-to-transaction repo (English, prefixed with the current module and Jira task, e.g. `[M0][TTTM-9] [Chore] ...`). Use whenever creating or rewording a git commit in this repo (git commit, "commit this", "write a commit message", /commit), instead of the global git-commit-message skill.
---

# Commit message format (Text-to-transaction)

Always write the whole commit message in **English**, even when the conversation, CLAUDE.md or Jira are in Vietnamese.

Every commit message has 3 parts.

## Part 1: header (required, exactly 1 line)

```
[M<n>][TTTM-<k>] [<Type>][path][to][changed][dir] <what changed, short>
```

- `[M<n>][TTTM-<k>]`: the module and the Jira issue this commit belongs to. One space separates it from the type.
- `<Type>` is one of: `Feat`, `Chore`, `Fix`.
- Then the path to the changed directory, one segment per bracket pair.
- If the path is too long, replace middle segments with `[...]`.
- If changes span several dirs, use their deepest common dir. Changes only at the repo root: no path segment.
- Then a short summary of what changed, lowercase imperative ("add", "fix", "switch").

Examples:
- `[M1][TTTM-12] [Feat][src] add label schema and LANGS list`
- `[M2][TTTM-16] [Feat][data][raw][vi] add Vietnamese MONEY fragments`
- `[M0][TTTM-9] [Chore] record Jira project key in CLAUDE.md`

### Finding the module and task

Check in this order and stop at the first that gives an answer:

1. The task the user named in this conversation, or a `TTTM-<k>` in the current branch name.
2. The work itself: match the changed files to a task in the CLAUDE.md schedule / Jira (e.g. `src/check_data.py` → `[M1] Định dạng JSONL và viết check_data.py`, TTTM-14).
3. Jira issues in project `TTTM` with status "In Progress".

Module comes from the task: its summary prefix `[M<n>]`, or its parent epic (`TTTM-1` = M0 … `TTTM-8` = M7).
Work that belongs to a module but to no single task: use that module's epic key (e.g. `[M0][TTTM-1]`).
If the task is still unclear or several tasks fit equally, ask the user before committing. Do not guess.

## Part 2: detail (required)

Leave ONE empty line after the header. Then describe the change in more detail: what changed, and why. Bullet points are fine.

## Part 3: before/after example (optional)

Add only when a short before/after example makes the change clearer (behavior, API, output, or code snippet). Separate it from part 2 with an empty line.

## Full example

```
[M1][TTTM-14] [Feat][src] add check_data.py JSONL validator

- Report line number and reason for each error; exit non-zero on any error.
- Check: missing field, tokens/tags length mismatch, unknown tag, invalid BIO, duplicate id, lang not in LANGS.

Before: no validation, bad lines reached training.
After:  python src/check_data.py data/synth/train.jsonl exits 1 and lists every bad line.
```

Pass the message with a heredoc or `git commit -F -` so the empty lines are kept.
Keep the attribution trailer (`Co-Authored-By: ...`) at the end when the session asks for it.
