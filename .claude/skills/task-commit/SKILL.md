---
name: task-commit
description: Review Jira TTTM tasks that are In Progress against the repo, report which are done and what is missing, then (after the user confirms each one) commit its files and move the task to Done. Run only when the user calls /task-commit.
disable-model-invocation: true
---

# task-commit

Talk to the user in Vietnamese. Commit messages follow the `tttm-commit-message` skill (English, `[M<n>][TTTM-<k>] ...` prefix).

Jira: site `https://ai-engineer-learning.atlassian.net`, cloudId `c3173150-c892-4fa8-8241-9358a18c9a7b`, project `TTTM`.

## Step 1: read In Progress tasks

`searchJiraIssuesUsingJql` with `project = TTTM AND status = "In Progress" ORDER BY key`, fields `summary`, `description`, `duedate`, `parent`.
If there are none, say so and stop.

## Step 2: check each task against the repo

For each task, take every line under **Tiêu chí hoàn thành** in its description and check it with evidence:

- Read the files the task names (`src/...`, `tests/...`, `data/...`, `README.md`, `CLAUDE.md`).
- Run what proves it: `uv run pytest -q` (or the task's test file), the script the task names (e.g. `uv run python src/check_data.py <file>`), `git status --short` for "đã commit" criteria.
- Check the project rules in CLAUDE.md that apply (forbidden libraries in section 6, Kotlin-portable `normalize.py` in section 5, Vietnamese comments, etc.).
- A `[M<n>] Ghi chú học được` task is done when the task has a comment with the notes; check with `getJiraIssue`.

Never mark a criterion as met without evidence. If a criterion cannot be checked automatically, say so and ask.

Report one block per task:

```
TTTM-<k> [M<n>] <summary> — XONG | CHƯA XONG
- [x] <criterion> — <evidence: test name, command output, file:line>
- [ ] <criterion> — <what is missing>
```

## Step 3: tasks that are not done

List what is still missing, as short concrete actions (file, function, test to add). Do not commit, do not change Jira status.

## Step 4: tasks that are done

For each done task, one at a time:

1. Write a short review (3-5 bullets): what was done well, risks, small issues (typos, missing edge-case tests, rule violations).
2. Ask with `AskUserQuestion` (in the chat, do not end the turn): "Done TTTM-<k> luôn không?" with options `Done luôn` and `Chưa`.
3. If `Done luôn`:
   - Stage only files that belong to this task (`git add <paths>`, never blindly `git add -A`). If changed files mix several tasks or it is unclear which belong here, show the list and ask.
   - If nothing is left to commit (work already committed), skip the commit and say so.
   - Commit using the `tttm-commit-message` format, with a heredoc / `git commit -F -`.
   - Move the Jira task to Done: list the issue's transitions (`listJiraIssueTransitions`; find it with the Atlassian `discover` tool if it is not loaded), pick the one whose target status is "Done", then call `transitionJiraIssue` with its `transitionId`. Transition names can differ from the target status, so match on the target.
   - Report: commit hash + header, new Jira status.
4. If `Chưa`: give a numbered improvement list (concrete: file, what to change, which test to add). No commit, no Jira change.

## Rules

- Do not push.
- Do not edit code to make a task pass; this skill only checks, commits, and updates Jira. Suggest fixes instead.
- Do not touch tasks that are not In Progress.
- End with a one-line summary: `<n> xong (đã done: ...), <m> chưa xong (...)`.
