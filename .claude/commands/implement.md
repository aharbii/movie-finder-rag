# Implement Issue — movie-finder-rag

**Repo:** `aharbii/movie-finder-rag`
**Parent tracker:** `aharbii/movie-finder`
**Pre-commit:** `uv run pre-commit run --all-files`

Implement GitHub issue #$ARGUMENTS from `aharbii/movie-finder-rag`.

---

## Step 1 — Read the child issue

```bash
gh issue view $ARGUMENTS --repo aharbii/movie-finder-rag
```

Find the **Agent Briefing** section. If absent, ask the user to add it before proceeding.

---

## Step 2 — Read the parent issue for full context

```bash
gh issue view [PARENT_NUMBER] --repo aharbii/movie-finder
```

---

## Step 3 — Read only the files listed in the Agent Briefing

---

## Step 4 — Create the branch

```bash
git checkout main && git pull
git checkout -b [type]/[kebab-case-title]
```

---

## Step 5 — Implement

RAG ingestion-specific patterns:
- Strategy pattern for embedding providers — new provider = new class implementing the interface; no `if provider == "openai"` branching
- Qdrant Cloud is always external — no local Qdrant container
- Embedding model: OpenAI `text-embedding-3-large` (3072-dim)
- This is a standalone `uv` workspace — interpreter is at `${workspaceFolder}/.venv/bin/python`
- Batch embedding is preferred (see issue #19)

General backend standards:
- Type annotations required, `mypy --strict`
- Line length ≤ 100 chars
- No bare `except:`, no `print()`, async all the way
- Docstrings on all public classes/functions (Google style)

---

## Step 6 — Run quality checks

```bash
uv run pre-commit run --all-files
```

---

## Step 7 — Commit

```bash
git add [only changed files — never git add -A]
git commit -m "$(cat <<'EOF'
type(scope): short summary

[why]

Closes #$ARGUMENTS
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Step 8 — Open PR

```bash
gh pr create \
  --repo aharbii/movie-finder-rag \
  --title "type(scope): short summary" \
  --body "$(cat <<'EOF'
[PR body]

Closes #$ARGUMENTS
Parent: [PARENT_ISSUE_URL]

---
> AI-assisted implementation: Claude Code (claude-sonnet-4-6)
EOF
)"
```

---

## Step 9 — Cross-cutting comments

Comment on related issues (from Agent Briefing), the child issue, and the parent issue.
