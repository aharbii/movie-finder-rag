# Review PR — movie-finder-rag

**Repo:** `aharbii/movie-finder-rag`

Post findings as a comment only. Do not submit a GitHub review status.
The human decides whether to merge.

---

## Step 1 — Read PR, issue, and diff

```bash
gh pr view $ARGUMENTS --repo aharbii/movie-finder-rag
gh issue view [LINKED_ISSUE] --repo aharbii/movie-finder-rag
gh pr diff $ARGUMENTS --repo aharbii/movie-finder-rag
```

---

## Blocking findings

**RAG-specific patterns:**
- Strategy pattern violated for embedding providers (if/else on provider name in core logic)
- Local Qdrant container referenced (must always be Qdrant Cloud external)
- Embedding dimensions hardcoded or mismatched (must match configured model: 3072 for text-embedding-3-large)

**Python standards:**
- Missing type annotations, bare `except:`, `print()`, `type: ignore` without comment
- Line > 100 chars, no tests for new logic

**PR hygiene:** AI disclosure missing, issue not linked, Conventional Commits not followed.

---

## Post as a comment

```bash
gh pr comment $ARGUMENTS --repo aharbii/movie-finder-rag \
  --body "[review comment body]"
```

```
## Review — [date]
Reviewed by: [tool and model]

### Verdict
PASS — no blocking findings. Human call to merge.
— or —
BLOCKING FINDINGS — must fix before merge.

### Blocking findings
[file:line] — [issue and fix]

### Non-blocking observations
[file:line] — [observation]

### Cross-cutting gaps
[any item not handled and not noted in PR body]
```
