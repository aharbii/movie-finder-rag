# Session Start — movie-finder-rag

Run these checks in parallel, then give a prioritised summary. Do not read any source files.

```bash
gh issue list --repo aharbii/movie-finder-rag --state open --limit 20 \
  --json number,title,labels,assignees
```

```bash
gh pr list --repo aharbii/movie-finder-rag --state open \
  --json number,title,state,labels,headRefName
```

```bash
gh issue list --repo aharbii/movie-finder-backend --state open --limit 5 \
  --json number,title,labels
```

```bash
git status && git log --oneline -5
```

Then summarise:

- **Open issues in this repo** — number, title, severity label
- **Open PRs** — which are ready to review, which are blocked
- **Backend parent issues** — any that involve RAG
- **Current branch and uncommitted changes**
- **Recommended next action** — one specific thing

Note: this is a standalone ingestion job (not part of the live API). Any embedding
model change here must be coordinated with `movie-finder-chain` (query-time embeddings).

Keep the summary under 20 lines. Do not propose solutions yet.
