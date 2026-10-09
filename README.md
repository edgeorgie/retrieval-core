# @edgeorgie/retrieval-core

Shared TF-IDF chunking/scoring/prompt-building core extracted from
[edgeorgie/repoask-mcp](https://github.com/edgeorgie/repoask-mcp) and
[edgeorgie/ask-edgeorgie-mcp](https://github.com/edgeorgie/ask-edgeorgie-mcp).

## Why this exists

Both MCP servers ran local TF-IDF retrieval (no embedding model, no API key)
over their own corpus — one over a fetched public GitHub repo, the other over
a local resume/evidence corpus. A direct diff of their source trees before
this extraction showed `chunk.ts`, `vector.ts`, `embedder.ts` and `rag.ts`
were **byte-identical** between the two repos — a real, unforced case of
duplicated logic, not an artificial abstraction. This package is exactly that
shared slice, unmodified, published once instead of copy-pasted twice.

What's deliberately **not** in here: each server's indexing orchestration
(`indexer.ts` / `corpus.ts`), its retrieval entrypoint (`retrieval.ts`, which
differs slightly between the two callers — field names, a `minScore`
default), and its tool/engine logic. Those stay in each repo because they are
genuinely different, not duplicated — forcing them into this package would
have been an artificial abstraction.

## What's in it

- `chunk.ts` — splits a file into overlapping line-range chunks (Markdown
  breaks on headings first).
- `vector.ts` — cosine similarity (dot product on L2-normalized vectors),
  top-k scoring, per-file diversification.
- `embedder.ts` — TF-IDF vocabulary fitting + embedding (no model download,
  no API key).
- `rag.ts` — numbered-source prompt construction + `[n]` citation parsing.

## Install

No npm registry publish (interactive `npm login` isn't available in this
environment — same blocker documented for this account's other packages).
Installed as a git dependency instead, which is free and requires no
registry account:

```json
"dependencies": {
  "@edgeorgie/retrieval-core": "github:edgeorgie/retrieval-core"
}
```

npm clones the repo and runs its `prepare` script (`npm run build`) on
install, so `dist/` is always built fresh — no committed build artifacts.

## Used by

- [edgeorgie/repoask-mcp](https://github.com/edgeorgie/repoask-mcp)
- [edgeorgie/ask-edgeorgie-mcp](https://github.com/edgeorgie/ask-edgeorgie-mcp)
