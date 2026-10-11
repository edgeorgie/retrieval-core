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
shared slice, published once instead of copy-pasted twice.

The extraction itself changed no behavior. Two correctness fixes have landed
since (see "Known limitations" below for what retrieval still cannot do).

Note that a fix landing here does **not** automatically reach the consumers.
Both pin this package to a commit SHA in their `package-lock.json`, so `npm ci`
in their CI and deploys installs that exact commit until the lockfile is bumped
on purpose. That is the intended behavior — a floating git dependency silently
changing a deployed server's retrieval on its next build would be worse — but it
means propagating a fix is a deliberate step in each consumer, not a
consequence of merging here.

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

## Verification

`npm run typecheck` (src **and** tests — `tsconfig.json` scopes `rootDir` to
`src` for emit, so `tsconfig.check.json` exists to widen it), `npm run build`,
`npm test`, and `npm audit --omit=dev` all run in CI on every push and PR.

The suite is 27 tests. They are written against independent oracles rather than
restating the implementation: `chunkFile`'s line ranges are checked by slicing
the original content separately and comparing; L2 normalization is checked as a
self-similarity invariant; `topK`'s ordering is checked against the language's
own `sort`; IDF is checked by the behavior it exists to produce (a term in every
document ranks below a term in one) rather than by recomputing its formula.

Every assertion was mutation-tested — the implementation was deliberately broken
and the suite confirmed to go red, then restored. Five mutations, each caught:
reverting the tokenizer, reverting the citation regex, making `diversify` mutate
its input, removing L2 normalization, and removing the chunk overlap.

That process is the only reason this suite is trustworthy, and it is worth
recording why. The first version of the end-to-end retrieval test indexed this
package's own `src/` — real code, no fixture to drift — and it was **vacuous**.
Fixing the tokenizer required writing a doc comment explaining the fix, and that
comment contains the words "cited indexes"; the test's query then matched that
comment as ordinary prose, with the tokenizer uninvolved, and ranked the *wrong*
file. It passed with the broken tokenizer restored. The guard was passing because
the prose explaining the guard mentioned the thing being guarded. It now runs
against a deliberately prose-free fixture, plus a second test asserting the
fixture stays that way.

## Known limitations

- **TF-IDF is lexical, not semantic.** A question sharing no vocabulary with the
  source it should retrieve scores zero. There is no synonym handling: asking
  about "authentication" will not retrieve a chunk that only ever says "login".
  This is the deliberate tradeoff for needing no model download and no API key.
- **The tokenizer is English-oriented.** Stopwords are an English list, and
  identifier splitting assumes ASCII case conventions.
- **`citedIndexes` cannot distinguish a citation from a `[n]` that is neither a
  citation nor a subscript** — e.g. inside a string literal in a code block. It
  excludes subscripts (`chunks[1]`), which is the common false positive, and
  accepts everything else.
- **No stemming.** "diversified" and "diversify" are different tokens, so a
  query's exact word form affects matching.
