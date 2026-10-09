/**
 * @edgeorgie/retrieval-core
 *
 * Extracted from edgeorgie/repoask-mcp and edgeorgie/ask-edgeorgie-mcp, where
 * chunk.ts, vector.ts, embedder.ts and rag.ts were byte-identical copies
 * (confirmed by diff before this extraction — both servers run the exact
 * same local TF-IDF scoring/chunking/prompt-building logic, just over
 * different corpora: a fetched GitHub repo vs. a local resume/evidence
 * corpus). This package is that shared logic, with no changes to behavior.
 *
 * Deliberately NOT included: each server's own indexing orchestration
 * (indexer.ts / corpus.ts), its retrieval entrypoint (retrieval.ts, which
 * differs in field names and a minScore default between the two callers),
 * and its engine/tool-handler logic — those stay in each repo because they
 * are genuinely different per server, not duplicated.
 */

export * from "./chunk.js";
export * from "./vector.js";
export * from "./embedder.js";
export * from "./rag.js";
