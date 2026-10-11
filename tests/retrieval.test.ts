/**
 * End-to-end retrieval test: index a small corpus of code and assert that a
 * prose question reaches the chunk defining the symbol it asks about.
 *
 * WHY A FIXTURE AND NOT THIS PACKAGE'S OWN `src/`
 *
 * The first version of this file indexed `src/`, which looked like the most
 * honest possible corpus — real code, no fixture to drift. It was vacuous, and
 * mutation testing is the only reason that is known: reverting the tokenizer to
 * the broken `.toLowerCase()`-then-split-on-`_` version left both tests in this
 * file GREEN.
 *
 * The cause is worth recording, because it is subtle and recurs. Fixing the
 * tokenizer meant writing a doc comment explaining the fix, and that comment
 * contains the words "cited indexes". Those words then exist as ordinary prose
 * in `embedder.ts`, so the query "how are cited indexes parsed" matches
 * `embedder.ts` on prose alone, with no tokenizer involvement at all — and
 * `embedder.ts` is the WRONG file; `citedIndexes` lives in `rag.ts`. The test
 * asserted "some expected file is in the top 3" and the prose match satisfied it.
 * The guard was passing because the comment explaining the guard mentioned the
 * thing being guarded.
 *
 * So the corpus below is code-shaped and deliberately prose-free: identifiers in
 * camelCase, and no comment anywhere that spells out those identifiers as
 * separate English words. The ONLY route from one of these questions to the
 * right chunk is splitting the identifier on its case boundaries. Verified by
 * mutation: with the old tokenizer, every case here fails.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { chunkFile, embedText, type Chunk } from "../src/chunk.js";
import { fitTfidf, embedTfidf, embedBatchTfidf } from "../src/embedder.js";
import { topK } from "../src/vector.js";

/**
 * A fixture repo. Each file defines one camelCase export and contains filler
 * lines so it survives chunking; the filler is deliberately meaningless so it
 * cannot accidentally match a query.
 */
const FIXTURE: Record<string, string> = {
  "src/billing.ts": [
    "export function calculateInvoiceTotal(items: Item[]): number {",
    "  let acc = 0;",
    "  for (const it of items) acc += it.qty * it.unit;",
    "  return acc;",
    "}",
    ...Array.from({ length: 12 }, (_, i) => `const fillerAlpha${i} = ${i};`),
  ].join("\n"),
  "src/auth.ts": [
    "export function refreshAccessToken(old: string): string {",
    "  return old.split('.').reverse().join('.');",
    "}",
    ...Array.from({ length: 12 }, (_, i) => `const fillerBeta${i} = ${i};`),
  ].join("\n"),
  "src/cache.ts": [
    "export function evictStaleEntries(store: Map<string, number>, ttl: number): void {",
    "  for (const [k, v] of store) if (v < ttl) store.delete(k);",
    "}",
    ...Array.from({ length: 12 }, (_, i) => `const fillerGamma${i} = ${i};`),
  ].join("\n"),
  "src/notify.ts": [
    "export function dispatchWebhookRetry(url: string, attempt: number): number {",
    "  return attempt * 2;",
    "}",
    ...Array.from({ length: 12 }, (_, i) => `const fillerDelta${i} = ${i};`),
  ].join("\n"),
};

function indexFixture(): (q: string, k?: number) => Chunk[] {
  const chunks = Object.entries(FIXTURE).flatMap(([path, content]) => chunkFile(path, content));
  const texts = chunks.map(embedText);
  const model = fitTfidf(texts);
  const vectors = embedBatchTfidf(texts, model);
  return (q, k = 2) => topK(embedTfidf(q, model), vectors, k, 0).map((s) => chunks[s.index]);
}

test("a prose question retrieves the file defining the camelCase symbol it describes", () => {
  const search = indexFixture();

  // Each question uses the identifier's words as separate English words, which
  // is how a human asks. None of these words appear as prose anywhere in the
  // fixture — only inside the camelCase identifier itself.
  const cases: Array<[question: string, expectedFile: string]> = [
    ["how is the invoice total calculated", "src/billing.ts"],
    ["how is an access token refreshed", "src/auth.ts"],
    ["how are stale entries evicted", "src/cache.ts"],
    ["how is a webhook retry dispatched", "src/notify.ts"],
  ];

  for (const [question, expectedFile] of cases) {
    const hits = search(question);
    assert.ok(hits.length > 0, `"${question}" retrieved nothing at all`);
    assert.equal(
      hits[0].path,
      expectedFile,
      `"${question}" should rank ${expectedFile} first; got ${JSON.stringify(hits.map((h) => h.path))}`
    );
  }
});

test("the retrieval route for these questions is case-splitting, not prose matching", () => {
  // Guards the premise the test above rests on. If someone later adds a comment
  // to the fixture spelling an identifier out in English, the test above would
  // keep passing for the wrong reason — exactly the vacuity this file's header
  // describes. This asserts the fixture contains no such prose.
  const words = ["invoice total", "access token", "stale entries", "webhook retry"];
  const corpus = Object.values(FIXTURE).join("\n").toLowerCase();
  for (const w of words) {
    assert.ok(
      !corpus.includes(w),
      `fixture leaked "${w}" as prose — the retrieval test above can now pass without the tokenizer`
    );
  }
});
