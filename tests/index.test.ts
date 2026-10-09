/**
 * Smoke tests confirming the extracted chunk/vector/embedder/rag logic
 * behaves identically to its origin in repoask-mcp/ask-edgeorgie-mcp
 * (same inputs -> same outputs, since the code is an unmodified copy).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { chunkFile, embedText } from "../src/chunk.js";
import { dot, topK, diversify } from "../src/vector.js";
import { fitTfidf, embedBatchTfidf, tokenize } from "../src/embedder.js";
import { buildPrompt, citedIndexes, SYSTEM_PROMPT } from "../src/rag.js";

test("chunkFile splits a file into overlapping line windows", () => {
  const content = Array.from({ length: 100 }, (_, i) => `line ${i}`).join("\n");
  const chunks = chunkFile("file.ts", content);
  assert.ok(chunks.length > 1);
  assert.equal(chunks[0].path, "file.ts");
  assert.equal(chunks[0].start, 1);
});

test("fitTfidf + embedBatchTfidf + topK retrieves the right chunk", () => {
  const docs = [
    "the quick brown fox jumps over the lazy dog",
    "completely unrelated text about oceans and whales",
    "fox and dog running in the park",
  ];
  const model = fitTfidf(docs);
  const vectors = embedBatchTfidf(docs, model);
  const q = embedBatchTfidf(["fox dog"], model)[0];
  const scored = topK(q, vectors, 2, 0.01);
  assert.ok(scored.length > 0);
  // doc 0 and doc 2 both mention fox/dog, doc 1 doesn't
  const topIndexes = scored.map((s) => s.index);
  assert.ok(topIndexes.includes(0) || topIndexes.includes(2));
});

test("diversify caps chunks per file", () => {
  const scored = [
    { index: 0, score: 0.9 },
    { index: 1, score: 0.8 },
    { index: 2, score: 0.7 },
  ];
  const pathOf = (i: number) => (i === 2 ? "b.ts" : "a.ts");
  const out = diversify(scored, pathOf, 1);
  assert.equal(out.length, 2); // one from a.ts, one from b.ts
});

test("buildPrompt + citedIndexes round-trip", () => {
  const chunks = [embedTextChunk("a.ts", "hello"), embedTextChunk("b.ts", "world")];
  const prompt = buildPrompt("what?", "repo", chunks);
  assert.ok(prompt.includes("a.ts"));
  const answer = "See [1] and [2].";
  assert.deepEqual(citedIndexes(answer, 2), [0, 1]);
});

function embedTextChunk(path: string, text: string) {
  return { path, start: 1, end: 1, text };
}

test("tokenize + embedText basic sanity", () => {
  assert.ok(tokenize("Hello World_Test").includes("world_test"));
  const chunk = { path: "x.ts", start: 1, end: 2, text: "body" };
  assert.ok(embedText(chunk).startsWith("x.ts"));
});

test("dot product + SYSTEM_PROMPT sanity", () => {
  const a = new Float32Array([1, 0]);
  const b = new Float32Array([1, 0]);
  assert.equal(dot(a, b), 1);
  assert.ok(SYSTEM_PROMPT.length > 10);
});
