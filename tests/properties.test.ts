/**
 * Property and invariant tests for the retrieval primitives.
 *
 * These deliberately avoid restating the implementation. Each assertion is
 * checked against something independent of the code under test: a reference
 * oracle computed with a language builtin, a mathematical invariant the
 * algorithm claims to hold, or a behaviour a caller depends on.
 *
 * The weaker `index.test.ts` smoke tests are kept as-is — they pin the exported
 * surface — and this file covers correctness.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { chunkFile, embedText, type Chunk } from "../src/chunk.js";
import { dot, topK, diversify, type Scored } from "../src/vector.js";
import { fitTfidf, embedTfidf, embedBatchTfidf, tokenize } from "../src/embedder.js";
import { buildPrompt, citedIndexes } from "../src/rag.js";

// ---------------------------------------------------------------------------
// tokenize
// ---------------------------------------------------------------------------

test("tokenize splits identifiers on case boundaries as well as underscores", () => {
  // Oracle: the words a human reads in the identifier. Checked for all three
  // naming conventions the doc comment claims to support, since only snake_case
  // worked before — camelCase collapsed into one opaque token.
  for (const id of ["getUserName", "get_user_name", "get-user-name"]) {
    const tokens = tokenize(id);
    for (const word of ["get", "user", "name"]) {
      assert.ok(tokens.includes(word), `${id} should yield sub-token "${word}", got ${JSON.stringify(tokens)}`);
    }
  }
});

test("tokenize splits an acronym running into a word", () => {
  // Two-pass case splitting exists for this case: the common
  // lower->upper rule cannot see HTTPServer, because every character around
  // the real boundary is already uppercase.
  const tokens = tokenize("HTTPServerConfig");
  assert.ok(tokens.includes("http"), JSON.stringify(tokens));
  assert.ok(tokens.includes("server"), JSON.stringify(tokens));
  assert.ok(tokens.includes("config"), JSON.stringify(tokens));
});

test("tokenize still emits the whole identifier, so exact-symbol queries keep working", () => {
  // The sub-token split must be ADDITIVE. If it replaced the whole token, a
  // query pasting an exact symbol name would stop matching the chunk that
  // defines it — a regression in the most common real query shape.
  assert.ok(tokenize("embedBatchTfidf").includes("embedbatchtfidf"));
  assert.ok(tokenize("get_user_name").includes("get_user_name"));
});

test("tokenize drops stopwords and single characters, and emits no duplicates of the whole token", () => {
  const tokens = tokenize("the a of query");
  assert.deepEqual(tokens, ["query"]);
  // A token with no case/underscore boundary must appear exactly once, not
  // twice via the sub-token path.
  assert.deepEqual(tokenize("query"), ["query"]);
});

test("stopwords are filtered on the sub-token path too, not only on whole tokens", () => {
  // Case splitting opens a back door the whole-token filter does not cover:
  // `toString` yields a "to" sub-token, `isValid` an "is", `canRetry` a "can".
  // Those are the exact terms the stopword list exists to suppress, and they
  // occur in nearly every identifier in a TypeScript codebase.
  for (const [identifier, leaked] of [
    ["toString", "to"],
    ["isValid", "is"],
    ["canRetry", "can"],
    ["of_the_thing", "the"],
  ] as const) {
    const tokens = tokenize(identifier);
    assert.ok(
      !tokens.includes(leaked),
      `"${identifier}" leaked stopword "${leaked}": ${JSON.stringify(tokens)}`
    );
  }
  // The meaningful halves must survive — this must not become a blanket
  // "drop short sub-tokens" rule.
  assert.ok(tokenize("toString").includes("string"));
  assert.ok(tokenize("isValid").includes("valid"));
  assert.ok(tokenize("canRetry").includes("retry"));
});

// ---------------------------------------------------------------------------
// TF-IDF: mathematical invariants, not recomputation of the same formula
// ---------------------------------------------------------------------------

test("every embedded vector is L2-normalized, so dot() really is cosine similarity", () => {
  // This is the contract vector.ts's `dot` doc comment depends on. Checked as a
  // property over a corpus with repeated terms, varying lengths and an
  // out-of-vocabulary document.
  const docs = [
    "alpha beta gamma",
    "beta beta beta delta",
    "gamma epsilon",
    "a single term: zeta",
  ];
  const model = fitTfidf(docs);
  for (const v of embedBatchTfidf(docs, model)) {
    assert.ok(Math.abs(dot(v, v) - 1) < 1e-5, `self-similarity ${dot(v, v)} is not 1`);
  }
  // A document sharing no vocabulary with the corpus is the one legitimate
  // exception: it has no non-zero component to normalize.
  const oov = embedTfidf("zzzz qqqq", model);
  assert.equal(dot(oov, oov), 0);
});

test("cosine similarity is bounded by 1 and a document is its own nearest neighbour", () => {
  const docs = ["retrieval scoring chunks", "unrelated oceans whales", "scoring chunks ranking"];
  const model = fitTfidf(docs);
  const vectors = embedBatchTfidf(docs, model);
  for (let i = 0; i < docs.length; i++) {
    const self = dot(vectors[i], vectors[i]);
    for (let j = 0; j < docs.length; j++) {
      if (i === j) continue;
      assert.ok(
        dot(vectors[i], vectors[j]) <= self + 1e-6,
        `doc ${j} scored higher against doc ${i} than doc ${i} did against itself`
      );
    }
  }
});

test("IDF downweights a term present in every document below one present in a few", () => {
  // Independent check of the point of IDF, without recomputing log((n+1)/(df+1)).
  // "common" appears in all 3 documents; "rare" in one. A query for both should
  // rank the document containing "rare" first.
  const docs = ["common rare", "common filler", "common other"];
  const model = fitTfidf(docs);
  const vectors = embedBatchTfidf(docs, model);
  const q = embedTfidf("common rare", model);
  const ranked = topK(q, vectors, 3, 0);
  assert.equal(ranked[0].index, 0, JSON.stringify(ranked));
});

// ---------------------------------------------------------------------------
// topK / diversify
// ---------------------------------------------------------------------------

test("topK returns at most k results, sorted descending, all above minScore", () => {
  const vectors = [
    new Float32Array([1, 0]),
    new Float32Array([0, 1]),
    new Float32Array([0.6, 0.8]),
    new Float32Array([0.8, 0.6]),
  ];
  const q = new Float32Array([1, 0]);
  const out = topK(q, vectors, 2, 0.5);
  assert.ok(out.length <= 2);
  // Oracle: sortedness checked against the language's own sort of the scores.
  const scores = out.map((s) => s.score);
  assert.deepEqual(scores, [...scores].sort((a, b) => b - a));
  for (const s of out) assert.ok(s.score >= 0.5, `${s.score} slipped past minScore`);
  // And the orthogonal vector, scoring 0, must be excluded entirely.
  assert.ok(!out.some((s) => s.index === 1));
});

test("diversify caps per file and preserves input order without mutating the caller's array", () => {
  const scored: Scored[] = [
    { index: 0, score: 0.9 },
    { index: 1, score: 0.8 },
    { index: 2, score: 0.7 },
    { index: 3, score: 0.6 },
  ];
  const snapshot = JSON.stringify(scored);
  const pathOf = (i: number) => (i < 3 ? "a.ts" : "b.ts");
  const out = diversify(scored, pathOf, 2);

  assert.deepEqual(out.map((s) => s.index), [0, 1, 3], "should keep 2 from a.ts and 1 from b.ts, in order");
  // Mutating the caller's array would be a real bug: callers hold the full
  // ranked list and re-diversify it with a different cap.
  assert.equal(JSON.stringify(scored), snapshot, "diversify must not mutate its input");
});

// ---------------------------------------------------------------------------
// chunkFile
// ---------------------------------------------------------------------------

test("chunk windows cover every line of the file and overlap rather than gap", () => {
  const content = Array.from({ length: 100 }, (_, i) => `line ${i}`).join("\n");
  const chunks = chunkFile("f.ts", content, 40, 6);
  assert.ok(chunks.length > 1);
  assert.equal(chunks[0].start, 1);
  assert.equal(chunks[chunks.length - 1].end, 100, "last window must reach the final line");
  for (let i = 1; i < chunks.length; i++) {
    assert.ok(
      chunks[i].start <= chunks[i - 1].end,
      `gap between window ${i - 1} (ends ${chunks[i - 1].end}) and ${i} (starts ${chunks[i].start})`
    );
    assert.ok(chunks[i].start > chunks[i - 1].start, "windows must advance, or chunking would not terminate");
  }
});

test("a chunk's text is exactly the lines its start/end claim", () => {
  // The whole point of chunking is that a citation's line range is accurate.
  // Oracle: slice the original content independently and compare.
  const lines = Array.from({ length: 60 }, (_, i) => `content ${i}`);
  const chunks = chunkFile("f.ts", lines.join("\n"), 25, 5);
  for (const c of chunks) {
    assert.equal(c.text, lines.slice(c.start - 1, c.end).join("\n").trim(), `window ${c.start}-${c.end} text mismatch`);
  }
});

test("markdown breaks on a heading while plain files break on the line cap", () => {
  const lines = Array.from({ length: 120 }, (_, i) => (i % 25 === 24 ? `## heading ${i}` : `body ${i}`));
  const md = chunkFile("doc.md", lines.join("\n"), 40, 6);
  const ts = chunkFile("doc.ts", lines.join("\n"), 40, 6);
  // The .ts file has no heading logic, so its first window is the full cap.
  assert.equal(ts[0].end, 40);
  // The .md file pulls its first boundary back to a heading line.
  assert.ok(md[0].end < 40, `md first window ended at ${md[0].end}, expected a heading pull-back`);
  assert.ok(/^#{1,4}\s/.test(lines[md[0].end]), `line ${md[0].end} is not a heading: ${lines[md[0].end]}`);
});

test("chunkFile terminates on pathological markdown where every line is a heading", () => {
  // The heading scan walks backwards and could pick a boundary that fails to
  // advance `start`, which would loop forever. `Math.max(start + 1, ...)` is the
  // guard; this asserts it holds for the worst input.
  const dense = Array.from({ length: 200 }, (_, i) => `## h${i}`).join("\n");
  const chunks = chunkFile("dense.md", dense, 40, 6);
  assert.ok(chunks.length > 0);
  assert.equal(chunks[chunks.length - 1].end, 200);
});

test("chunkFile drops content too short to be a useful citation, and normalizes CRLF", () => {
  assert.deepEqual(chunkFile("t.ts", "const a = 1;"), []);
  const crlf = chunkFile("t.ts", "a".repeat(30) + "\r\n" + "b".repeat(30));
  assert.equal(crlf.length, 1);
  assert.ok(!crlf[0].text.includes("\r"), "CRLF must be normalized so line counts match the citation");
});

test("embedText prefixes the path and bounds its length", () => {
  const c: Chunk = { path: "src/deep/file.ts", start: 1, end: 2, text: "x".repeat(5000) };
  const t = embedText(c);
  assert.ok(t.startsWith("src/deep/file.ts"), "path context is what lets a path-shaped query match");
  assert.equal(t.length, 1500);
});

// ---------------------------------------------------------------------------
// citations
// ---------------------------------------------------------------------------

test("citedIndexes dedupes, sorts, and drops out-of-range markers", () => {
  // Out-of-range matters: a model inventing [9] against 3 sources must not
  // produce a citation that points at nothing.
  assert.deepEqual(citedIndexes("See [2], then [1], then [2] again, and [9] and [0].", 3), [0, 1]);
  assert.deepEqual(citedIndexes("no markers here", 3), []);
});

test("citedIndexes reads consecutive markers, the form the system prompt asks for", () => {
  assert.deepEqual(citedIndexes("Both apply [2][3].", 3), [1, 2]);
  assert.deepEqual(citedIndexes("[1][2][3]", 3), [0, 1, 2]);
});

test("citedIndexes ignores array subscripts in code blocks", () => {
  // A citation is a provenance claim. `chunks[1]` inside an example is not one,
  // and attributing it to source 1 is a false claim in exactly the place this
  // package exists to be exact about.
  assert.deepEqual(citedIndexes("```ts\nconst c = chunks[1];\nvec[0] = 1;\n```", 3), []);
  // ...but a real citation in the same answer still registers.
  assert.deepEqual(citedIndexes("Per [2], write `chunks[1]`.", 3), [1]);
});

test("buildPrompt numbers sources from 1 and round-trips through citedIndexes", () => {
  const chunks: Chunk[] = [
    { path: "a.ts", start: 1, end: 5, text: "first" },
    { path: "b.ts", start: 10, end: 20, text: "second" },
  ];
  const prompt = buildPrompt("what?", "owner/repo", chunks);
  for (const [i, c] of chunks.entries()) {
    assert.ok(prompt.includes(`[${i + 1}] ${c.path} (lines ${c.start}-${c.end})`), `source ${i + 1} header missing`);
    assert.ok(prompt.includes(c.text));
  }
  assert.ok(prompt.includes("owner/repo"));
  assert.ok(prompt.includes("Question: what?"));
  // The marker numbering buildPrompt emits must be the numbering citedIndexes
  // parses — if these ever drift, every citation silently points one off.
  assert.deepEqual(citedIndexes("[1] and [2]", chunks.length), [0, 1]);
});
