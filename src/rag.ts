/**
 * RAG prompt construction, ported from edgeorgie/repoask (lib/rag.ts).
 * Builds numbered source excerpts with inline [n] citation markers.
 */

import type { Chunk } from "./chunk.js";

export const SYSTEM_PROMPT = `You answer questions about a GitHub repository using only the numbered source excerpts provided.
Rules:
- Ground every claim in the sources and cite them inline like [1] or [2][3].
- If the sources do not contain the answer, say so plainly and suggest which files to look at. Do not invent code or APIs.
- Be concise. Use short paragraphs and code blocks when useful.`;

export function buildPrompt(question: string, repo: string, chunks: Chunk[]): string {
  const sources = chunks
    .map((c, i) => `[${i + 1}] ${c.path} (lines ${c.start}-${c.end})\n${c.text}`)
    .join("\n\n---\n\n");
  return `Repository: ${repo}\n\nSources:\n${sources}\n\nQuestion: ${question}`;
}

/**
 * Extracts the zero-based source indexes an answer actually cites, from inline
 * `[n]` markers, deduplicated and sorted ascending. Markers outside `1..max` are
 * dropped, so a model that invents `[9]` against 3 sources cannot produce a
 * citation pointing at nothing.
 *
 * A marker is only read as a citation when it is NOT a subscript — `[n]`
 * immediately preceded by a word character is array/index syntax from a code
 * block (`chunks[1]`, `vec[0]`), not a citation. Without that guard an answer
 * containing a code example attributed a citation to whichever source the
 * subscript happened to number, which is a false provenance claim in the one
 * place this package exists to be exact about.
 *
 * `]` is deliberately NOT treated as a subscript prefix: `[2][3]` is the
 * consecutive-citation form `SYSTEM_PROMPT` explicitly asks models to produce,
 * so excluding a `]`-prefixed marker would silently drop every second citation
 * in a run. A lookbehind (rather than a consuming `(^|[^\w])` group) is required
 * for the same reason: a consuming prefix eats the `]` of `[2]` and makes the
 * adjacent `[3]` unmatchable.
 */
export function citedIndexes(answer: string, max: number): number[] {
  const found = new Set<number>();
  for (const m of answer.matchAll(/(?<!\w)\[(\d+)\]/g)) {
    const n = Number(m[1]);
    if (n >= 1 && n <= max) found.add(n - 1);
  }
  return [...found].sort((a, b) => a - b);
}
