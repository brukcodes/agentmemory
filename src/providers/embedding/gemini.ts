import type { EmbeddingProvider } from "../../types.js";
import { getEnvVar } from "../../config.js";

const BATCH_LIMIT = 100;
const MODEL = "models/gemini-embedding-001";
const API_BASE = `https://generativelanguage.googleapis.com/v1beta/${MODEL}:batchEmbedContents`;
const DEFAULT_TIMEOUT_MS = 30_000;

function getLLMTimeout(): number {
  const raw = getEnvVar("AGENTMEMORY_LLM_TIMEOUT_MS");
  if (!raw) return DEFAULT_TIMEOUT_MS;
  const parsed = parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    process.stderr.write(
      `[agentmemory] warn: AGENTMEMORY_LLM_TIMEOUT_MS="${raw}" is invalid; ` +
        `falling back to default ${DEFAULT_TIMEOUT_MS}ms.\n`,
    );
    return DEFAULT_TIMEOUT_MS;
  }
  return parsed;
}

export class GeminiEmbeddingProvider implements EmbeddingProvider {
  readonly name = "gemini";
  readonly dimensions = 768;
  private apiKey: string;

  constructor(apiKey?: string) {
    this.apiKey = apiKey || getEnvVar("GEMINI_API_KEY") || "";
    if (!this.apiKey) throw new Error("GEMINI_API_KEY is required");
  }

  async embed(text: string): Promise<Float32Array> {
    const [result] = await this.embedBatch([text]);
    return result;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    const results: Float32Array[] = [];
    const timeoutMs = getLLMTimeout();

    for (let i = 0; i < texts.length; i += BATCH_LIMIT) {
      const chunk = texts.slice(i, i + BATCH_LIMIT);
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);

      let response: Response;
      try {
        response = await fetch(`${API_BASE}?key=${this.apiKey}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            requests: chunk.map((t) => ({
              model: MODEL,
              content: { parts: [{ text: t }] },
              outputDimensionality: this.dimensions,
            })),
          }),
          signal: controller.signal,
        });
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") {
          throw new Error(
            `Gemini embedding request timed out after ${timeoutMs}ms. ` +
              `Increase AGENTMEMORY_LLM_TIMEOUT_MS to allow more time.`,
          );
        }
        throw err;
      } finally {
        clearTimeout(timer);
      }

      if (!response.ok) {
        const err = await response.text();
        throw new Error(`Gemini embedding failed (${response.status}): ${err}`);
      }

      const data = (await response.json()) as {
        embeddings: Array<{ values: number[] }>;
      };

      for (const emb of data.embeddings) {
        results.push(l2Normalize(new Float32Array(emb.values)));
      }
    }

    return results;
  }
}

let zeroNormWarned = false;

function l2Normalize(vec: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i]! * vec[i]!;
  const norm = Math.sqrt(sum);
  if (norm === 0) {
    if (!zeroNormWarned) {
      zeroNormWarned = true;
      process.stderr.write(
        `[agentmemory] warn: gemini-embedding-001 returned a zero-norm ` +
          `embedding (length=${vec.length}); leaving it un-normalized. ` +
          `Subsequent zero-norm vectors will not be reported.\n`,
      );
    }
    return vec;
  }
  for (let i = 0; i < vec.length; i++) vec[i] = vec[i]! / norm;
  return vec;
}
