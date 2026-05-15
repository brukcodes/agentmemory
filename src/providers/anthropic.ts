import Anthropic from "@anthropic-ai/sdk";
import type { MemoryProvider } from "../types.js";
import { getEnvVar } from "../config.js";

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

export class AnthropicProvider implements MemoryProvider {
  name = "anthropic";
  private client: Anthropic;
  private model: string;
  private maxTokens: number;
  private timeoutMs: number;

  constructor(
    apiKey: string,
    model: string,
    maxTokens: number,
    baseURL?: string,
  ) {
    this.timeoutMs = getLLMTimeout();
    this.client = new Anthropic({
      apiKey,
      timeout: this.timeoutMs,
      ...(baseURL ? { baseURL } : {}),
    });
    this.model = model;
    this.maxTokens = maxTokens;
  }

  async compress(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.call(systemPrompt, userPrompt);
  }

  async summarize(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.call(systemPrompt, userPrompt);
  }

  async describeImage(
    imageData: string,
    mimeType: string,
    prompt: string,
  ): Promise<string> {
    const response = await this.client.messages.create({
      model: this.model,
      max_tokens: this.maxTokens,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "base64",
                media_type: mimeType as
                  | "image/png"
                  | "image/jpeg"
                  | "image/gif"
                  | "image/webp",
                data: imageData,
              },
            },
            { type: "text", text: prompt },
          ],
        },
      ],
    });

    const textBlock = response.content.find((b) => b.type === "text");
    return textBlock?.text ?? "";
  }

  private async call(
    systemPrompt: string,
    userPrompt: string,
  ): Promise<string> {
    const response = await this.client.messages.create({
      model: this.model,
      max_tokens: this.maxTokens,
      system: systemPrompt,
      messages: [{ role: "user", content: userPrompt }],
    });

    const textBlock = response.content.find((b) => b.type === "text");
    return textBlock?.text ?? "";
  }
}
