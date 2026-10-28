import assert from "node:assert/strict";
import test from "node:test";
import app from "../src/index";
import { GenerationConfigValidator } from "../src/helpers/generation-config-validator";
import {
  mapGeminiFinishReason,
  validateChatCompletionRequest,
} from "../src/helpers/openai-compat";
import { createOpenAIStreamTransformer } from "../src/stream-transformer";
import { StreamChunk } from "../src/types";

interface CapturedRequest {
  url: string;
  init?: RequestInit;
}

function upstreamSse(finishReason = "STOP", text = "Hello"): string {
  return `data: ${JSON.stringify({
    response: {
      candidates: [{ content: { parts: [{ text }] }, finishReason }],
      usageMetadata: { promptTokenCount: 2, candidatesTokenCount: 3 },
    },
  })}\n\n`;
}

function installUpstreamMock(finishReason = "STOP", text = "Hello"): {
  requests: CapturedRequest[];
  restore: () => void;
} {
  const originalFetch = globalThis.fetch;
  const requests: CapturedRequest[] = [];

  globalThis.fetch = (async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    const url =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    requests.push({ url, init });

    if (url.includes(":fetchAvailableModels")) {
      return Response.json({ models: {} });
    }
    if (url.includes(":loadCodeAssist")) {
      return Response.json({ cloudaicompanionProject: "test-project" });
    }
    if (url.includes(":streamGenerateContent")) {
      return new Response(upstreamSse(finishReason, text), {
        headers: { "Content-Type": "text/event-stream" },
      });
    }
    throw new Error(`Unexpected upstream URL: ${url}`);
  }) as typeof fetch;

  return {
    requests,
    restore: () => {
      globalThis.fetch = originalFetch;
    },
  };
}

function testEnv(): Record<string, string> {
  return {
    GCP_SERVICE_ACCOUNT: JSON.stringify({
      access_token: "test-token",
      refresh_token: "test-refresh-token",
      expiry_date: Date.now() + 60 * 60 * 1000,
    }),
    ENABLE_REAL_THINKING: "false",
    STREAM_THINKING_AS_CONTENT: "false",
  };
}

function parseSse(text: string): Array<Record<string, unknown> | "[DONE]"> {
  return text
    .trim()
    .split("\n\n")
    .map((event) => event.replace(/^data: /, ""))
    .map((data) =>
      data === "[DONE]" ? data : (JSON.parse(data) as Record<string, unknown>),
    );
}

test("chat completions defaults to a non-streaming JSON response and forwards generation parameters", async () => {
  const mock = installUpstreamMock("MAX_TOKENS", '{"ok":true}');
  try {
    const response = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [
            { role: "developer", content: "Follow the contract." },
            { role: "user", content: "Hello" },
          ],
          max_completion_tokens: 64,
          temperature: 0.2,
          top_p: 0.8,
          stop: ["END"],
          presence_penalty: 0.1,
          frequency_penalty: -0.1,
          seed: 7,
          response_format: { type: "json_object" },
        }),
      },
      testEnv(),
    );

    assert.equal(response.status, 200);
    assert.match(
      response.headers.get("content-type") || "",
      /application\/json/,
    );
    const body = (await response.json()) as Record<string, any>;
    assert.equal(body.object, "chat.completion");
    assert.deepEqual(JSON.parse(body.choices[0].message.content), { ok: true });
    assert.equal(body.choices[0].finish_reason, "length");
    assert.deepEqual(body.usage, {
      prompt_tokens: 2,
      completion_tokens: 3,
      total_tokens: 5,
    });

    const upstream = mock.requests.find((request) =>
      request.url.includes(":streamGenerateContent"),
    );
    assert.ok(upstream?.init?.body);
    const payload = JSON.parse(String(upstream.init.body));
    assert.deepEqual(payload.request.systemInstruction.parts, [
      { text: "Follow the contract." },
    ]);
    assert.deepEqual(payload.request.generationConfig, {
      temperature: 0.2,
      maxOutputTokens: 64,
      topP: 0.8,
      stopSequences: ["END"],
      presencePenalty: 0.1,
      frequencyPenalty: -0.1,
      seed: 7,
      responseMimeType: "application/json",
      responseSchema: { type: "OBJECT" },
    });
  } finally {
    mock.restore();
  }
});

test("streaming emits role, finish, an optional standalone usage chunk, and DONE", async () => {
  const mock = installUpstreamMock("MAX_TOKENS");
  try {
    const response = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [{ role: "user", content: "Hello" }],
          stream: true,
          stream_options: { include_usage: true },
        }),
      },
      testEnv(),
    );

    assert.equal(response.status, 200);
    assert.match(
      response.headers.get("content-type") || "",
      /text\/event-stream/,
    );
    const events = parseSse(await response.text());
    assert.equal(events.at(-1), "[DONE]");

    const chunks = events.slice(0, -1) as Array<Record<string, any>>;
    assert.deepEqual(chunks[0].choices[0].delta, {
      role: "assistant",
      content: "",
    });
    assert.equal(chunks.at(-2)?.choices[0].finish_reason, "length");
    assert.deepEqual(chunks.at(-1)?.choices, []);
    assert.deepEqual(chunks.at(-1)?.usage, {
      prompt_tokens: 2,
      completion_tokens: 3,
      total_tokens: 5,
    });
  } finally {
    mock.restore();
  }
});

test("stream transformer omits usage when it was not requested", async () => {
  const chunks: StreamChunk[] = [
    { type: "text", data: "ok" },
    { type: "usage", data: { inputTokens: 1, outputTokens: 1 } },
    { type: "finish", data: { reason: "stop" } },
  ];
  const readable = new ReadableStream<StreamChunk>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
  const output = await new Response(
    readable.pipeThrough(createOpenAIStreamTransformer("test-model")),
  ).text();
  const events = parseSse(output).slice(0, -1) as Array<
    Record<string, unknown>
  >;
  assert.equal(
    events.some((event) => "usage" in event),
    false,
  );
});

test("stream transformer emits OpenAI tool calls and a tool_calls finish reason", async () => {
  const readable = new ReadableStream<StreamChunk>({
    start(controller) {
      controller.enqueue({
        type: "tool_code",
        data: { name: "lookup", args: { query: "earth" } },
      });
      controller.enqueue({ type: "finish", data: { reason: "stop" } });
      controller.close();
    },
  });
  const output = await new Response(
    readable.pipeThrough(createOpenAIStreamTransformer("test-model")),
  ).text();
  const events = parseSse(output).slice(0, -1) as Array<Record<string, any>>;
  const toolCall = events[1].choices[0].delta.tool_calls[0];
  assert.equal(toolCall.index, 0);
  assert.equal(toolCall.type, "function");
  assert.equal(toolCall.function.name, "lookup");
  assert.equal(toolCall.function.arguments, JSON.stringify({ query: "earth" }));
  assert.equal(events.at(-1)?.choices[0].finish_reason, "tool_calls");
});

test("stream transformer dual-writes reasoning and reasoning_content", async () => {
  const readable = new ReadableStream<StreamChunk>({
    start(controller) {
      controller.enqueue({ type: "real_thinking", data: "thought" });
      controller.enqueue({ type: "text", data: "answer" });
      controller.enqueue({ type: "finish", data: { reason: "stop" } });
      controller.close();
    },
  });
  const output = await new Response(
    readable.pipeThrough(createOpenAIStreamTransformer("test-model")),
  ).text();
  const events = parseSse(output).slice(0, -1) as Array<Record<string, any>>;
  const reasoningEvent = events.find(
    (event) => event.choices?.[0]?.delta?.reasoning,
  );
  assert.ok(reasoningEvent);
  assert.equal(reasoningEvent.choices[0].delta.reasoning, "thought");
  assert.equal(
    reasoningEvent.choices[0].delta.reasoning_content,
    "thought",
  );
  const content = events
    .map((event) => event.choices?.[0]?.delta?.content ?? "")
    .join("");
  assert.equal(content, "answer");
});

test("invalid JSON returns a standard OpenAI error envelope", async () => {
  const response = await app.request(
    "http://localhost/v1/chat/completions",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{",
    },
    testEnv(),
  );
  assert.equal(response.status, 400);
  assert.deepEqual(await response.json(), {
    error: {
      message: "Invalid JSON request body",
      type: "invalid_request_error",
      param: null,
      code: "invalid_json",
    },
  });
});

test("request validation and compatibility mappings reject unsupported shapes", () => {
  assert.equal(
    validateChatCompletionRequest({
      messages: [{ role: "user", content: "hi" }],
      stream_options: {},
    })?.code,
    "invalid_stream_options",
  );
  assert.equal(
    validateChatCompletionRequest({
      messages: [{ role: "user", content: "hi" }],
      n: 2,
    })?.code,
    "unsupported_value",
  );
  assert.equal(
    validateChatCompletionRequest({
      messages: [{ role: "user", content: "hi" }],
      tool_choice: "required",
    })?.code,
    "invalid_tool_choice",
  );
  assert.equal(mapGeminiFinishReason("MAX_TOKENS"), "length");
  assert.equal(mapGeminiFinishReason("SAFETY"), "content_filter");
  assert.equal(mapGeminiFinishReason("STOP"), "stop");
});

test("required tool choice maps to Gemini ANY mode", () => {
  const config = GenerationConfigValidator.createValidateTools({
    tools: [
      {
        type: "function",
        function: { name: "lookup", parameters: { type: "object" } },
      },
    ],
    tool_choice: "required",
  });
  assert.deepEqual(config.toolConfig, {
    functionCallingConfig: { mode: "ANY" },
  });
});

test("OpenAI json_schema maps to Gemini structured output configuration", () => {
  const schema = {
    type: "object",
    properties: { status: { type: "string" } },
    required: ["status"],
  };
  const config = GenerationConfigValidator.createValidatedConfig(
    "gemini-2.5-flash",
    {
      response_format: {
        type: "json_schema",
        json_schema: { name: "status_response", strict: true, schema },
      },
    },
    false,
    false,
  );

  assert.equal(config.responseMimeType, "application/json");
  assert.deepEqual(config.responseSchema, {
    type: "OBJECT",
    properties: { status: { type: "STRING" } },
    required: ["status"],
  });
  assert.equal(
    validateChatCompletionRequest({
      messages: [{ role: "user", content: "hi" }],
      response_format: {
        type: "json_schema",
        json_schema: { name: "invalid name", schema },
      },
    })?.code,
    "invalid_json_schema",
  );
});
