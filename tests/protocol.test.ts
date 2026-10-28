import assert from "node:assert/strict";
import test from "node:test";
import app from "../src/index";
import { clearDynamicModelCaches } from "../src/gemini-client";
import {
  fetchTextWithTimeout,
  fetchWithTimeout,
  RequestTimeoutError,
} from "../src/utils/http";

test.beforeEach(() => {
  clearDynamicModelCaches();
});

interface CapturedRequest {
  url: string;
  init?: RequestInit;
}

type FetchHandler = (
  url: string,
  init?: RequestInit,
) => Response | Promise<Response>;

function installFetchMock(handler: FetchHandler): {
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
    return handler(url, init);
  }) as typeof fetch;
  return {
    requests,
    restore: () => {
      globalThis.fetch = originalFetch;
    },
  };
}

function testEnv(extra: Record<string, string> = {}): Record<string, string> {
  return {
    GCP_SERVICE_ACCOUNT: JSON.stringify({
      access_token: "test-token",
      refresh_token: "test-refresh-token",
      scope: "",
      token_type: "Bearer",
      id_token: "",
      expiry_date: Date.now() + 60 * 60 * 1000,
    }),
    ENABLE_REAL_THINKING: "false",
    STREAM_THINKING_AS_CONTENT: "false",
    ...extra,
  };
}

function data(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

function sseResponse(body: string): Response {
  return new Response(body, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

function fragmentedSseResponse(body: string): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (let index = 0; index < body.length; index += 7) {
        controller.enqueue(encoder.encode(body.slice(index, index + 7)));
      }
      controller.close();
    },
  });
  return new Response(stream, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

function defaultInfrastructureResponse(url: string): Response | undefined {
  if (url.includes(":fetchAvailableModels")) {
    return Response.json({ models: {} });
  }
  if (url.includes(":loadCodeAssist")) {
    return Response.json({ cloudaicompanionProject: "test-project" });
  }
  return undefined;
}

function genResponse(
  parts: unknown[],
  finishReason = "STOP",
  usageMetadata: Record<string, number> = {
    promptTokenCount: 2,
    candidatesTokenCount: 3,
  },
): string {
  return data({
    response: {
      candidates: [{ content: { parts }, finishReason }],
      usageMetadata,
    },
  });
}

function parseSse(text: string): Array<Record<string, any> | "[DONE]"> {
  return text
    .trim()
    .split("\n\n")
    .map((event) => event.replace(/^data: /, ""))
    .map((payload) =>
      payload === "[DONE]"
        ? payload
        : (JSON.parse(payload) as Record<string, any>),
    );
}

test("legacy /v1/completions returns a text_completion object", async () => {
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      return sseResponse(genResponse([{ text: "Hello world" }]));
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const response = await app.request(
      "http://localhost/v1/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          prompt: "Say hello",
          max_tokens: 16,
        }),
      },
      testEnv(),
    );

    assert.equal(response.status, 200);
    const body = (await response.json()) as Record<string, any>;
    assert.equal(body.object, "text_completion");
    assert.equal(body.choices[0].text, "Hello world");
    assert.equal(body.choices[0].index, 0);
    assert.equal(body.choices[0].finish_reason, "stop");
    assert.deepEqual(body.usage, {
      prompt_tokens: 2,
      completion_tokens: 3,
      total_tokens: 5,
    });

    const upstream = mock.requests.find((request) =>
      request.url.includes(":streamGenerateContent"),
    );
    const payload = JSON.parse(String(upstream?.init?.body));
    assert.deepEqual(payload.request.contents, [
      { role: "user", parts: [{ text: "Say hello" }] },
    ]);
  } finally {
    mock.restore();
  }
});

test("legacy /v1/completions streams text_completion chunks", async () => {
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      return fragmentedSseResponse(
        genResponse([{ text: "Hello" }]) + genResponse([{ text: " world" }]),
      );
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const response = await app.request(
      "http://localhost/v1/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          prompt: "Say hello",
          stream: true,
          stream_options: { include_usage: true },
        }),
      },
      testEnv(),
    );
    assert.equal(response.status, 200);
    const events = parseSse(await response.text());
    assert.equal(events.at(-1), "[DONE]");
    const chunks = events.slice(0, -1) as Array<Record<string, any>>;
    assert.equal(chunks[0].object, "text_completion");
    const text = chunks
      .map((chunk) => chunk.choices?.[0]?.text ?? "")
      .join("");
    assert.equal(text, "Hello world");
    assert.equal(chunks.at(-2)?.choices[0].finish_reason, "stop");
    assert.deepEqual(chunks.at(-1)?.usage, {
      prompt_tokens: 2,
      completion_tokens: 3,
      total_tokens: 5,
    });
  } finally {
    mock.restore();
  }
});

test("parallel tool calls preserve Gemini IDs and emit sequential indices", async () => {
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      return sseResponse(
        genResponse([
          { functionCall: { id: "call_a", name: "lookup", args: { q: "one" } } },
          { functionCall: { id: "call_b", name: "search", args: { q: "two" } } },
        ]),
      );
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const response = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [{ role: "user", content: "use tools" }],
          stream: true,
        }),
      },
      testEnv(),
    );
    assert.equal(response.status, 200);
    const events = parseSse(await response.text()).slice(0, -1) as Array<
      Record<string, any>
    >;
    const calls = events.flatMap(
      (event) => event.choices?.[0]?.delta?.tool_calls ?? [],
    );
    assert.equal(calls.length, 2);
    assert.equal(calls[0].index, 0);
    assert.equal(calls[0].id, "call_a");
    assert.equal(calls[1].index, 1);
    assert.equal(calls[1].id, "call_b");
    assert.equal(events.at(-1)?.choices[0].finish_reason, "tool_calls");
  } finally {
    mock.restore();
  }
});

test("multi-turn tool messages map back to Gemini function responses", async () => {
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      return sseResponse(genResponse([{ text: "Final answer" }]));
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const response = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [
            { role: "user", content: "Look it up" },
            {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: { name: "lookup", arguments: '{"q":"earth"}' },
                },
              ],
            },
            { role: "tool", tool_call_id: "call_1", content: "planet" },
          ],
        }),
      },
      testEnv(),
    );
    assert.equal(response.status, 200);
    const body = (await response.json()) as Record<string, any>;
    assert.equal(body.choices[0].message.content, "Final answer");

    const upstream = mock.requests.find((request) =>
      request.url.includes(":streamGenerateContent"),
    );
    const payload = JSON.parse(String(upstream?.init?.body));
    const parts = payload.request.contents.flatMap((message: any) => message.parts);
    assert.ok(parts.some((part: any) => part.functionCall?.name === "lookup"));
    assert.ok(
      parts.some(
        (part: any) =>
          part.functionResponse?.name === "lookup" &&
          part.functionResponse?.id === "call_1" &&
          part.functionResponse?.response?.result === "planet",
      ),
    );
  } finally {
    mock.restore();
  }
});

test("401 refreshes credentials exactly once and retries the generation", async () => {
  let generateCalls = 0;
  let refreshCalls = 0;
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes("oauth2.googleapis.com/token")) {
      refreshCalls += 1;
      return Response.json({ access_token: "refreshed-token", expires_in: 3600 });
    }
    if (url.includes(":streamGenerateContent")) {
      generateCalls += 1;
      if (generateCalls === 1) {
        return new Response("unauthorized", { status: 401 });
      }
      return sseResponse(genResponse([{ text: "Recovered" }]));
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const response = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [{ role: "user", content: "hi" }],
        }),
      },
      testEnv(),
    );
    assert.equal(response.status, 200);
    assert.equal(refreshCalls, 1);
    assert.equal(generateCalls, 2);
    const generateRequest = mock.requests.filter((request) =>
      request.url.includes(":streamGenerateContent"),
    )[1];
    assert.equal(
      new Headers(generateRequest.init?.headers).get("authorization"),
      "Bearer refreshed-token",
    );
  } finally {
    mock.restore();
  }
});

test("persistent 401 stops after one refresh and preserves the upstream status", async () => {
  let generateCalls = 0;
  let refreshCalls = 0;
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes("oauth2.googleapis.com/token")) {
      refreshCalls += 1;
      return Response.json({ access_token: "refreshed-token", expires_in: 3600 });
    }
    if (url.includes(":streamGenerateContent")) {
      generateCalls += 1;
      return new Response("still unauthorized", { status: 401 });
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const response = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [{ role: "user", content: "hi" }],
        }),
      },
      testEnv(),
    );
    assert.equal(response.status, 401);
    assert.equal(refreshCalls, 1);
    assert.equal(generateCalls, 2);
  } finally {
    mock.restore();
  }
});

test("429 honors Retry-After and retries a bounded number of times", async () => {
  let generateCalls = 0;
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      generateCalls += 1;
      if (generateCalls === 1) {
        return new Response("slow down", {
          status: 429,
          headers: { "Retry-After": "0" },
        });
      }
      return sseResponse(genResponse([{ text: "After retry" }]));
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const response = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [{ role: "user", content: "hi" }],
        }),
      },
      testEnv(),
    );
    assert.equal(response.status, 200);
    assert.equal(generateCalls, 2);
  } finally {
    mock.restore();
  }
});

test("upstream HTTP status and request ID are preserved in the error envelope", async () => {
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      return new Response('{"error":{"message":"upstream exploded"}}', {
        status: 503,
        headers: {
          "Content-Type": "application/json",
          "x-request-id": "req-abc-123",
        },
      });
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const response = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [{ role: "user", content: "hi" }],
        }),
      },
      testEnv(),
    );
    assert.equal(response.status, 503);
    assert.equal(response.headers.get("x-request-id"), "req-abc-123");
    const body = (await response.json()) as Record<string, any>;
    assert.equal(body.error.request_id, "req-abc-123");
    assert.equal(body.error.type, "api_error");
  } finally {
    mock.restore();
  }
});

test("mid-stream upstream errors become a stable SSE error event", async () => {
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      return sseResponse(
        genResponse([{ text: "partial" }]) +
          data({ error: { code: 429, status: "RESOURCE_EXHAUSTED", message: "quota" } }),
      );
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const response = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [{ role: "user", content: "hi" }],
          stream: true,
        }),
      },
      testEnv(),
    );
    assert.equal(response.status, 200);
    const events = parseSse(await response.text());
    const errorEvent = events.find(
      (event) => event !== "[DONE]" && "error" in event,
    ) as Record<string, any> | undefined;
    assert.ok(errorEvent);
    assert.equal(errorEvent.error.message, "quota");
    assert.equal(errorEvent.error.type, "rate_limit_error");
    assert.equal(events.at(-1), "[DONE]");
  } finally {
    mock.restore();
  }
});

test("malformed upstream SSE is reported instead of silently truncating", async () => {
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      return sseResponse("data: {not-json\n\n");
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const response = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [{ role: "user", content: "hi" }],
        }),
      },
      testEnv(),
    );
    assert.equal(response.status, 502);
    const body = (await response.json()) as Record<string, any>;
    assert.equal(body.error.code, "malformed_sse");
  } finally {
    mock.restore();
  }
});

test("structured output is validated before returning a chat completion", async () => {
  let mode: "invalid" | "valid" = "invalid";
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      return sseResponse(
        genResponse([{ text: mode === "invalid" ? "not json" : '{"value":42}' }]),
      );
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  const requestBody = JSON.stringify({
    model: "gemini-2.5-flash",
    messages: [{ role: "user", content: "return json" }],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "value_payload",
        strict: true,
        schema: {
          type: "object",
          properties: { value: { type: "integer", minimum: 1 } },
          required: ["value"],
          additionalProperties: false,
        },
      },
    },
  });

  try {
    const invalidResponse = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestBody,
      },
      testEnv(),
    );
    assert.equal(invalidResponse.status, 502);
    const invalidBody = (await invalidResponse.json()) as Record<string, any>;
    assert.equal(invalidBody.error.code, "invalid_structured_output");

    mode = "valid";
    const validResponse = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestBody,
      },
      testEnv(),
    );
    assert.equal(validResponse.status, 200);
    const validBody = (await validResponse.json()) as Record<string, any>;
    assert.deepEqual(JSON.parse(validBody.choices[0].message.content), { value: 42 });
  } finally {
    mock.restore();
  }
});

test("streaming structured output is validated as a whole at the end", async () => {
  let mode: "invalid" | "valid" = "invalid";
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      return sseResponse(
        genResponse([
          { text: '{"value":' },
          { text: mode === "invalid" ? "}" : "42}" },
        ]),
      );
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  const requestBody = JSON.stringify({
    model: "gemini-2.5-flash",
    messages: [{ role: "user", content: "return json" }],
    stream: true,
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "value_payload",
        strict: true,
        schema: {
          type: "object",
          properties: { value: { type: "integer", minimum: 10 } },
          required: ["value"],
          additionalProperties: false,
        },
      },
    },
  });

  try {
    const invalidResponse = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestBody,
      },
      testEnv(),
    );
    assert.equal(invalidResponse.status, 200);
    const invalidEvents = parseSse(await invalidResponse.text());
    const invalidError = invalidEvents.find(
      (event) => event !== "[DONE]" && "error" in event,
    ) as Record<string, any> | undefined;
    assert.ok(invalidError);
    assert.equal(invalidError.error.code, "invalid_structured_output");

    mode = "valid";
    const validResponse = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestBody,
      },
      testEnv(),
    );
    const validEvents = parseSse(await validResponse.text());
    const content = (validEvents.slice(0, -1) as Array<Record<string, any>>)
      .map((event) => event.choices?.[0]?.delta?.content ?? "")
      .join("");
    assert.deepEqual(JSON.parse(content), { value: 42 });
  } finally {
    mock.restore();
  }
});

test("non-streaming responses never mix thinking_content into content", async () => {
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      return sseResponse(
        genResponse([
          { text: "<think>secret reasoning</think>Final answer" },
        ]),
      );
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const response = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [{ role: "user", content: "hi" }],
        }),
      },
      testEnv({
        ENABLE_REAL_THINKING: "true",
        STREAM_THINKING_AS_CONTENT: "true",
      }),
    );
    assert.equal(response.status, 200);
    const body = (await response.json()) as Record<string, any>;
    assert.equal(body.choices[0].message.content, "Final answer");
    assert.equal(body.choices[0].message.content.includes("<think>"), false);
    assert.equal(body.choices[0].message.reasoning_content, "secret reasoning");
    assert.equal(body.choices[0].message.reasoning, "secret reasoning");
  } finally {
    mock.restore();
  }
});

test("prompt-feedback safety blocks map to content_filter", async () => {
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      return sseResponse(
        data({
          response: {
            promptFeedback: { blockReason: "SAFETY" },
            usageMetadata: { promptTokenCount: 3 },
          },
        }),
      );
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const response = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [{ role: "user", content: "blocked" }],
        }),
      },
      testEnv(),
    );
    assert.equal(response.status, 200);
    const body = (await response.json()) as Record<string, any>;
    assert.equal(body.choices[0].finish_reason, "content_filter");
  } finally {
    mock.restore();
  }
});

test("usage includes reasoning, cached, and tool-use token details", async () => {
  const mock = installFetchMock((url) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      return sseResponse(
        genResponse([{ text: "ok" }], "STOP", {
          promptTokenCount: 10,
          candidatesTokenCount: 4,
          thoughtsTokenCount: 3,
          cachedContentTokenCount: 2,
          toolUsePromptTokenCount: 1,
        }),
      );
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const response = await app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [{ role: "user", content: "hi" }],
        }),
      },
      testEnv(),
    );
    const body = (await response.json()) as Record<string, any>;
    assert.deepEqual(body.usage, {
      prompt_tokens: 10,
      completion_tokens: 7,
      total_tokens: 17,
      completion_tokens_details: { reasoning_tokens: 3, tool_use_tokens: 1 },
      prompt_tokens_details: { cached_tokens: 2, tool_use_tokens: 1 },
    });
  } finally {
    mock.restore();
  }
});

test("GET /v1/models/:model resolves static and dynamic models with OpenAI errors", async () => {
  const mock = installFetchMock((url) => {
    if (url.includes(":fetchAvailableModels")) {
      return Response.json({
        models: {
          "gemini-dynamic": { displayName: "Dynamic Model", isInternal: false },
        },
      });
    }
    if (url.includes(":loadCodeAssist")) {
      return Response.json({ cloudaicompanionProject: "test-project" });
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  try {
    const staticResponse = await app.request(
      "http://localhost/v1/models/gemini-2.5-flash",
      undefined,
      testEnv(),
    );
    assert.equal(staticResponse.status, 200);

    const dynamicResponse = await app.request(
      `http://localhost/v1/models/${encodeURIComponent("Dynamic Model")}`,
      undefined,
      testEnv(),
    );
    assert.equal(dynamicResponse.status, 200);
    const dynamicBody = (await dynamicResponse.json()) as Record<string, any>;
    assert.equal(dynamicBody.id, "Dynamic Model");

    const missingResponse = await app.request(
      "http://localhost/v1/models/not-a-model",
      undefined,
      testEnv(),
    );
    assert.equal(missingResponse.status, 404);
    const missingBody = (await missingResponse.json()) as Record<string, any>;
    assert.equal(missingBody.error.code, "model_not_found");
  } finally {
    mock.restore();
  }
});

test("client cancellation aborts the upstream generation stream", async () => {
  let upstreamAborted = false;
  const mock = installFetchMock((url, init) => {
    const infrastructure = defaultInfrastructureResponse(url);
    if (infrastructure) return infrastructure;
    if (url.includes(":streamGenerateContent")) {
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          init?.signal?.addEventListener("abort", () => {
            upstreamAborted = true;
            const error = new Error("aborted");
            error.name = "AbortError";
            controller.error(error);
          });
        },
      });
      return new Response(stream, {
        headers: { "Content-Type": "text/event-stream" },
      });
    }
    throw new Error(`Unexpected URL ${url}`);
  });

  const controller = new AbortController();
  try {
    const request = app.request(
      "http://localhost/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemini-2.5-flash",
          messages: [{ role: "user", content: "hi" }],
          stream: true,
        }),
        signal: controller.signal,
      },
      testEnv(),
    );
    setTimeout(() => controller.abort(), 20);
    const response = await request;
    assert.equal(response.status, 499);
    assert.equal(upstreamAborted, true);
  } finally {
    mock.restore();
  }
});

test("fetchWithTimeout enforces timeouts and honors caller cancellation", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = ((_input: RequestInfo | URL, init?: RequestInit) =>
    new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => {
        const error = new Error("aborted");
        error.name = "AbortError";
        reject(error);
      });
    })) as typeof fetch;

  try {
    await assert.rejects(
      fetchWithTimeout("http://example.invalid", {}, 20),
      RequestTimeoutError,
    );

    const controller = new AbortController();
    const promise = fetchWithTimeout("http://example.invalid", {}, 1000, controller.signal);
    controller.abort();
    await assert.rejects(promise);
    assert.equal(controller.signal.aborted, true);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("fetchTextWithTimeout keeps the timeout active while reading the body", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = ((_input: RequestInfo | URL, init?: RequestInit) => {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        init?.signal?.addEventListener("abort", () => {
          const error = new Error("aborted");
          error.name = "AbortError";
          controller.error(error);
        });
      },
    });
    return Promise.resolve(new Response(stream));
  }) as typeof fetch;

  try {
    await assert.rejects(
      fetchTextWithTimeout("http://example.invalid", {}, 20),
      RequestTimeoutError,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
