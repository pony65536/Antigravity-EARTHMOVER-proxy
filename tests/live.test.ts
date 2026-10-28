import assert from "node:assert/strict";
import test from "node:test";
import app from "../src/index";

const liveEnabled =
  process.env.LIVE === "1" && Boolean(process.env.GCP_SERVICE_ACCOUNT);

// Live runs must not print prompts, response bodies, or credentials. The
// assertions below are the only intended output.
if (process.env.LIVE === "1") {
  const noop = () => undefined;
  console.log = noop;
  console.info = noop;
  console.warn = noop;
  console.error = noop;
}

const maxRequests = Number(process.env.LIVE_MAX_REQUESTS || "20");
const maxTokens = Number(process.env.LIVE_MAX_TOKENS || "1024");
const timeoutMs = Number(process.env.LIVE_TIMEOUT_MS || "120000");

let requestCount = 0;

function liveEnv(): Record<string, string> {
  return {
    GCP_SERVICE_ACCOUNT: process.env.GCP_SERVICE_ACCOUNT || "",
    ...(process.env.GEMINI_PROJECT_ID
      ? { GEMINI_PROJECT_ID: process.env.GEMINI_PROJECT_ID }
      : {}),
    ...(process.env.ENABLE_REAL_THINKING
      ? { ENABLE_REAL_THINKING: process.env.ENABLE_REAL_THINKING }
      : { ENABLE_REAL_THINKING: "false" }),
    ...(process.env.STREAM_THINKING_AS_CONTENT
      ? { STREAM_THINKING_AS_CONTENT: process.env.STREAM_THINKING_AS_CONTENT }
      : { STREAM_THINKING_AS_CONTENT: "false" }),
    GEMINI_MODEL_DISCOVERY_TIMEOUT_MS:
      process.env.GEMINI_MODEL_DISCOVERY_TIMEOUT_MS || "60000",
    GEMINI_PROJECT_DISCOVERY_TIMEOUT_MS:
      process.env.GEMINI_PROJECT_DISCOVERY_TIMEOUT_MS || "60000",
    GEMINI_GENERATION_TIMEOUT_MS:
      process.env.GEMINI_GENERATION_TIMEOUT_MS || "180000",
  };
}

async function liveRequest(
  path: string,
  body: Record<string, unknown>,
): Promise<Response> {
  requestCount += 1;
  assert.ok(
    requestCount <= maxRequests,
    `live test request budget exceeded (${maxRequests})`,
  );

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await app.request(
      `http://localhost${path}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      },
      liveEnv(),
    );
  } finally {
    clearTimeout(timer);
  }
}

async function liveGet(path: string): Promise<Response> {
  requestCount += 1;
  assert.ok(
    requestCount <= maxRequests,
    `live test request budget exceeded (${maxRequests})`,
  );

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await app.request(
      `http://localhost${path}`,
      { signal: controller.signal },
      liveEnv(),
    );
  } finally {
    clearTimeout(timer);
  }
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

test(
  "live: normal completion is repeatable",
  { skip: !liveEnabled },
  async () => {
    const requestBody = {
      model: process.env.LIVE_MODEL || "gemini-3.8-flash-tiered",
      messages: [{ role: "user", content: "Reply with the single word: ready" }],
      max_completion_tokens: maxTokens,
      temperature: 0,
    };

    const first = await liveRequest("/v1/chat/completions", requestBody);
    const second = await liveRequest("/v1/chat/completions", requestBody);
    assert.equal(first.status, 200);
    assert.equal(second.status, 200);

    const firstBody = (await first.json()) as Record<string, any>;
    const secondBody = (await second.json()) as Record<string, any>;
    assert.equal(firstBody.object, "chat.completion");
    assert.equal(secondBody.object, "chat.completion");
    assert.equal(typeof firstBody.choices?.[0]?.message?.content, "string");
    assert.equal(typeof secondBody.choices?.[0]?.message?.content, "string");
    assert.ok(firstBody.choices[0].message.content.length > 0);
    assert.ok(secondBody.choices[0].message.content.length > 0);
    assert.equal(
      firstBody.choices?.[0]?.finish_reason,
      secondBody.choices?.[0]?.finish_reason,
    );
    assert.ok(firstBody.usage?.total_tokens > 0);
  },
);

test("live: streaming completion ends with DONE", { skip: !liveEnabled }, async () => {
  const response = await liveRequest("/v1/chat/completions", {
    model: process.env.LIVE_MODEL || "gemini-3.8-flash-tiered",
    messages: [{ role: "user", content: "Count from one to three." }],
    max_completion_tokens: maxTokens,
    temperature: 0,
    stream: true,
    stream_options: { include_usage: true },
  });
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") || "", /text\/event-stream/);
  const events = parseSse(await response.text());
  assert.equal(events.at(-1), "[DONE]");
  const chunks = events.slice(0, -1) as Array<Record<string, any>>;
  const text = chunks
    .map((chunk) => chunk.choices?.[0]?.delta?.content ?? "")
    .join("");
  assert.ok(text.length > 0);
  assert.equal(typeof chunks.at(-1)?.usage?.total_tokens, "number");
});

test("live: json_schema returns a matching object", { skip: !liveEnabled }, async () => {
  const response = await liveRequest("/v1/chat/completions", {
    model: process.env.LIVE_MODEL || "gemini-3.8-flash-tiered",
    messages: [
      {
        role: "user",
        content: "Return a JSON object with answer set to the word ready.",
      },
    ],
    max_completion_tokens: maxTokens,
    temperature: 0,
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "live_answer",
        strict: true,
        schema: {
          type: "object",
          properties: { answer: { type: "string" } },
          required: ["answer"],
          additionalProperties: false,
        },
      },
    },
  });
  assert.equal(response.status, 200);
  const body = (await response.json()) as Record<string, any>;
  const parsed = JSON.parse(body.choices[0].message.content);
  assert.equal(typeof parsed.answer, "string");
});

test("live: json_object mode returns a parseable object", { skip: !liveEnabled }, async () => {
  const response = await liveRequest("/v1/chat/completions", {
    model: process.env.LIVE_MODEL || "gemini-3.8-flash-tiered",
    messages: [
      {
        role: "user",
        content: "Return a JSON object with status set to the word ready.",
      },
    ],
    max_completion_tokens: maxTokens,
    temperature: 0,
    response_format: { type: "json_object" },
  });
  assert.equal(response.status, 200);
  const body = (await response.json()) as Record<string, any>;
  const parsed = JSON.parse(body.choices[0].message.content);
  assert.equal(typeof parsed, "object");
  assert.ok(parsed !== null && !Array.isArray(parsed));
});

test("live: legacy completions works without streaming", { skip: !liveEnabled }, async () => {
  const response = await liveRequest("/v1/completions", {
    model: process.env.LIVE_MODEL || "gemini-3.8-flash-tiered",
    prompt: "Reply with the single word: ready",
    max_tokens: maxTokens,
    temperature: 0,
  });
  assert.equal(response.status, 200);
  const body = (await response.json()) as Record<string, any>;
  assert.equal(body.object, "text_completion");
  assert.equal(body.choices?.[0]?.index, 0);
  assert.equal(typeof body.choices?.[0]?.text, "string");
  assert.ok(body.choices[0].text.length > 0);
  assert.equal(typeof body.usage?.total_tokens, "number");
});

test("live: legacy completions streams text chunks", { skip: !liveEnabled }, async () => {
  const response = await liveRequest("/v1/completions", {
    model: process.env.LIVE_MODEL || "gemini-3.8-flash-tiered",
    prompt: "Count from one to three.",
    max_tokens: maxTokens,
    temperature: 0,
    stream: true,
    stream_options: { include_usage: true },
  });
  assert.equal(response.status, 200);
  const events = parseSse(await response.text());
  assert.equal(events.at(-1), "[DONE]");
  const chunks = events.slice(0, -1) as Array<Record<string, any>>;
  assert.equal(chunks[0]?.object, "text_completion");
  const text = chunks
    .map((chunk) => chunk.choices?.[0]?.text ?? "")
    .join("");
  assert.ok(text.length > 0);
  assert.equal(typeof chunks.at(-1)?.usage?.total_tokens, "number");
});

test("live: model list and model retrieval", { skip: !liveEnabled }, async () => {
  const model = process.env.LIVE_MODEL || "gemini-3.8-flash-tiered";
  const listResponse = await liveGet("/v1/models");
  assert.equal(listResponse.status, 200);
  const listBody = (await listResponse.json()) as Record<string, any>;
  assert.ok(Array.isArray(listBody.data));
  assert.ok(listBody.data.some((entry: any) => entry.id === model));

  const modelResponse = await liveGet(`/v1/models/${encodeURIComponent(model)}`);
  assert.equal(modelResponse.status, 200);
  const modelBody = (await modelResponse.json()) as Record<string, any>;
  assert.equal(modelBody.id, model);
  assert.equal(modelBody.object, "model");
});

test("live: streaming tool call fragments and finish reason", { skip: !liveEnabled }, async () => {
  const tools = [
    {
      type: "function",
      function: {
        name: "lookup_planet",
        description: "Look up a planet by name.",
        strict: true,
        parameters: {
          type: "object",
          properties: { name: { type: "string" } },
          required: ["name"],
          additionalProperties: false,
        },
      },
    },
  ];

  const response = await liveRequest("/v1/chat/completions", {
    model: process.env.LIVE_MODEL || "gemini-3.8-flash-tiered",
    messages: [{ role: "user", content: "Look up the planet Mars." }],
    tools,
    tool_choice: "required",
    max_completion_tokens: maxTokens,
    temperature: 0,
    stream: true,
  });
  assert.equal(response.status, 200);
  const events = parseSse(await response.text());
  assert.equal(events.at(-1), "[DONE]");
  const chunks = events.slice(0, -1) as Array<Record<string, any>>;
  const calls = chunks.flatMap(
    (chunk) => chunk.choices?.[0]?.delta?.tool_calls ?? [],
  );
  assert.ok(calls.length >= 1);
  assert.equal(typeof calls[0].id, "string");
  assert.equal(calls[0].type, "function");
  assert.equal(typeof calls[0].function?.name, "string");
  JSON.parse(calls[0].function.arguments);
  assert.equal(chunks.at(-1)?.choices?.[0]?.finish_reason, "tool_calls");
});

test(
  "live: forced function call and multi-turn tool result",
  { skip: !liveEnabled },
  async () => {
    const tools = [
      {
        type: "function",
        function: {
          name: "lookup_planet",
          description: "Look up a planet by name.",
          strict: true,
          parameters: {
            type: "object",
            properties: { name: { type: "string" } },
            required: ["name"],
            additionalProperties: false,
          },
        },
      },
    ];

    const toolResponse = await liveRequest("/v1/chat/completions", {
      model: process.env.LIVE_MODEL || "gemini-3.8-flash-tiered",
      messages: [{ role: "user", content: "Look up the planet Mars." }],
      tools,
      tool_choice: "required",
      max_completion_tokens: maxTokens,
      temperature: 0,
    });
    assert.equal(toolResponse.status, 200);
    const toolBody = (await toolResponse.json()) as Record<string, any>;
    const toolCalls = toolBody.choices?.[0]?.message?.tool_calls;
    assert.ok(Array.isArray(toolCalls) && toolCalls.length >= 1);
    assert.equal(toolCalls[0].type, "function");
    assert.equal(typeof toolCalls[0].id, "string");
    JSON.parse(toolCalls[0].function.arguments);
    assert.equal(toolBody.choices[0].finish_reason, "tool_calls");

    const finalResponse = await liveRequest("/v1/chat/completions", {
      model: process.env.LIVE_MODEL || "gemini-3.8-flash-tiered",
      messages: [
        { role: "user", content: "Look up the planet Mars." },
        {
          role: "assistant",
          content: null,
          tool_calls: toolCalls,
        },
        {
          role: "tool",
          tool_call_id: toolCalls[0].id,
          content: "Mars is the fourth planet from the Sun.",
        },
        { role: "user", content: "Answer in one short sentence using the tool result." },
      ],
      tools,
      max_completion_tokens: maxTokens,
      temperature: 0,
    });
    assert.equal(finalResponse.status, 200);
    const finalBody = (await finalResponse.json()) as Record<string, any>;
    assert.equal(typeof finalBody.choices?.[0]?.message?.content, "string");
  },
);
