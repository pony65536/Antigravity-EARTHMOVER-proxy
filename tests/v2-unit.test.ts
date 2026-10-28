import assert from "node:assert/strict";
import test from "node:test";
import { GenerationConfigValidator } from "../src/helpers/generation-config-validator";
import {
  convertSchemaForGemini,
  SchemaConversionError,
  selectSchemaProtocol,
  validateStructuredOutputText,
} from "../src/helpers/structured-output";
import {
  mapGeminiFinishReason,
  mapGeminiUsage,
  validateTextCompletionRequest,
  validateChatCompletionRequest,
} from "../src/helpers/openai-compat";
import { redactSensitiveText } from "../src/utils/redact";

test("schema conversion supports enums, nullable, anyOf, arrays, nested objects, and ranges", () => {
  const converted = convertSchemaForGemini(
    {
      type: "object",
      properties: {
        status: { type: "string", enum: ["ok", "error", "skipped"], nullable: true },
        nested: {
          type: "object",
          properties: {
            score: { type: "number", minimum: 0, maximum: 1 },
          },
          required: ["score"],
          additionalProperties: false,
        },
        tags: {
          type: "array",
          items: { type: "string", maxLength: 20 },
          minItems: 1,
          maxItems: 5,
        },
        variant: { anyOf: [{ type: "string" }, { type: "integer" }, { type: "null" }] },
      },
      required: ["status", "nested", "tags"],
      additionalProperties: false,
    },
    "responseSchema",
  );

  assert.equal(converted.type, "OBJECT");
  assert.deepEqual((converted.properties as any).status.enum, ["ok", "error", "skipped"]);
  assert.equal((converted.properties as any).status.nullable, true);
  assert.equal((converted.properties as any).nested.type, "OBJECT");
  assert.equal((converted.properties as any).nested.additionalProperties, undefined);
  assert.equal((converted.properties as any).nested.properties.score.type, "NUMBER");
  assert.equal((converted.properties as any).nested.properties.score.minimum, 0);
  assert.equal((converted.properties as any).tags.items.type, "STRING");
  assert.equal((converted.properties as any).tags.maxItems, 5);
  assert.equal((converted.properties as any).variant.nullable, true);
  assert.deepEqual(
    (converted.properties as any).variant.anyOf.map((branch: any) => branch.type),
    ["STRING", "INTEGER"],
  );
});

test("schema protocol follows model capability and honors explicit override", () => {
  assert.equal(selectSchemaProtocol("gemini-2.5-flash"), "responseSchema");
  assert.equal(selectSchemaProtocol("gemini-3.5-flash-high"), "responseJsonSchema");
  assert.equal(
    selectSchemaProtocol("gemini-2.5-flash", "responseJsonSchema"),
    "responseJsonSchema",
  );

  const config = GenerationConfigValidator.createValidatedConfig(
    "gemini-3.5-flash-high",
    {
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "payload",
          strict: true,
          schema: {
            type: "object",
            properties: { value: { type: "string", nullable: true } },
            required: ["value"],
          },
        },
      },
    },
    false,
    false,
  );
  assert.equal(config.responseSchema, undefined);
  assert.deepEqual(config.responseJsonSchema, {
    type: "object",
    properties: { value: { type: ["string", "null"] } },
    required: ["value"],
  });
});

test("unsupported schema features fail loudly instead of being dropped", () => {
  for (const schema of [
    { type: "object", properties: { a: { $ref: "#/$defs/a" } } },
    { type: "object", additionalProperties: true },
    { type: "object", allOf: [{ type: "object" }] },
    { type: "object", properties: { a: { type: "string", exclusiveMinimum: 1 } } },
  ]) {
    assert.throws(
      () => convertSchemaForGemini(schema, "responseSchema"),
      SchemaConversionError,
    );
  }

  const jsonSchema = convertSchemaForGemini(
    { type: "string", exclusiveMinimum: 1 },
    "responseJsonSchema",
  );
  assert.equal(jsonSchema.exclusiveMinimum, 1);
});

test("structured output validation rejects parse errors and schema violations", () => {
  const schema = {
    type: "object",
    properties: {
      count: { type: "integer", minimum: 1 },
      label: { type: "string" },
    },
    required: ["count", "label"],
    additionalProperties: false,
  };

  assert.equal(validateStructuredOutputText('{"count":2,"label":"ok"}', { schema }).ok, true);
  const fenced = validateStructuredOutputText(
    '```json\n{"count":2,"label":"ok"}\n```',
    { schema },
  );
  assert.equal(fenced.ok, true);
  if (fenced.ok) assert.equal(fenced.text, '{"count":2,"label":"ok"}');

  const nestedFencePayload = {
    count: 2,
    label: "```js\nconsole.log(1)\n```",
  };
  const nestedFence = validateStructuredOutputText(
    `Here is the JSON:\n\`\`\`json\n${JSON.stringify(nestedFencePayload)}\n\`\`\`\nDone`,
    { schema },
  );
  assert.equal(nestedFence.ok, true);
  if (nestedFence.ok) {
    assert.deepEqual(JSON.parse(nestedFence.text), nestedFencePayload);
  }
  assert.equal(validateStructuredOutputText("not json", { schema }).ok, false);
  assert.equal(validateStructuredOutputText('{"count":0,"label":"ok"}', { schema }).ok, false);
  assert.equal(validateStructuredOutputText('{"count":2,"label":"ok","x":1}', { schema }).ok, false);
  assert.equal(validateStructuredOutputText("[1,2]", { requireObject: true }).ok, false);
});

test("legacy completion request validation rejects unsupported parameters", () => {
  const base = { model: "gemini-2.5-flash", prompt: "hello" };
  assert.equal(validateTextCompletionRequest(base), null);
  for (const [key, value] of [
    ["n", 2],
    ["best_of", 2],
    ["logprobs", 1],
    ["echo", true],
    ["suffix", "x"],
  ] as const) {
    assert.equal(
      validateTextCompletionRequest({ ...base, [key]: value })?.code,
      "unsupported_value",
      key,
    );
  }
  assert.equal(
    validateTextCompletionRequest({ ...base, prompt: ["a", "b"] })?.code,
    "unsupported_value",
  );
  assert.equal(
    validateTextCompletionRequest({ ...base, stream_options: { include_usage: true } })?.code,
    "invalid_stream_options",
  );
});

test("tool and multi-turn tool call validation", () => {
  const message = (index: number, extra: Record<string, unknown>) => ({
    role: "user",
    content: `m${index}`,
    ...extra,
  });

  assert.equal(
    validateChatCompletionRequest({
      messages: [
        { role: "user", content: "hi" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            { id: "call_1", type: "function", function: { name: "lookup", arguments: "{}" } },
          ],
        },
        { role: "tool", tool_call_id: "call_2", content: "result" },
      ],
    })?.code,
    "invalid_tool_call_id",
  );

  assert.equal(
    validateChatCompletionRequest({
      messages: [
        { role: "user", content: "hi" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            { id: "call_1", type: "function", function: { name: "lookup", arguments: "{}" } },
          ],
        },
      ],
    })?.code,
    "missing_tool_call_response",
  );

  assert.equal(
    validateChatCompletionRequest({
      messages: [
        { role: "user", content: "hi" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            { id: "call_1", type: "function", function: { name: "lookup", arguments: "{bad" } },
          ],
        },
        { role: "tool", tool_call_id: "call_1", content: "result" },
      ],
    })?.code,
    "invalid_tool_arguments",
  );

  assert.equal(
    validateChatCompletionRequest({
      messages: [
        { role: "user", content: "hi" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            { id: "call_1", type: "function", function: { name: "lookup", arguments: "{}" } },
            { id: "call_2", type: "function", function: { name: "lookup", arguments: "{}" } },
          ],
        },
        { role: "tool", tool_call_id: "call_1", content: "one" },
        { role: "tool", tool_call_id: "call_2", content: "two" },
        { role: "user", content: "thanks" },
      ],
      messages_extra: message(0, {}),
    }),
    null,
  );

  assert.equal(
    validateChatCompletionRequest({
      messages: [{ role: "user", content: "hi" }],
      tools: [
        {
          type: "function",
          function: {
            name: "lookup",
            strict: true,
            parameters: {
              type: "object",
              properties: { q: { type: "string" } },
              required: ["q"],
              additionalProperties: false,
            },
          },
        },
      ],
      tool_choice: { type: "function", function: { name: "missing" } },
    })?.code,
    "invalid_tool_choice",
  );
});

test("finish reason and usage mapping cover safety, malformed tools, cache and reasoning", () => {
  assert.equal(mapGeminiFinishReason("MAX_TOKENS"), "length");
  assert.equal(mapGeminiFinishReason("SAFETY"), "content_filter");
  assert.equal(mapGeminiFinishReason("MALFORMED_FUNCTION_CALL"), "tool_calls");
  assert.equal(mapGeminiFinishReason("UNEXPECTED_TOOL_CALL"), "tool_calls");
  assert.equal(mapGeminiFinishReason("STOP"), "stop");

  assert.deepEqual(
    mapGeminiUsage({
      promptTokenCount: 10,
      candidatesTokenCount: 4,
      thoughtsTokenCount: 3,
      cachedContentTokenCount: 2,
      toolUsePromptTokenCount: 1,
    }),
    {
      prompt_tokens: 10,
      completion_tokens: 7,
      total_tokens: 17,
      completion_tokens_details: { reasoning_tokens: 3, tool_use_tokens: 1 },
      prompt_tokens_details: { cached_tokens: 2, tool_use_tokens: 1 },
    },
  );
});

test("log redaction removes credentials, OAuth codes, and media payloads", () => {
  const redacted = redactSensitiveText(
    JSON.stringify({
      authorization: "Bearer secret-token",
      refresh_token: "refresh-secret",
      code: "oauth-code",
      image_url: { url: "data:image/png;base64,AAAA" },
      input_audio: { data: "A".repeat(500) },
      max_tokens: 10,
    }),
  );

  assert.equal(redacted.includes("secret-token"), false);
  assert.equal(redacted.includes("refresh-secret"), false);
  assert.equal(redacted.includes("oauth-code"), false);
  assert.equal(redacted.includes("A".repeat(500)), false);
  assert.match(redacted, /"max_tokens":10/);
});
