import {
  ChatCompletionFinishReason,
  ChatCompletionRequest,
  ChatCompletionUsage,
} from "../types";
import { isAbortError } from "../utils/http";
import { SchemaConversionError, convertSchemaForGemini, selectSchemaProtocol } from "./structured-output";

export interface ValidationIssue {
  message: string;
  param?: string;
  code: string;
}

export interface OpenAIErrorBody {
  error: {
    message: string;
    type: string;
    param: string | null;
    code: string | number | null;
    request_id?: string | null;
  };
}

export class UpstreamApiError extends Error {
  public readonly code?: string | number | null;
  public readonly requestId?: string;
  public readonly retryAfterMs?: number;

  constructor(
    public readonly status: number,
    message: string,
    public readonly upstreamBody?: string,
    options: {
      code?: string | number | null;
      requestId?: string;
      retryAfterMs?: number;
    } = {},
  ) {
    super(message);
    this.name = "UpstreamApiError";
    this.code = options.code;
    this.requestId = options.requestId;
    this.retryAfterMs = options.retryAfterMs;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateOptionalNumber(
  body: Record<string, unknown>,
  key: string,
  minimum: number,
  maximum?: number,
  integer = false,
): ValidationIssue | null {
  const value = body[key];
  if (value === undefined || value === null) return null;
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    (integer && !Number.isInteger(value))
  ) {
    return {
      message: `${key} must be ${integer ? "an integer" : "a number"}`,
      param: key,
      code: "invalid_type",
    };
  }
  if (value < minimum || (maximum !== undefined && value > maximum)) {
    return {
      message: `${key} must be between ${minimum}${maximum === undefined ? " and infinity" : ` and ${maximum}`}`,
      param: key,
      code: "invalid_value",
    };
  }
  return null;
}

function validateStreamOptions(
  body: Record<string, unknown>,
  stream: boolean,
): ValidationIssue | null {
  if (body.stream_options === undefined || body.stream_options === null) return null;
  if (!isRecord(body.stream_options)) {
    return {
      message: "stream_options must be an object",
      param: "stream_options",
      code: "invalid_type",
    };
  }
  if (!stream) {
    return {
      message: "stream_options may only be set when stream is true",
      param: "stream_options",
      code: "invalid_stream_options",
    };
  }
  for (const key of ["include_usage", "include_obfuscation"]) {
    if (
      body.stream_options[key] !== undefined &&
      typeof body.stream_options[key] !== "boolean"
    ) {
      return {
        message: `stream_options.${key} must be a boolean`,
        param: `stream_options.${key}`,
        code: "invalid_type",
      };
    }
  }
  return null;
}

function validateToolDefinition(
  tool: unknown,
  index: number,
): ValidationIssue | null {
  if (!isRecord(tool) || tool.type !== "function" || !isRecord(tool.function)) {
    return {
      message: `tools[${index}] must be a function tool`,
      param: `tools[${index}]`,
      code: "invalid_tools",
    };
  }
  const fn = tool.function;
  if (
    typeof fn.name !== "string" ||
    !/^[A-Za-z0-9_-]{1,64}$/.test(fn.name)
  ) {
    return {
      message: `tools[${index}].function.name must be a valid function name`,
      param: `tools[${index}].function.name`,
      code: "invalid_tool_name",
    };
  }
  if (fn.description !== undefined && fn.description !== null && typeof fn.description !== "string") {
    return {
      message: `tools[${index}].function.description must be a string`,
      param: `tools[${index}].function.description`,
      code: "invalid_type",
    };
  }
  if (fn.strict !== undefined && fn.strict !== null && typeof fn.strict !== "boolean") {
    return {
      message: `tools[${index}].function.strict must be a boolean`,
      param: `tools[${index}].function.strict`,
      code: "invalid_type",
    };
  }
  if (fn.parameters !== undefined && fn.parameters !== null) {
    if (!isRecord(fn.parameters)) {
      return {
        message: `tools[${index}].function.parameters must be an object`,
        param: `tools[${index}].function.parameters`,
        code: "invalid_tool_parameters",
      };
    }
    try {
      // Validate the same Gemini subset used for function declarations.
      convertSchemaForGemini(fn.parameters, "responseSchema");
    } catch (error: unknown) {
      if (error instanceof SchemaConversionError) {
        return {
          message: `tools[${index}].function.parameters is not supported: ${error.message}`,
          param: `tools[${index}].function.parameters`,
          code: error.code,
        };
      }
      return {
        message: `tools[${index}].function.parameters must be valid JSON schema`,
        param: `tools[${index}].function.parameters`,
        code: "invalid_tool_parameters",
      };
    }
  }
  return null;
}

function validateToolChoice(
  body: Record<string, unknown>,
  tools: unknown[] | undefined,
): ValidationIssue | null {
  const choice = body.tool_choice;
  if (choice === undefined || choice === null) return null;
  if (choice === "none" || choice === "auto") return null;
  if (choice === "required") {
    if (!tools || tools.length === 0) {
      return {
        message: "tool_choice 'required' requires at least one tool",
        param: "tool_choice",
        code: "invalid_tool_choice",
      };
    }
    return null;
  }
  if (
    !isRecord(choice) ||
    choice.type !== "function" ||
    !isRecord(choice.function) ||
    typeof choice.function.name !== "string"
  ) {
    return {
      message: "tool_choice must be 'none', 'auto', 'required', or a function selector",
      param: "tool_choice",
      code: "invalid_tool_choice",
    };
  }
  const selectedFunctionName = choice.function.name;
  if (!tools || tools.length === 0) {
    return {
      message: "tool_choice function selector requires at least one tool",
      param: "tool_choice",
      code: "invalid_tool_choice",
    };
  }
  const exists = tools.some(
    (tool) =>
      isRecord(tool) &&
      isRecord(tool.function) &&
      tool.function.name === selectedFunctionName,
  );
  if (!exists) {
    return {
      message: `tool_choice references unknown function '${selectedFunctionName}'`,
      param: "tool_choice.function.name",
      code: "invalid_tool_choice",
    };
  }
  return null;
}

function validateToolConversation(
  messages: unknown[],
): ValidationIssue | null {
  const pendingToolCalls = new Set<string>();

  for (let index = 0; index < messages.length; index++) {
    const message = messages[index];
    if (!isRecord(message)) continue;

    if (message.role === "assistant" && Array.isArray(message.tool_calls)) {
      if (pendingToolCalls.size > 0) {
        return {
          message: `messages[${index}] starts a new assistant turn before all tool calls were answered`,
          param: `messages[${index}]`,
          code: "missing_tool_call_response",
        };
      }
      for (const toolCall of message.tool_calls) {
        if (isRecord(toolCall) && typeof toolCall.id === "string") {
          pendingToolCalls.add(toolCall.id);
        }
      }
      continue;
    }

    if (message.role === "tool") {
      const toolCallId = message.tool_call_id;
      if (typeof toolCallId !== "string" || !pendingToolCalls.has(toolCallId)) {
        return {
          message: `messages[${index}].tool_call_id does not match a pending assistant tool call`,
          param: `messages[${index}].tool_call_id`,
          code: "invalid_tool_call_id",
        };
      }
      pendingToolCalls.delete(toolCallId);
      continue;
    }

    if (
      pendingToolCalls.size > 0 &&
      (message.role === "user" || message.role === "assistant" || message.role === "system" || message.role === "developer")
    ) {
      return {
        message: `messages[${index}] continues the conversation before all tool calls were answered`,
        param: `messages[${index}]`,
        code: "missing_tool_call_response",
      };
    }
  }

  if (pendingToolCalls.size > 0) {
    return {
      message: "Every assistant tool call must be followed by a matching tool message",
      param: "messages",
      code: "missing_tool_call_response",
    };
  }
  return null;
}

export function validateChatCompletionRequest(
  body: unknown,
  schemaMode?: string,
): ValidationIssue | null {
  if (!isRecord(body)) {
    return {
      message: "Request body must be a JSON object",
      code: "invalid_request_body",
    };
  }

  if (
    body.model !== undefined &&
    (typeof body.model !== "string" || body.model.trim() === "")
  ) {
    return {
      message: "model must be a non-empty string",
      param: "model",
      code: "invalid_model",
    };
  }

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return {
      message: "messages is a required field",
      param: "messages",
      code: "missing_messages",
    };
  }

  const allowedRoles = new Set([
    "developer",
    "system",
    "user",
    "assistant",
    "tool",
  ]);
  for (let index = 0; index < body.messages.length; index++) {
    const message = body.messages[index];
    if (!isRecord(message)) {
      return {
        message: `messages[${index}] must be an object`,
        param: `messages[${index}]`,
        code: "invalid_message",
      };
    }
    if (typeof message.role !== "string" || !allowedRoles.has(message.role)) {
      return {
        message: `messages[${index}].role is invalid`,
        param: `messages[${index}].role`,
        code: "invalid_role",
      };
    }

    const contentIsValid =
      typeof message.content === "string" ||
      Array.isArray(message.content) ||
      message.content === null;
    if (!contentIsValid) {
      return {
        message: `messages[${index}].content must be a string, an array, or null`,
        param: `messages[${index}].content`,
        code: "invalid_content",
      };
    }
    if (
      message.content === null &&
      !(message.role === "assistant" && Array.isArray(message.tool_calls))
    ) {
      return {
        message: `messages[${index}].content may only be null for an assistant tool-call message`,
        param: `messages[${index}].content`,
        code: "invalid_content",
      };
    }
    if (message.role === "tool" && typeof message.tool_call_id !== "string") {
      return {
        message: `messages[${index}].tool_call_id is required for tool messages`,
        param: `messages[${index}].tool_call_id`,
        code: "missing_tool_call_id",
      };
    }

    if (message.role === "assistant" && message.tool_calls !== undefined) {
      if (!Array.isArray(message.tool_calls)) {
        return {
          message: `messages[${index}].tool_calls must be an array`,
          param: `messages[${index}].tool_calls`,
          code: "invalid_tool_calls",
        };
      }
      for (
        let toolIndex = 0;
        toolIndex < message.tool_calls.length;
        toolIndex++
      ) {
        const toolCall = message.tool_calls[toolIndex];
        if (
          !isRecord(toolCall) ||
          typeof toolCall.id !== "string" ||
          !isRecord(toolCall.function) ||
          typeof toolCall.function.name !== "string"
        ) {
          return {
            message: `messages[${index}].tool_calls[${toolIndex}] is invalid`,
            param: `messages[${index}].tool_calls[${toolIndex}]`,
            code: "invalid_tool_call",
          };
        }
        if (typeof toolCall.function.arguments !== "string") {
          return {
            message: `messages[${index}].tool_calls[${toolIndex}].function.arguments must be a JSON string`,
            param: `messages[${index}].tool_calls[${toolIndex}].function.arguments`,
            code: "invalid_tool_arguments",
          };
        }
        try {
          JSON.parse(toolCall.function.arguments);
        } catch {
          return {
            message: `messages[${index}].tool_calls[${toolIndex}].function.arguments must contain valid JSON`,
            param: `messages[${index}].tool_calls[${toolIndex}].function.arguments`,
            code: "invalid_tool_arguments",
          };
        }
      }
    }
  }

  const toolConversationIssue = validateToolConversation(body.messages);
  if (toolConversationIssue) return toolConversationIssue;

  for (const [key, minimum, maximum, integer] of [
    ["max_completion_tokens", 1, undefined, true],
    ["max_tokens", 1, undefined, true],
    ["temperature", 0, 2, false],
    ["top_p", 0, 1, false],
    ["presence_penalty", -2, 2, false],
    ["frequency_penalty", -2, 2, false],
    ["seed", Number.MIN_SAFE_INTEGER, Number.MAX_SAFE_INTEGER, true],
  ] as const) {
    const issue = validateOptionalNumber(body, key, minimum, maximum, integer);
    if (issue) return issue;
  }

  if (body.n !== undefined && body.n !== null && body.n !== 1) {
    return {
      message: "Only n=1 is currently supported",
      param: "n",
      code: "unsupported_value",
    };
  }

  if (
    body.stop !== undefined &&
    body.stop !== null &&
    typeof body.stop !== "string" &&
    !(
      Array.isArray(body.stop) &&
      body.stop.every((entry) => typeof entry === "string")
    )
  ) {
    return {
      message: "stop must be a string or an array of strings",
      param: "stop",
      code: "invalid_type",
    };
  }

  if (
    body.logprobs === true ||
    (body.top_logprobs !== undefined && body.top_logprobs !== null)
  ) {
    return {
      message: "logprobs are not supported",
      param:
        body.top_logprobs !== undefined && body.top_logprobs !== null
          ? "top_logprobs"
          : "logprobs",
      code: "unsupported_value",
    };
  }

  const streamIssue = validateStreamOptions(body, body.stream === true);
  if (streamIssue) return streamIssue;

  if (body.tools !== undefined && body.tools !== null && !Array.isArray(body.tools)) {
    return {
      message: "tools must be an array",
      param: "tools",
      code: "invalid_tools",
    };
  }

  if (Array.isArray(body.tools)) {
    for (let index = 0; index < body.tools.length; index++) {
      const issue = validateToolDefinition(body.tools[index], index);
      if (issue) return issue;
    }
  }

  const toolChoiceIssue = validateToolChoice(
    body,
    body.tools as unknown[] | undefined,
  );
  if (toolChoiceIssue) return toolChoiceIssue;

  if (body.response_format !== undefined && body.response_format !== null) {
    if (!isRecord(body.response_format)) {
      return {
        message: "response_format must be an object",
        param: "response_format",
        code: "invalid_type",
      };
    }

    const formatType = body.response_format.type;
    if (!["text", "json_object", "json_schema"].includes(String(formatType))) {
      return {
        message: "response_format.type must be text, json_object, or json_schema",
        param: "response_format.type",
        code: "invalid_value",
      };
    }

    if (formatType === "json_schema") {
      const jsonSchema = body.response_format.json_schema;
      if (
        !isRecord(jsonSchema) ||
        typeof jsonSchema.name !== "string" ||
        jsonSchema.name.length === 0 ||
        jsonSchema.name.length > 64 ||
        !/^[A-Za-z0-9_-]+$/.test(jsonSchema.name) ||
        !isRecord(jsonSchema.schema)
      ) {
        return {
          message: "response_format.json_schema must include a valid name and schema object",
          param: "response_format.json_schema",
          code: "invalid_json_schema",
        };
      }
      if (
        jsonSchema.strict !== undefined &&
        jsonSchema.strict !== null &&
        typeof jsonSchema.strict !== "boolean"
      ) {
        return {
          message: "response_format.json_schema.strict must be a boolean",
          param: "response_format.json_schema.strict",
          code: "invalid_json_schema",
        };
      }
      try {
        convertSchemaForGemini(
          jsonSchema.schema,
          selectSchemaProtocol(
            typeof body.model === "string" ? body.model : "",
            schemaMode,
          ),
        );
      } catch (error: unknown) {
        if (error instanceof SchemaConversionError) {
          return {
            message: error.message,
            param: "response_format.json_schema.schema",
            code: error.code,
          };
        }
        return {
          message: "response_format.json_schema.schema is invalid",
          param: "response_format.json_schema.schema",
          code: "invalid_json_schema",
        };
      }
    }
  }

  return null;
}

export function validateTextCompletionRequest(
  body: unknown,
): ValidationIssue | null {
  if (!isRecord(body)) {
    return {
      message: "Request body must be a JSON object",
      code: "invalid_request_body",
    };
  }
  if (
    body.model !== undefined &&
    (typeof body.model !== "string" || body.model.trim() === "")
  ) {
    return {
      message: "model must be a non-empty string",
      param: "model",
      code: "invalid_model",
    };
  }

  const prompt = body.prompt;
  if (
    typeof prompt !== "string" &&
    !(Array.isArray(prompt) && prompt.every((entry) => typeof entry === "string"))
  ) {
    return {
      message: "prompt must be a string or an array of strings",
      param: "prompt",
      code: "invalid_prompt",
    };
  }
  if (Array.isArray(prompt) && prompt.length !== 1) {
    return {
      message: "Only a single prompt is currently supported",
      param: "prompt",
      code: "unsupported_value",
    };
  }

  if (body.n !== undefined && body.n !== null && body.n !== 1) {
    return {
      message: "Only n=1 is currently supported",
      param: "n",
      code: "unsupported_value",
    };
  }
  if (body.best_of !== undefined && body.best_of !== null && body.best_of !== 1) {
    return {
      message: "best_of is not supported",
      param: "best_of",
      code: "unsupported_value",
    };
  }
  if (body.logprobs !== undefined && body.logprobs !== null) {
    return {
      message: "logprobs are not supported",
      param: "logprobs",
      code: "unsupported_value",
    };
  }
  if (body.echo === true) {
    return {
      message: "echo is not supported",
      param: "echo",
      code: "unsupported_value",
    };
  }
  if (body.suffix !== undefined && body.suffix !== null && body.suffix !== "") {
    return {
      message: "suffix is not supported",
      param: "suffix",
      code: "unsupported_value",
    };
  }
  if (
    body.stop !== undefined &&
    body.stop !== null &&
    typeof body.stop !== "string" &&
    !(Array.isArray(body.stop) && body.stop.every((entry) => typeof entry === "string"))
  ) {
    return {
      message: "stop must be a string or an array of strings",
      param: "stop",
      code: "invalid_type",
    };
  }

  for (const [key, minimum, maximum, integer] of [
    ["max_tokens", 1, undefined, true],
    ["temperature", 0, 2, false],
    ["top_p", 0, 1, false],
    ["presence_penalty", -2, 2, false],
    ["frequency_penalty", -2, 2, false],
    ["seed", Number.MIN_SAFE_INTEGER, Number.MAX_SAFE_INTEGER, true],
  ] as const) {
    const issue = validateOptionalNumber(body, key, minimum, maximum, integer);
    if (issue) return issue;
  }

  return validateStreamOptions(body, body.stream === true);
}

export function createOpenAIError(
  message: string,
  type: string,
  code: string | number | null,
  param: string | null = null,
  requestId?: string,
): OpenAIErrorBody {
  const error: OpenAIErrorBody["error"] = { message, type, param, code };
  if (requestId) error.request_id = requestId;
  return { error };
}

export function getErrorStatus(error: unknown): number {
  if (error instanceof UpstreamApiError) return error.status;
  if (error instanceof SchemaConversionError) return 400;
  if (
    typeof error === "object" &&
    error !== null &&
    "status" in error &&
    typeof (error as { status?: unknown }).status === "number"
  ) {
    return (error as { status: number }).status;
  }
  if (isAbortError(error)) {
    return /timed out/i.test(error instanceof Error ? error.message : String(error))
      ? 504
      : 499;
  }
  const message = error instanceof Error ? error.message : String(error);
  if (/timed out|timeout/i.test(message)) return 504;
  if (/authentication|authorization|invalid[_ ]grant/i.test(message)) return 401;
  return 500;
}

export function getErrorType(status: number): string {
  if (status === 400 || status === 404 || status === 422) return "invalid_request_error";
  if (status === 401) return "authentication_error";
  if (status === 403) return "permission_error";
  if (status === 429) return "rate_limit_error";
  if (status === 499) return "invalid_request_error";
  return "api_error";
}

export function getErrorRequestId(error: unknown): string | undefined {
  return error instanceof UpstreamApiError ? error.requestId : undefined;
}

export function mapGeminiFinishReason(
  reason: string | undefined,
): ChatCompletionFinishReason {
  switch (reason?.toUpperCase()) {
    case "MAX_TOKENS":
      return "length";
    case "SAFETY":
    case "RECITATION":
    case "BLOCKLIST":
    case "PROHIBITED_CONTENT":
    case "SPII":
    case "IMAGE_SAFETY":
    case "IMAGE_PROHIBITED_CONTENT":
      return "content_filter";
    case "MALFORMED_FUNCTION_CALL":
    case "UNEXPECTED_TOOL_CALL":
    case "TOO_MANY_TOOL_CALLS":
      return "tool_calls";
    default:
      return "stop";
  }
}

export function mapGeminiUsage(usage: {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  thoughtsTokenCount?: number;
  cachedContentTokenCount?: number;
  toolUsePromptTokenCount?: number;
}): ChatCompletionUsage {
  const promptTokens = usage.promptTokenCount || 0;
  const reasoningTokens = usage.thoughtsTokenCount || 0;
  const completionTokens = (usage.candidatesTokenCount || 0) + reasoningTokens;
  const mapped: ChatCompletionUsage = {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
  };

  if (reasoningTokens > 0) {
    mapped.completion_tokens_details = { reasoning_tokens: reasoningTokens };
  }
  if ((usage.cachedContentTokenCount || 0) > 0) {
    mapped.prompt_tokens_details = {
      cached_tokens: usage.cachedContentTokenCount,
    };
  }
  if ((usage.toolUsePromptTokenCount || 0) > 0) {
    mapped.prompt_tokens_details = {
      ...mapped.prompt_tokens_details,
      tool_use_tokens: usage.toolUsePromptTokenCount,
    };
    mapped.completion_tokens_details = {
      ...mapped.completion_tokens_details,
      tool_use_tokens: usage.toolUsePromptTokenCount,
    };
  }
  return mapped;
}

export function asChatCompletionRequest(body: unknown): ChatCompletionRequest {
  return body as ChatCompletionRequest;
}
