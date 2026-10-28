import {
  StreamChunk,
  ReasoningData,
  GeminiFunctionCall,
  UsageData,
  FinishData,
  StreamErrorData,
  ChatCompletionFinishReason,
} from "./types";
import { NativeToolResponse } from "./types/native-tools";
import { OPENAI_CHAT_COMPLETION_OBJECT } from "./config";
import { validateStructuredOutputText } from "./helpers/structured-output";

// OpenAI API interfaces
interface OpenAIToolCall {
	index: number;
	id: string;
	type: "function";
	function: {
		name: string;
		arguments: string;
	};
}

interface OpenAIChoice {
	index: number;
	delta: OpenAIDelta;
	finish_reason: string | null;
	logprobs?: null;
	matched_stop?: null;
}

interface OpenAIDelta {
	role?: string;
	content?: string | null;
	reasoning?: string;
	reasoning_content?: string | null;
	tool_calls?: OpenAIToolCall[];
	native_tool_calls?: NativeToolResponse[];
	grounding?: unknown;
}

interface OpenAIChunk {
	id: string;
	object: string;
	created: number;
	model: string;
	choices: OpenAIChoice[];
  usage?: OpenAIUsage | null;
}

interface OpenAITextChoice {
	index: number;
	text: string;
	logprobs?: null;
	finish_reason: string | null;
}

interface OpenAITextChunk {
	id: string;
	object: "text_completion";
	created: number;
	model: string;
	choices: OpenAITextChoice[];
	usage?: OpenAIUsage | null;
}

interface OpenAIUsage {
	prompt_tokens: number;
	completion_tokens: number;
	total_tokens: number;
	completion_tokens_details?: {
		reasoning_tokens?: number;
		tool_use_tokens?: number;
	};
	prompt_tokens_details?: {
		cached_tokens?: number;
		tool_use_tokens?: number;
	};
}

export interface StructuredStreamOptions {
  schema?: unknown;
  requireObject?: boolean;
  jsonSchemaName?: string;
}

export interface OpenAIStreamTransformerOptions {
  includeUsage?: boolean;
  structuredOutput?: StructuredStreamOptions;
}

// Type guard functions
function isReasoningData(data: unknown): data is ReasoningData {
  return (
    typeof data === "object" &&
    data !== null &&
    ("reasoning" in data || "toolCode" in data)
  );
}

function isGeminiFunctionCall(data: unknown): data is GeminiFunctionCall {
  return (
    typeof data === "object" &&
    data !== null &&
    "name" in data &&
    "args" in data
  );
}

function isUsageData(data: unknown): data is UsageData {
  return (
    typeof data === "object" &&
    data !== null &&
    "inputTokens" in data &&
    "outputTokens" in data
  );
}
function isNativeToolResponse(data: unknown): data is NativeToolResponse {
  return (
    typeof data === "object" &&
    data !== null &&
    "type" in data &&
    "data" in data
  );
}

function isFinishData(data: unknown): data is FinishData {
  return typeof data === "object" && data !== null && "reason" in data;
}

function isStreamErrorData(data: unknown): data is StreamErrorData {
  return (
    typeof data === "object" &&
    data !== null &&
    "message" in data &&
    "type" in data
  );
}

function toOpenAIUsage(usageData: UsageData): OpenAIUsage {
  const usage: OpenAIUsage = {
    prompt_tokens: usageData.inputTokens,
    completion_tokens: usageData.outputTokens,
    total_tokens: usageData.inputTokens + usageData.outputTokens,
  };
  if (
    usageData.reasoningTokens !== undefined ||
    usageData.toolUseTokens !== undefined
  ) {
    usage.completion_tokens_details = {};
    if (usageData.reasoningTokens !== undefined) {
      usage.completion_tokens_details.reasoning_tokens =
        usageData.reasoningTokens;
    }
    if (usageData.toolUseTokens !== undefined) {
      usage.completion_tokens_details.tool_use_tokens =
        usageData.toolUseTokens;
    }
  }
  if (
    usageData.cachedTokens !== undefined ||
    usageData.toolUseTokens !== undefined
  ) {
    usage.prompt_tokens_details = {};
    if (usageData.cachedTokens !== undefined) {
      usage.prompt_tokens_details.cached_tokens = usageData.cachedTokens;
    }
    if (usageData.toolUseTokens !== undefined) {
      usage.prompt_tokens_details.tool_use_tokens = usageData.toolUseTokens;
    }
  }
  return usage;
}

/**
 * Creates a TransformStream to convert Gemini's output chunks
 * into OpenAI-compatible server-sent events.
 */
export function createOpenAIStreamTransformer(
  model: string,
  options: OpenAIStreamTransformerOptions = {},
): TransformStream<StreamChunk, Uint8Array> {
	const chatID = `chatcmpl-${crypto.randomUUID()}`;
	const creationTime = Math.floor(Date.now() / 1000);
	const encoder = new TextEncoder();
	let toolCallIndex = 0; // Counter for parallel tool calls
	let hasToolCalls = false;
	let usageData: UsageData | undefined;
  let finishReason: ChatCompletionFinishReason = "stop";
  const includeUsage = options.includeUsage === true;
  const structuredOutput = options.structuredOutput;
  let structuredBuffer = "";
  let streamError: StreamErrorData | undefined;

  const createChunk = (
    choices: OpenAIChoice[],
    usage?: OpenAIUsage | null,
  ): OpenAIChunk => {
    const chunk: OpenAIChunk = {
      id: chatID,
      object: OPENAI_CHAT_COMPLETION_OBJECT,
      created: creationTime,
      model,
      choices,
    };
    if (includeUsage) chunk.usage = usage ?? null;
    return chunk;
  };

  const enqueueChunk = (
    controller: TransformStreamDefaultController<Uint8Array>,
    chunk: OpenAIChunk,
  ): void => {
    controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
  };

  const enqueueError = (
    controller: TransformStreamDefaultController<Uint8Array>,
    error: StreamErrorData,
  ): void => {
    controller.enqueue(
      encoder.encode(`data: ${JSON.stringify({ error })}\n\n`),
    );
  };

	return new TransformStream({
    start(controller) {
      enqueueChunk(
        controller,
        createChunk([
          {
            index: 0,
            delta: { role: "assistant", content: "" },
            finish_reason: null,
            logprobs: null,
          },
        ]),
      );
    },
		transform(chunk, controller) {
      if (streamError) return;
			const delta: OpenAIDelta = {};
			let openAIChunk: OpenAIChunk | null = null;

			switch (chunk.type) {
        case "error":
          if (isStreamErrorData(chunk.data)) {
            streamError = chunk.data;
          }
          return;
				case "text":
          if (typeof chunk.data === "string") {
            if (structuredOutput) {
              structuredBuffer += chunk.data;
              return;
            }
            delta.content = chunk.data;
          }
          break;
        case "thinking_content":
          if (typeof chunk.data === "string") {
            if (structuredOutput) return;
            delta.content = chunk.data;
          }
          break;
				case "real_thinking":
					if (typeof chunk.data === "string") {
						delta.reasoning = chunk.data;
						delta.reasoning_content = chunk.data;
					}
					break;
				case "reasoning":
					if (isReasoningData(chunk.data)) {
						delta.reasoning = chunk.data.reasoning;
						delta.reasoning_content = chunk.data.reasoning;
					}
					break;
				case "tool_code":
					if (isGeminiFunctionCall(chunk.data)) {
						const toolData = chunk.data;
						hasToolCalls = true;
						delta.tool_calls = [
							{
								index: toolCallIndex++, // Increment index for each call
								id: toolData.id || `call_${crypto.randomUUID()}`,
								type: "function",
								function: {
									name: toolData.name,
                  arguments: JSON.stringify(toolData.args ?? {}),
                },
              },
						];
					}
					break;
				case "native_tool":
					if (isNativeToolResponse(chunk.data)) {
						delta.native_tool_calls = [chunk.data];
					}
					break;
				case "grounding_metadata":
					if (chunk.data) {
						delta.grounding = chunk.data;
					}
					break;
				case "usage":
					if (isUsageData(chunk.data)) {
						usageData = chunk.data;
					}
					return; // Don't send a chunk for usage data
        case "finish":
          if (isFinishData(chunk.data)) finishReason = chunk.data.reason;
          return;
			}

			if (Object.keys(delta).length > 0) {
        openAIChunk = createChunk([
						{
							index: 0,
							delta: delta,
							finish_reason: null,
							logprobs: null,
          },
        ]);
        enqueueChunk(controller, openAIChunk);
			}
		},
		flush(controller) {
      if (streamError) {
        enqueueError(controller, streamError);
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        return;
      }

      let structuredText = structuredBuffer;
      if (
        structuredOutput &&
        !hasToolCalls &&
        (structuredBuffer.length > 0 || structuredOutput.requireObject)
      ) {
        const validation = validateStructuredOutputText(structuredBuffer, {
          schema: structuredOutput.schema,
          requireObject: structuredOutput.requireObject,
        });
        if (!validation.ok) {
          enqueueError(controller, {
            message: validation.error.message,
            type: "api_error",
            code: "invalid_structured_output",
            param: validation.error.path ?? null,
          });
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          return;
        }
        structuredText = validation.text;
      }

      if (structuredText) {
        enqueueChunk(
          controller,
          createChunk([
            {
              index: 0,
              delta: { content: structuredText },
              finish_reason: null,
              logprobs: null,
            },
          ]),
        );
      }

      const resolvedFinishReason =
        finishReason === "length" || finishReason === "content_filter"
          ? finishReason
          : hasToolCalls
            ? "tool_calls"
            : finishReason;
      enqueueChunk(
        controller,
        createChunk([
          {
            index: 0,
            delta: {},
            finish_reason: resolvedFinishReason,
            logprobs: null,
          },
        ]),
      );

      if (includeUsage && usageData) {
        enqueueChunk(controller, createChunk([], toOpenAIUsage(usageData)));
			}

			controller.enqueue(encoder.encode("data: [DONE]\n\n"));
    },
	});
}

/**
 * Legacy /v1/completions streaming transformer.
 */
export function createOpenAITextStreamTransformer(
  model: string,
  options: { includeUsage?: boolean } = {},
): TransformStream<StreamChunk, Uint8Array> {
  const completionID = `cmpl-${crypto.randomUUID()}`;
  const creationTime = Math.floor(Date.now() / 1000);
  const encoder = new TextEncoder();
  const includeUsage = options.includeUsage === true;
  let usageData: UsageData | undefined;
  let finishReason: ChatCompletionFinishReason = "stop";
  let streamError: StreamErrorData | undefined;

  const createChunk = (
    choices: OpenAITextChoice[],
    usage?: OpenAIUsage | null,
  ): OpenAITextChunk => {
    const chunk: OpenAITextChunk = {
      id: completionID,
      object: "text_completion",
      created: creationTime,
      model,
      choices,
    };
    if (includeUsage) chunk.usage = usage ?? null;
    return chunk;
  };

  const enqueue = (
    controller: TransformStreamDefaultController<Uint8Array>,
    chunk: OpenAITextChunk,
  ) => {
    controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
  };

  return new TransformStream({
    transform(chunk, controller) {
      if (streamError) return;
      switch (chunk.type) {
        case "error":
          if (isStreamErrorData(chunk.data)) streamError = chunk.data;
          return;
        case "text":
        case "thinking_content":
          if (typeof chunk.data === "string") {
            enqueue(controller, createChunk([
              { index: 0, text: chunk.data, logprobs: null, finish_reason: null },
            ]));
          }
          return;
        case "usage":
          if (isUsageData(chunk.data)) usageData = chunk.data;
          return;
        case "finish":
          if (isFinishData(chunk.data)) finishReason = chunk.data.reason;
          return;
        default:
          return;
      }
    },
    flush(controller) {
      if (streamError) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: streamError })}\n\n`));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        return;
      }
      enqueue(controller, createChunk([
        { index: 0, text: "", logprobs: null, finish_reason: finishReason },
      ]));
      if (includeUsage && usageData) {
        enqueue(controller, createChunk([], toOpenAIUsage(usageData)));
      }
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
    },
  });
}
