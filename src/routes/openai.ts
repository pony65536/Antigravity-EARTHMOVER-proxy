import { Hono } from "hono";
import {
	Env,
	ChatCompletionRequest,
	ChatCompletionResponse,
	ChatMessage,
	ChatCompletionUsage,
	ModelInfo,
	MessageContent,
	EffortLevel,
	StreamErrorData,
	TextCompletionRequest,
	TextCompletionResponse,
	UsageData,
} from "../types";
import { DEFAULT_MODEL, getAllModelIds } from "../models";
import { OPENAI_MODEL_OWNER } from "../config";
import { MIME_TYPE_MAP } from "../constants";
import { AuthManager } from "../auth";
import { GeminiApiClient } from "../gemini-client";
import {
	createOpenAIStreamTransformer,
	createOpenAITextStreamTransformer,
	StructuredStreamOptions,
} from "../stream-transformer";
import { isMediaTypeSupported, validateContent, validateModel } from "../utils/validation";
import { Buffer } from "node:buffer";
import {
	asChatCompletionRequest,
	createOpenAIError,
	getErrorRequestId,
	getErrorStatus,
	getErrorType,
	validateChatCompletionRequest,
	validateTextCompletionRequest,
} from "../helpers/openai-compat";
import {
	convertSchemaForGemini,
	SchemaConversionError,
	selectSchemaProtocol,
	validateStructuredOutputText,
} from "../helpers/structured-output";
import { safeLogText } from "../utils/redact";

// Helper function to remove thinking blocks from message content
function stripThinkingBlocks(messages: ChatCompletionRequest["messages"]): ChatCompletionRequest["messages"] {
	if (!messages) return [];
	const thinkingRegex = /<think>[\s\S]*?<\/think>\s*/g;
	return messages.map((msg) => {
		if (typeof msg.content === "string") {
			return { ...msg, content: msg.content.replace(thinkingRegex, "") };
		}
		// Note: This doesn't handle array content, as thinking blocks are not expected there.
		return msg;
	});
}

function usageToOpenAI(usage: UsageData): ChatCompletionUsage {
	const mapped: ChatCompletionUsage = {
		prompt_tokens: usage.inputTokens,
		completion_tokens: usage.outputTokens,
		total_tokens: usage.inputTokens + usage.outputTokens,
	};
	if (
		usage.reasoningTokens !== undefined ||
		usage.toolUseTokens !== undefined
	) {
		mapped.completion_tokens_details = {};
		if (usage.reasoningTokens !== undefined) {
			mapped.completion_tokens_details.reasoning_tokens =
				usage.reasoningTokens;
		}
		if (usage.toolUseTokens !== undefined) {
			mapped.completion_tokens_details.tool_use_tokens =
				usage.toolUseTokens;
		}
	}
	if (usage.cachedTokens !== undefined || usage.toolUseTokens !== undefined) {
		mapped.prompt_tokens_details = {};
		if (usage.cachedTokens !== undefined) {
			mapped.prompt_tokens_details.cached_tokens = usage.cachedTokens;
		}
		if (usage.toolUseTokens !== undefined) {
			mapped.prompt_tokens_details.tool_use_tokens = usage.toolUseTokens;
		}
	}
	return mapped;
}

function errorPayload(error: unknown): StreamErrorData {
	const status = getErrorStatus(error);
	return {
		message: safeLogText(error instanceof Error ? error.message : String(error), 1000),
		type: getErrorType(status),
		code:
			typeof error === "object" &&
			error !== null &&
			"code" in error &&
			(error as { code?: string | number | null }).code != null
				? (error as { code: string | number }).code
				: status,
		param: null,
		request_id: getErrorRequestId(error) ?? null,
	};
}

function respondWithError(
	c: {
		header: (name: string, value: string) => void;
		json: (body: unknown, status?: number | string) => Response;
	},
	error: unknown,
	overrideStatus?: number,
): Response {
	const requestId = getErrorRequestId(error);
	if (requestId) c.header("x-request-id", requestId);
	const payload = errorPayload(error);
	const status = overrideStatus ?? getErrorStatus(error);
	return c.json(
		createOpenAIError(
			payload.message,
			payload.type,
			payload.code,
			payload.param ?? null,
			payload.request_id ?? undefined,
		),
		status as never,
	);
}

function structuredOutputFor(
	body: ChatCompletionRequest,
	model: string,
	env: Env,
): StructuredStreamOptions | undefined {
	const format = body.response_format;
	if (!format || format.type === "text") return undefined;
	if (format.type === "json_object") {
		return { requireObject: true };
	}
	const schema = format.json_schema.schema;
	convertSchemaForGemini(
		schema,
		selectSchemaProtocol(model, env.GEMINI_RESPONSE_SCHEMA_MODE),
	);
	return {
		schema,
		requireObject: true,
		jsonSchemaName: format.json_schema.name,
	};
}

function modelListFromDynamic(
	dynamicModels: Record<string, any> | null,
): Set<string> {
	const modelIds = new Set<string>();
	if (dynamicModels && Object.keys(dynamicModels).length > 0) {
		for (const [key, value] of Object.entries(dynamicModels)) {
			const modelObj = value as {
				isInternal?: boolean;
				displayName?: string;
			};
			if (
				modelObj.isInternal === true ||
				key.startsWith("chat_") ||
				key.startsWith("tab_") ||
				key.includes("image") ||
				(modelObj.displayName &&
					modelObj.displayName.toLowerCase().includes("image"))
			) {
				continue;
			}
			modelIds.add(modelObj.displayName || key);
			modelIds.add(key);
		}
	}
	return modelIds;
}

/**
 * OpenAI-compatible API routes for models and chat completions.
 */
export const OpenAIRoute = new Hono<{ Bindings: Env }>();

// List available models
OpenAIRoute.get("/models", async (c) => {
	const authManager = new AuthManager(c.env);
	const geminiClient = new GeminiApiClient(c.env, authManager);
	
	try {
		await authManager.initializeAuth();
		await geminiClient.fetchDynamicModels();
	} catch (e) {
		console.warn("Could not fetch dynamic models for /models endpoint", e);
	}

	const dynamicModels = geminiClient.getDynamicModels();
	const staticModelIds = getAllModelIds();
	
	const allModelIds = new Set<string>();
	
	if (dynamicModels && Object.keys(dynamicModels).length > 0) {
		for (const [key, value] of Object.entries(dynamicModels)) {
			const modelObj = value as any;
			
			// Filter out junk, internal, tab completion, and image generation models
			if (
				modelObj.isInternal === true ||
				key.startsWith("chat_") ||
				key.startsWith("tab_") ||
				key.includes("image") ||
				(modelObj.displayName && modelObj.displayName.toLowerCase().includes("image"))
			) {
				continue;
			}

			// Add the display name if available, otherwise fallback to key
			// This makes the IDE dropdown look clean (e.g. "Gemini 3.5 Flash (High)")
			allModelIds.add(modelObj.displayName || key);
		}
	} else {
		// Fallback to static models if dynamic fetch fails
		staticModelIds.forEach(id => allModelIds.add(id));
	}

	const modelData = Array.from(allModelIds).map((modelId) => ({
		id: modelId,
		object: "model",
		created: Math.floor(Date.now() / 1000),
		owned_by: OPENAI_MODEL_OWNER
	}));

	return c.json({
		object: "list",
		data: modelData
	});
});

// Retrieve a specific model
OpenAIRoute.get("/models/:model", async (c) => {
	const modelId = c.req.param("model");
	const authManager = new AuthManager(c.env);
	const geminiClient = new GeminiApiClient(c.env, authManager);

	let dynamicModelIds: Set<string> | undefined;
	try {
		await authManager.initializeAuth();
		await geminiClient.fetchDynamicModels();
		dynamicModelIds = modelListFromDynamic(geminiClient.getDynamicModels());
	} catch (error: unknown) {
		console.warn("Could not fetch dynamic models for /models/:model", safeLogText(String(error)));
	}

	const found =
		getAllModelIds().includes(modelId) ||
		(dynamicModelIds?.has(modelId) ?? false);

	if (!found) {
		return c.json(
			createOpenAIError(
				`The model '${modelId}' does not exist`,
				"invalid_request_error",
				"model_not_found",
				"model",
			),
			404,
		);
	}

	return c.json({
		id: modelId,
		object: "model",
		created: Math.floor(Date.now() / 1000),
		owned_by: OPENAI_MODEL_OWNER,
	});
});

// Chat completions endpoint
OpenAIRoute.post("/chat/completions", async (c) => {
	try {
		console.log("Chat completions request received");
		let rawBody: unknown;
		try {
			rawBody = await c.req.json();
		} catch {
			return c.json(createOpenAIError("Invalid JSON request body", "invalid_request_error", "invalid_json"), 400);
		}

		// --- Chub.AI Compatibility: Handle 'template' field ---
		// If 'messages' is missing but a 'template' field is present, create a messages array from it.
		if (
			typeof rawBody === "object" &&
			rawBody !== null &&
			!("messages" in rawBody) &&
			"template" in rawBody &&
			typeof rawBody.template === "string"
		) {
			console.log("Handling non-standard 'template' field for compatibility.");
			(rawBody as Record<string, unknown>).messages = [{ role: "user", content: rawBody.template }];
		}
		// --- End Compatibility ---

		const validationIssue = validateChatCompletionRequest(
			rawBody,
			c.env.GEMINI_RESPONSE_SCHEMA_MODE,
		);
		if (validationIssue) {
			return c.json(
				createOpenAIError(
					validationIssue.message,
					"invalid_request_error",
					validationIssue.code,
					validationIssue.param ?? null
				),
				400
			);
		}
		const body = asChatCompletionRequest(rawBody);

		const model = body.model || DEFAULT_MODEL;

		let structuredOutput: StructuredStreamOptions | undefined;
		try {
			structuredOutput = structuredOutputFor(body, model, c.env);
		} catch (error: unknown) {
			if (error instanceof SchemaConversionError) {
				return c.json(
					createOpenAIError(
						error.message,
						"invalid_request_error",
						error.code,
						"response_format.json_schema.schema",
					),
					400,
				);
			}
			throw error;
		}

		const messages = body.messages || [];
		// OpenAI API compatibility: streaming is opt-in.
		const stream = body.stream === true;

		// Extract system prompt and user/assistant messages first to allow for prompt-based commands
		let fullSystemPrompt = "";
		const otherMessages = messages.filter((msg) => {
			if (msg.role === "system" || msg.role === "developer") {
				// Handle system messages with both string and array content
				if (typeof msg.content === "string") {
					fullSystemPrompt += msg.content + "\n";
				} else if (Array.isArray(msg.content)) {
					// For system messages, only extract text content
					const textContent = msg.content
						.filter((part) => part.type === "text")
						.map((part) => part.text || "")
						.join(" ");
					fullSystemPrompt += textContent + "\n";
				}
				return false;
			}
			return true;
		});

		// Handle commands from system prompt (gives users control in limited UIs)
		const effortRegex = /reasoning_effort=(low|medium|high|none)\s*/i;
		const showRegex = /show_reasoning=(true|false)\s*/i;
		const cleanRegex = /clean_context=(true|false)\s*/i;
		
		const promptEffortMatch = fullSystemPrompt.match(effortRegex);
		const promptShowMatch = fullSystemPrompt.match(showRegex);
		const promptCleanMatch = fullSystemPrompt.match(cleanRegex);
		
		let effortFromPrompt: string | null = null;
		let showReasoning = true; // Default to showing reasoning
		let cleanContext = true; // Default to cleaning the context

		if (promptEffortMatch) {
			effortFromPrompt = promptEffortMatch[1].toLowerCase();
			fullSystemPrompt = fullSystemPrompt.replace(effortRegex, "").trim();
			console.log(`Reasoning effort '${effortFromPrompt}' detected in system prompt.`);
		}

		if (promptShowMatch) {
			showReasoning = promptShowMatch[1].toLowerCase() === "true";
			fullSystemPrompt = fullSystemPrompt.replace(showRegex, "").trim();
			console.log(`Show reasoning set to '${showReasoning}' from system prompt.`);
		}

		if (promptCleanMatch) {
			cleanContext = promptCleanMatch[1].toLowerCase() === "true";
			fullSystemPrompt = fullSystemPrompt.replace(cleanRegex, "").trim();
			console.log(`Clean context set to '${cleanContext}' from system prompt.`);
		}
		
		const systemPrompt = fullSystemPrompt.trim();

		// Determine the final reasoning effort, giving precedence to the system prompt
		const reasoning_effort =
			effortFromPrompt || // Precedence for prompt
			body.reasoning_effort ||
			body.extra_body?.reasoning_effort ||
			body.model_params?.reasoning_effort;

		// Determine if reasoning should be included at all
		const isRealThinkingEnabled = c.env.ENABLE_REAL_THINKING === "true";
		const includeReasoning = reasoning_effort ? reasoning_effort !== "none" : isRealThinkingEnabled;

		// --- End Reasoning Configuration ---

		// Conditionally clean thinking blocks from the history before sending to the model
		const cleanedMessages = cleanContext ? stripThinkingBlocks(otherMessages) : otherMessages;

		// --- DEBUG: Log the cleaned messages to verify stripping ---
		// console.log("Cleaned messages being sent to model:", JSON.stringify(cleanedMessages, null, 2));
		// --- END DEBUG ---

		// Newly added parameters
		const generationOptions = {
			max_completion_tokens: body.max_completion_tokens,
			max_tokens: body.max_tokens,
			temperature: body.temperature,
			top_p: body.top_p,
			stop: body.stop,
			presence_penalty: body.presence_penalty,
			frequency_penalty: body.frequency_penalty,
			seed: body.seed,
			response_format: body.response_format,
			thinking_budget: body.thinking_budget,
			enable_search: body.enable_search,
			enable_url_context: body.enable_url_context,
			enable_native_tools: body.enable_native_tools,
			native_tools_priority: body.native_tools_priority,
			extra_body: body.extra_body,
			model_params: body.model_params,
			signal: c.req.raw.signal
		};

		const tools = body.tools;
		const tool_choice = body.tool_choice;

		console.log("Request body parsed:", {
			model,
			messageCount: messages.length,
			stream,
			includeReasoning,
			reasoning_effort,
			tools,
			tool_choice
		});

		// Unified media validation
		const mediaChecks: {
			type: string;
			supportKey: keyof ModelInfo;
			name: string;
		}[] = [
			{ type: "image_url", supportKey: "supportsImages", name: "image inputs" },
			{ type: "input_audio", supportKey: "supportsAudios", name: "audio inputs" },
			{ type: "input_video", supportKey: "supportsVideos", name: "video inputs" },
			{ type: "input_pdf", supportKey: "supportsPdfs", name: "PDF inputs" }
		];

		for (const { type, supportKey, name } of mediaChecks) {
			const messagesWithMedia = messages.filter(
				(msg) => Array.isArray(msg.content) && msg.content.some((content) => content.type === type)
			);

			if (messagesWithMedia.length > 0) {
				if (!isMediaTypeSupported(model, supportKey)) {
					return c.json(
						createOpenAIError(
							`Model '${model}' does not support ${name}. Please use a model that supports this feature.`,
							"invalid_request_error",
							"unsupported_media_type",
							"messages"
						),
						400
					);
				}

				for (const msg of messagesWithMedia) {
					for (const content of msg.content as MessageContent[]) {
						if (content.type === type) {
							const { isValid, error } = validateContent(type, content);
							if (!isValid) {
								return c.json(
									createOpenAIError(error || "Invalid media content", "invalid_request_error", "invalid_media", "messages"),
									400
								);
							}
						}
					}
				}
			}
		}

		// Initialize services
		const authManager = new AuthManager(c.env);
		const geminiClient = new GeminiApiClient(c.env, authManager);

		if (stream) {
			console.log("Starting stream generation");
			const geminiStream = geminiClient.streamContent(model, systemPrompt, cleanedMessages, {
				includeReasoning,
				reasoning_effort: (reasoning_effort as EffortLevel) || undefined, // Pass the string effort
				tools,
				tool_choice,
				showReasoning,
				...generationOptions
			});

			let firstResult;
			try {
				// Peek at the first chunk to catch any immediate errors (like 429 or 400)
				// before we commit to sending a 200 OK SSE response.
				firstResult = await geminiStream.next();
			} catch (streamError: unknown) {
				console.error("Stream initialization error:", safeLogText(String(streamError)));
				return respondWithError(c, streamError);
			}

			// Streaming response
			const { readable, writable } = new TransformStream();
			const writer = writable.getWriter();
			const openAITransformer = createOpenAIStreamTransformer(model, {
				includeUsage: body.stream_options?.include_usage === true,
				structuredOutput,
			});
			const openAIStream = readable.pipeThrough(openAITransformer);

			// Asynchronously pipe data from Gemini to transformer
			(async () => {
				try {
					if (!firstResult.done) {
						await writer.write(firstResult.value);
					}

					for await (const chunk of geminiStream) {
						await writer.write(chunk);
					}
					console.log("Stream completed successfully");
					await writer.close();
				} catch (streamError: unknown) {
					if (c.req.raw.signal.aborted) {
						await writer.abort(streamError).catch(() => undefined);
						return;
					}
					console.error("Stream error mid-stream:", safeLogText(String(streamError)));
					// Once headers are sent, represent upstream failures as a stable SSE
					// error event followed by DONE instead of silently truncating.
					try {
						await writer.write({ type: "error", data: errorPayload(streamError) });
						await writer.close();
					} catch {
						await writer.abort(streamError).catch(() => undefined);
					}
				}
			})();

			// Return streaming response
			console.log("Returning streaming response");
			return new Response(openAIStream, {
				headers: {
					"Content-Type": "text/event-stream",
					"Cache-Control": "no-cache",
					Connection: "keep-alive"
				}
			});
		} else {
			// Non-streaming response
			try {
				console.log("Starting non-streaming completion");
				const completion = await geminiClient.getCompletion(model, systemPrompt, cleanedMessages, {
					includeReasoning,
					reasoning_effort: (reasoning_effort as EffortLevel) || undefined,
					tools,
					tool_choice,
					showReasoning,
					...generationOptions
				});

				let responseContent = completion.content;
				if (structuredOutput) {
					const validation = validateStructuredOutputText(completion.content, {
						schema: structuredOutput.schema,
						requireObject: structuredOutput.requireObject,
					});
					if (!validation.ok) {
						return c.json(
							createOpenAIError(
								validation.error.message,
								"api_error",
								"invalid_structured_output",
								validation.error.path ?? null,
							),
							502,
						);
					}
					responseContent = validation.text;
				}

				const response: ChatCompletionResponse = {
					id: `chatcmpl-${crypto.randomUUID()}`,
					object: "chat.completion",
					created: Math.floor(Date.now() / 1000),
					model: model,
					choices: [
						{
							index: 0,
							message: {
								role: "assistant",
								content: completion.tool_calls?.length && !responseContent ? null : responseContent,
								tool_calls: completion.tool_calls,
								reasoning: completion.reasoning,
								reasoning_content: completion.reasoning_content
							},
							finish_reason: completion.finish_reason ?? "stop"
						}
					]
				};

				// Add usage information if available
				if (completion.usage) {
					response.usage = usageToOpenAI(completion.usage);
				}

				console.log("Non-streaming completion successful");
				return c.json(response);
			} catch (completionError: unknown) {
				console.error("Completion error:", safeLogText(String(completionError)));
				return respondWithError(c, completionError);
			}
		}
	} catch (e: unknown) {
		console.error("Top-level error:", safeLogText(String(e)));
		return respondWithError(c, e);
	}
});

// Legacy text completions endpoint
OpenAIRoute.post("/completions", async (c) => {
	try {
		let rawBody: unknown;
		try {
			rawBody = await c.req.json();
		} catch {
			return c.json(
				createOpenAIError(
					"Invalid JSON request body",
					"invalid_request_error",
					"invalid_json",
				),
				400,
			);
		}

		const validationIssue = validateTextCompletionRequest(rawBody);
		if (validationIssue) {
			return c.json(
				createOpenAIError(
					validationIssue.message,
					"invalid_request_error",
					validationIssue.code,
					validationIssue.param ?? null,
				),
				400,
			);
		}

		const body = rawBody as TextCompletionRequest;
		const model = body.model || DEFAULT_MODEL;
		const rawPrompt = Array.isArray(body.prompt) ? body.prompt[0] : body.prompt;
		const promptValue = typeof rawPrompt === "string" ? rawPrompt : String(rawPrompt);
		const messages: ChatMessage[] = [{ role: "user", content: promptValue }];
		const stream = body.stream === true;
		const generationOptions = {
			max_tokens: body.max_tokens,
			temperature: body.temperature,
			top_p: body.top_p,
			stop: body.stop,
			presence_penalty: body.presence_penalty,
			seed: body.seed,
			frequency_penalty: body.frequency_penalty,
			signal: c.req.raw.signal,
		};

		const authManager = new AuthManager(c.env);
		const geminiClient = new GeminiApiClient(c.env, authManager);

		if (stream) {
			const geminiStream = geminiClient.streamContent(model, "", messages, {
				includeReasoning: false,
				showReasoning: false,
				...generationOptions,
			});

			let firstResult;
			try {
				firstResult = await geminiStream.next();
			} catch (streamError: unknown) {
				console.error("Completion stream initialization error:", safeLogText(String(streamError)));
				return respondWithError(c, streamError);
			}

			const { readable, writable } = new TransformStream();
			const writer = writable.getWriter();
			const transformer = createOpenAITextStreamTransformer(model, {
				includeUsage: body.stream_options?.include_usage === true,
			});
			const openAIStream = readable.pipeThrough(transformer);

			(async () => {
				try {
					if (!firstResult.done) await writer.write(firstResult.value);
					for await (const chunk of geminiStream) {
						await writer.write(chunk);
					}
					await writer.close();
				} catch (streamError: unknown) {
					if (c.req.raw.signal.aborted) {
						await writer.abort(streamError).catch(() => undefined);
						return;
					}
					console.error("Completion stream error mid-stream:", safeLogText(String(streamError)));
					try {
						await writer.write({ type: "error", data: errorPayload(streamError) });
						await writer.close();
					} catch {
						await writer.abort(streamError).catch(() => undefined);
					}
				}
			})();

			return new Response(openAIStream, {
				headers: {
					"Content-Type": "text/event-stream",
					"Cache-Control": "no-cache",
					Connection: "keep-alive",
				},
			});
		}

		try {
			const completion = await geminiClient.getCompletion(model, "", messages, {
				includeReasoning: false,
				showReasoning: false,
				...generationOptions,
			});
			const response: TextCompletionResponse = {
				id: `cmpl-${crypto.randomUUID()}`,
				object: "text_completion",
				created: Math.floor(Date.now() / 1000),
				model,
				choices: [
					{
						text: completion.content,
						index: 0,
						logprobs: null,
						finish_reason: completion.finish_reason ?? "stop",
					},
				],
			};
			if (completion.usage) response.usage = usageToOpenAI(completion.usage);
			return c.json(response);
		} catch (completionError: unknown) {
			console.error("Completion error:", safeLogText(String(completionError)));
			return respondWithError(c, completionError);
		}
	} catch (error: unknown) {
		console.error("Top-level completion error:", safeLogText(String(error)));
		return respondWithError(c, error);
	}
});

// Audio transcriptions endpoint
OpenAIRoute.post("/audio/transcriptions", async (c) => {
	try {
		console.log("Audio transcription request received");
		const body = await c.req.parseBody();
		const file = body["file"];
		const model = (body["model"] as string) || DEFAULT_MODEL;
		const prompt = (body["prompt"] as string) || "Transcribe this audio in detail.";

		if (!file || !(file instanceof File)) {
			return c.json({ error: "File is required" }, 400);
		}

		// Validate model
		const modelValidation = validateModel(model);
		if (!modelValidation.isValid) {
			return c.json({ error: modelValidation.error }, 400);
		}

		let mimeType = file.type;

		// Fallback for application/octet-stream
		if (mimeType === "application/octet-stream" && file.name) {
			const ext = file.name.split(".").pop()?.toLowerCase();
			if (ext && MIME_TYPE_MAP[ext]) {
				mimeType = MIME_TYPE_MAP[ext];
				console.log(`Detected MIME type from extension .${ext}: ${mimeType}`);
			}
		}

		// Check for video or audio support based on MIME type
		const isVideo = mimeType.startsWith("video/");
		// gemini can generate transcriptions of videos too
		const isAudio = mimeType.startsWith("audio/");

		if (isVideo) {
			if (!isMediaTypeSupported(model, "supportsVideos")) {
				return c.json(
					{
						error: `Model '${model}' does not support video inputs.`
					},
					400
				);
			}
		} else if (isAudio) {
			if (!isMediaTypeSupported(model, "supportsAudios")) {
				return c.json(
					{
						error: `Model '${model}' does not support audio inputs.`
					},
					400
				);
			}
		} else {
			return c.json(
				{
					error: `Unsupported media type: ${mimeType}. Only audio and video files are supported.`
				},
				400
			);
		}

		// Convert File to base64
		const arrayBuffer = await file.arrayBuffer();
		console.log(`Processing audio file: size=${arrayBuffer.byteLength} bytes, type=${file.type}`);

		let base64Audio: string;
		try {
			base64Audio = Buffer.from(arrayBuffer).toString("base64");
		} catch (e: unknown) {
			const errorMessage = e instanceof Error ? e.message : String(e);
			console.error("Base64 conversion failed:", errorMessage);
			throw new Error(`Failed to process audio file: ${errorMessage}`);
		}

		// Construct message
		const messages: ChatMessage[] = [
			{
				role: "user",
				content: [
					{
						type: "text",
						text: prompt
					},
					{
						type: "input_audio",
						input_audio: {
							data: base64Audio,
							format: mimeType
						}
					}
				]
			}
		];

		// Initialize client
		const authManager = new AuthManager(c.env);
		const geminiClient = new GeminiApiClient(c.env, authManager);

		// Get completion
		const completion = await geminiClient.getCompletion(model, "", messages);

		return c.json({ text: completion.content });
	} catch (e: unknown) {
		const errorMessage = e instanceof Error ? e.message : String(e);
		console.error("Transcription error:", errorMessage);
		return c.json({ error: errorMessage }, 500);
	}
});
