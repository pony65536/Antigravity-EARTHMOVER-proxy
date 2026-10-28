import {
	Env,
	StreamChunk,
	ReasoningData,
	UsageData,
	ChatMessage,
	MessageContent,
	GeminiFunctionCall,
	GeminiCompletionOptions,
	ChatCompletionFinishReason
} from "./types";
import { AuthManager } from "./auth";
import {
	CODE_ASSIST_ENDPOINT,
	CODE_ASSIST_API_VERSION,
	DYNAMIC_MODEL_CACHE_TTL_MS,
	DYNAMIC_MODEL_STALE_TTL_MS,
	GENERATION_TIMEOUT_MS,
	MAX_RATE_LIMIT_RETRIES,
	MAX_RETRY_DELAY_MS,
	MODEL_DISCOVERY_TIMEOUT_MS,
	PROJECT_DISCOVERY_TIMEOUT_MS,
	PROJECT_ID_CACHE_TTL_MS
} from "./config";
import { REASONING_MESSAGES, REASONING_CHUNK_DELAY, THINKING_CONTENT_CHUNK_SIZE } from "./constants";
import { geminiCliModels } from "./models";
import { validateContent } from "./utils/validation";
import { GenerationConfigValidator } from "./helpers/generation-config-validator";
import { AutoModelSwitchingHelper } from "./helpers/auto-model-switching";
import { NativeToolsManager } from "./helpers/native-tools-manager";
import { CitationsProcessor } from "./helpers/citations-processor";
import { GeminiUrlContextMetadata, GroundingMetadata, NativeToolsRequestParams } from "./types/native-tools";
import { mapGeminiFinishReason, UpstreamApiError } from "./helpers/openai-compat";
import { selectSchemaProtocol } from "./helpers/structured-output";
import {
	createAbortScope,
	envTimeoutMs,
	fetchTextWithTimeout,
	getUpstreamRequestId,
	isAbortError,
	parseRetryAfterMs,
	sleep
} from "./utils/http";
import { safeLogText } from "./utils/redact";

// Gemini API response types
interface GeminiCandidate {
	content?: {
		parts?: Array<{ text?: string }>;
	};
	groundingMetadata?: GroundingMetadata;
	finishReason?: string;
}

interface GeminiUsageMetadata {
	promptTokenCount?: number;
	candidatesTokenCount?: number;
	thoughtsTokenCount?: number;
	cachedContentTokenCount?: number;
	toolUsePromptTokenCount?: number;
}

interface GeminiResponse {
	response?: {
		candidates?: GeminiCandidate[];
		usageMetadata?: GeminiUsageMetadata;
		promptFeedback?: {
			blockReason?: string;
		};
	};
	traceId?: string;
	error?: {
		code?: number;
		message?: string;
		status?: string;
	};
}

export interface GeminiPart {
	text?: string;
	thought?: boolean; // For real thinking chunks from Gemini
	functionCall?: {
		id?: string;
		name: string;
		args: object;
	};
	thoughtSignature?: string;
	functionResponse?: {
		id?: string;
		name: string;
		response: {
			result?: string;
			output?: string;
		};
		parts?: Array<{
			inlineData?: {
				mimeType: string;
				data: string;
			};
		}>;
	};
	inlineData?: {
		mimeType: string;
		data: string;
	};
	fileData?: {
		mimeType: string;
		fileUri: string;
	};
	url_context_metadata?: GeminiUrlContextMetadata;
	// docs: https://ai.google.dev/gemini-api/docs/video-understanding#clipping-intervals
	// all must not exceed video real values
	videoMetadata?: {
		startOffset?: string; // string in seconds (40s)
		endOffset?: string; // string in seconds (80s)
		fps?: number;
	};
}

// Message content types - keeping only the local ones needed
interface TextContent {
	type: "text";
	text: string;
}

interface GeminiFormattedMessage {
	role: string;
	parts: GeminiPart[];
}

interface ProjectDiscoveryResponse {
	cloudaicompanionProject?: string;
}

interface DynamicModelCacheEntry {
	models: Record<string, any>;
	expiresAt: number;
	staleUntil: number;
}
const dynamicModelsCache = new Map<string, DynamicModelCacheEntry>();
const dynamicModelsFetchInFlight = new Map<string, Promise<Record<string, any> | null>>();
const projectIdCache = new Map<string, { value: string | undefined; expiresAt: number }>();

export function clearDynamicModelCaches(): void {
	dynamicModelsCache.clear();
	dynamicModelsFetchInFlight.clear();
	projectIdCache.clear();
}

// Type guard functions
function isTextContent(content: MessageContent): content is TextContent {
	return content.type === "text" && typeof content.text === "string";
}

/**
 * Handles communication with Google's Gemini API through the Code Assist endpoint.
 * Manages project discovery, streaming, and response parsing.
 */
export class GeminiApiClient {
	private env: Env;
	private authManager: AuthManager;
	private projectId: string | null = null;
	private autoSwitchHelper: AutoModelSwitchingHelper;
	private dynamicModelMap: Record<string, any> | null = null;

	constructor(env: Env, authManager: AuthManager) {
		this.env = env;
		this.authManager = authManager;
		this.autoSwitchHelper = new AutoModelSwitchingHelper(env);
	}

	private accountCacheKey(): string {
		return this.authManager.getCurrentAccountId() || "legacy_single_account";
	}

	private modelDiscoveryTimeoutMs(): number {
		return envTimeoutMs(
			this.env as unknown as Record<string, unknown>,
			"GEMINI_MODEL_DISCOVERY_TIMEOUT_MS",
			MODEL_DISCOVERY_TIMEOUT_MS
		);
	}

	private projectDiscoveryTimeoutMs(): number {
		return envTimeoutMs(
			this.env as unknown as Record<string, unknown>,
			"GEMINI_PROJECT_DISCOVERY_TIMEOUT_MS",
			PROJECT_DISCOVERY_TIMEOUT_MS
		);
	}

	private generationTimeoutMs(): number {
		return envTimeoutMs(
			this.env as unknown as Record<string, unknown>,
			"GEMINI_GENERATION_TIMEOUT_MS",
			GENERATION_TIMEOUT_MS
		);
	}

	public async fetchDynamicModels(): Promise<void> {
		if (this.dynamicModelMap) return;

		const accountKey = this.accountCacheKey();
		const cached = dynamicModelsCache.get(accountKey);
		const now = Date.now();

		if (cached && now < cached.expiresAt) {
			this.dynamicModelMap = cached.models;
			return;
		}

		if (cached && now < cached.staleUntil) {
			this.dynamicModelMap = cached.models;
			// stale-while-revalidate: serve immediately and refresh in the background
			void this.refreshDynamicModels(accountKey, true);
			return;
		}

		const models = await this.refreshDynamicModels(accountKey, false);
		if (models) {
			this.dynamicModelMap = models;
		} else if (cached) {
			this.dynamicModelMap = cached.models;
		}
	}

	private refreshDynamicModels(
		accountKey: string,
		background: boolean
	): Promise<Record<string, any> | null> {
		const existing = dynamicModelsFetchInFlight.get(accountKey);
		if (existing) return existing;

		if (!background) {
			console.log("[DynamicModels] Fetching available models from Antigravity...");
		}

		const request = this.fetchModelsFromUpstream(background)
			.then((models) => {
				if (models) {
					const now = Date.now();
					dynamicModelsCache.set(accountKey, {
						models,
						expiresAt: now + DYNAMIC_MODEL_CACHE_TTL_MS,
						staleUntil: now + DYNAMIC_MODEL_STALE_TTL_MS
					});
					console.log(
						`[DynamicModels] Loaded ${Object.keys(models).length} models for account ${accountKey}.`
					);
				}
				return models;
			})
			.catch((error: unknown) => {
				if (!background) {
					console.error(
						`[DynamicModels] Failed to fetch dynamic models: ${safeLogText(String(error))}`
					);
				}
				return null;
			})
			.finally(() => {
				dynamicModelsFetchInFlight.delete(accountKey);
			});

		dynamicModelsFetchInFlight.set(accountKey, request);
		return request;
	}

	private async fetchModelsFromUpstream(background: boolean): Promise<Record<string, any> | null> {
		let accessToken = this.authManager.getAccessToken();
		if (!accessToken) {
			await this.authManager.initializeAuth();
			accessToken = this.authManager.getAccessToken();
		}
		if (!accessToken) return null;

		const requestModels = (token: string) =>
			fetchTextWithTimeout(
				`${CODE_ASSIST_ENDPOINT}/${CODE_ASSIST_API_VERSION}:fetchAvailableModels`,
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						"Authorization": `Bearer ${token}`,
						"User-Agent": "antigravity/cli/1.0.2 windows/amd64"
					},
					body: "{}"
				},
				this.modelDiscoveryTimeoutMs()
			);

		let { response, text } = await requestModels(accessToken);

		if (response.status === 401) {
			await this.authManager.clearTokenCache();
			await this.authManager.initializeAuth();
			accessToken = this.authManager.getAccessToken();
			if (!accessToken) return null;
			({ response, text } = await requestModels(accessToken));
		}

		if (!response.ok) {
			if (!background) {
				console.warn(
					`[DynamicModels] Server rejected the request: HTTP ${response.status} - ${safeLogText(text, 300)}`
				);
			}
			return null;
		}

		try {
			const data = JSON.parse(text) as { models?: Record<string, any> };
			return data.models || {};
		} catch {
			if (!background) {
				console.warn("[DynamicModels] Server returned invalid JSON");
			}
			return null;
		}
	}

	public getDynamicModels(): Record<string, any> | null {
		return this.dynamicModelMap;
	}

	/**
	 * Discovers the Google Cloud project ID. Uses the environment variable if provided.
	 */
	public async discoverProjectId(): Promise<string | undefined> {
		if (this.env.GEMINI_PROJECT_ID) {
			return this.env.GEMINI_PROJECT_ID;
		}
		if (this.projectId) {
			return this.projectId;
		}

		const accountKey = this.accountCacheKey();
		const cached = projectIdCache.get(accountKey);
		if (cached && cached.expiresAt > Date.now()) {
			this.projectId = cached.value ?? null;
			return cached.value;
		}

		try {
			const loadResponse = (await this.authManager.callEndpoint(
				"loadCodeAssist",
				{ metadata: { ideType: "ANTIGRAVITY" } },
				{ timeoutMs: this.projectDiscoveryTimeoutMs() }
			)) as ProjectDiscoveryResponse;

			// Handle both string and object responses for project ID
			const project = loadResponse.cloudaicompanionProject;
			const discoveredId =
				typeof project === "string" ? project : (project as { id?: string } | undefined)?.id;

			projectIdCache.set(accountKey, {
				value: discoveredId,
				expiresAt: Date.now() + PROJECT_ID_CACHE_TTL_MS
			});

			if (discoveredId) {
				this.projectId = discoveredId;
				return discoveredId;
			}
			console.warn("Project ID discovery returned empty, sending request without project ID.");
			return undefined;
		} catch (error: unknown) {
			console.warn("Failed to discover project ID:", safeLogText(String(error)));
			console.warn("Sending request without project ID to let Antigravity handle it.");
			return undefined;
		}
	}

	private sendFakeTelemetry(traceId: string, projectId: string, firstMessageLatency: number, totalLatency: number, includedCode: boolean = false, citationCount: number = 0): void {
		try {
			const payload = {
				project: projectId,
				metadata: {
					ideType: "GEMINI_CLI",
					pluginType: "GEMINI",
					duetProject: projectId
				},
				metrics: [
					{
						conversationOffered: {
							citationCount: String(citationCount),
							includedCode: includedCode,
							status: 1, // ACTION_STATUS_NO_ERROR
							traceId: traceId,
							streamingLatency: {
								firstMessageLatency: `${(firstMessageLatency / 1000).toFixed(3)}s`,
								totalLatency: `${(totalLatency / 1000).toFixed(3)}s`
							},
							isAgentic: true,
							initiationMethod: 2 // COMMAND
						},
						timestamp: new Date().toISOString()
					}
				]
			};

			// Fire and forget, don't await
			fetch(`${CODE_ASSIST_ENDPOINT}/${CODE_ASSIST_API_VERSION}:recordCodeAssistMetrics`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${this.authManager.getAccessToken()}`
				},
				body: JSON.stringify(payload)
			}).catch(() => { /* Silent fail */ });
		} catch (e) {
			// Silent fail
		}
	}

	/**
	 * Preprocesses messages to merge consecutive tool response messages into a single message.
	 * This is required for Gemini to correctly handle parallel tool call results.
	 */
	private preprocessMessages(messages: ChatMessage[]): ChatMessage[] {
		const processed: ChatMessage[] = [];
		let currentToolMessage: any = null;
		
		// Map to store tool_call_id -> function_name mapping from assistant messages
		const toolNameMap = new Map<string, string>();

		for (const msg of messages) {
			// Extract function names from assistant's tool calls
			if (msg.role === "assistant" && msg.tool_calls) {
				for (const call of msg.tool_calls) {
					if (call.id && call.function?.name) {
						toolNameMap.set(call.id, call.function.name);
					}
				}
			}

			if (msg.role === "tool") {
				// Resolve the real function name, fallback to ID if not found
				const realFunctionName = msg.tool_call_id ? (toolNameMap.get(msg.tool_call_id) || msg.tool_call_id) : "unknown_function";

				if (!currentToolMessage) {
					currentToolMessage = {
						role: "tool",
						_is_merged_tool: true,
						_tool_responses: [{ id: msg.tool_call_id, name: realFunctionName, content: msg.content }]
					};
				} else {
					currentToolMessage._tool_responses.push({ id: msg.tool_call_id, name: realFunctionName, content: msg.content });
				}
			} else {
				if (currentToolMessage) {
					processed.push(currentToolMessage);
					currentToolMessage = null;
				}
				processed.push(msg);
			}
		}
		if (currentToolMessage) processed.push(currentToolMessage);
		return processed;
	}

	/**
	 * Parses a server-sent event (SSE) stream from the Gemini API.
	 */
	private parseSSEPayload(payload: string): GeminiResponse | null {
		const trimmed = payload.trim();
		if (!trimmed || trimmed === "[DONE]") return null;
		try {
			return JSON.parse(trimmed) as GeminiResponse;
		} catch {
			throw new UpstreamApiError(
				502,
				"Upstream returned a malformed SSE event",
				undefined,
				{ code: "malformed_sse" }
			);
		}
	}

	private async *parseSSEStream(stream: ReadableStream<Uint8Array>): AsyncGenerator<GeminiResponse> {
		const reader = stream.pipeThrough(new TextDecoderStream()).getReader();
		let buffer = "";
		let objectBuffer = "";

		try {
			while (true) {
				const { done, value } = await reader.read();
				if (done) {
					const finalLine = buffer.trim();
					if (finalLine.startsWith("data:")) {
						objectBuffer += finalLine.substring(5).trimStart();
					}
					if (objectBuffer) {
						const parsed = this.parseSSEPayload(objectBuffer);
						if (parsed) yield parsed;
					}
					break;
				}

				buffer += value;
				const lines = buffer.split("\n");
				buffer = lines.pop() || ""; // Keep the last, possibly incomplete, line.

				for (const line of lines) {
					if (line.trim() === "") {
						if (objectBuffer) {
							const parsed = this.parseSSEPayload(objectBuffer);
							objectBuffer = "";
							if (parsed) yield parsed;
						}
					} else if (line.startsWith("data:")) {
						objectBuffer += line.substring(5).trimStart();
						if (objectBuffer.length > 16 * 1024 * 1024) {
							throw new UpstreamApiError(
								502,
								"Upstream SSE event exceeded the maximum supported size",
								undefined,
								{ code: "sse_event_too_large" }
							);
						}
					}
				}
			}
		} finally {
			reader.releaseLock();
		}
	}

	/**
	 * Converts a message to Gemini format, handling both text and image content.
	 */
	private async messageToGeminiFormat(msg: ChatMessage): Promise<GeminiFormattedMessage | GeminiFormattedMessage[]> {
		const role = msg.role === "assistant" ? "model" : "user";

		// Handle tool call results (tool role in OpenAI format)
		if (msg.role === "tool") {
			const parts: GeminiPart[] = [];

			if ((msg as any)._is_merged_tool && (msg as any)._tool_responses) {
				// Handle merged tool responses for parallel calls
				for (const response of (msg as any)._tool_responses) {
					parts.push({
						functionResponse: {
							id: response.id,
							name: response.name || "unknown_function",
							response: {
								result: typeof response.content === "string" ? response.content : JSON.stringify(response.content)
							}
						}
					});
				}
			} else {
				// Single tool response
				parts.push({
					functionResponse: {
						id: msg.tool_call_id,
						name: msg.tool_call_id || "unknown_function",
						response: {
							result: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)
						}
					}
				});
			}

			return { role: "user", parts };
		}

		// Handle assistant messages with tool calls
		if (msg.role === "assistant" && msg.tool_calls && msg.tool_calls.length > 0) {
			const parts: GeminiPart[] = [];

			// Add text content if present
			if (typeof msg.content === "string" && msg.content.trim()) {
				parts.push({ text: msg.content });
			}

			// Add function calls
			for (const toolCall of msg.tool_calls) {
				if (toolCall.type === "function") {
					parts.push({
						functionCall: {
							id: toolCall.id,
							name: toolCall.function.name,
							args: JSON.parse(toolCall.function.arguments)
						},
						thoughtSignature: "skip_thought_signature_validator"
					});
				}
			}

			return { role: "model", parts };
		}

		if (typeof msg.content === "string") {
			// Simple text message
			return {
				role,
				parts: [{ text: msg.content }]
			};
		}

		if (Array.isArray(msg.content)) {
			// Multimodal message with text and/or images/media
			const textParts: GeminiPart[] = [];
			const mediaParts: { part: GeminiPart; filename: string }[] = [];
			let mediaCount = 0;

			for (const content of msg.content) {
				if (content.type === "text") {
					textParts.push({ text: content.text });
				} else if (content.type === "image_url" && content.image_url) {
					mediaCount++;
					const imageUrl = content.image_url.url;

					const { isValid, error, mimeType } = validateContent("image_url", content);
					if (!isValid) throw new Error(`Invalid image: ${error}`);

					if (imageUrl.startsWith("data:")) {
						const [header, base64Data] = imageUrl.split(",");
						if (!base64Data) throw new Error("Invalid base64 data URL: missing data part.");
						const finalMimeType = header.split(":")[1].split(";")[0] || mimeType || "image/jpeg";
						const ext = finalMimeType.split("/")[1] || "jpg";

						mediaParts.push({
							part: { inlineData: { mimeType: finalMimeType, data: base64Data } },
							filename: `C:\\user_uploaded_image_${mediaCount}.${ext}`
						});
					} else {
						throw new Error("Only base64 image URLs are supported currently.");
					}
				} else if (content.type === "input_audio" && content.input_audio) {
					mediaCount++;
					const ext = content.input_audio.format.split("/")[1] || "wav";
					mediaParts.push({
						part: { inlineData: { mimeType: content.input_audio.format, data: content.input_audio.data } },
						filename: `C:\\user_uploaded_audio_${mediaCount}.${ext}`
					});
				} else if (content.type === "input_video" && content.input_video) {
					if (content.input_video.data && content.input_video.format) {
						mediaCount++;
						const ext = content.input_video.format.split("/")[1] || "mp4";
						const part: GeminiPart = {
							inlineData: { mimeType: content.input_video.format, data: content.input_video.data }
						};

						if (content.input_video.videoMetadata) {
							const { startOffset, endOffset, fps } = content.input_video.videoMetadata;
							if (startOffset || endOffset || fps) {
								part.videoMetadata = {};
								if (startOffset) part.videoMetadata.startOffset = startOffset;
								if (endOffset) part.videoMetadata.endOffset = endOffset;
								if (fps) part.videoMetadata.fps = fps;
							}
						}
						mediaParts.push({ part, filename: `C:\\user_uploaded_video_${mediaCount}.${ext}` });
					}
				} else if (content.type === "input_pdf" && content.input_pdf) {
					if (content.input_pdf.data) {
						mediaCount++;
						const { isValid, error } = validateContent("input_pdf", content);
						if (!isValid) throw new Error(`Invalid PDF: ${error}`);

						mediaParts.push({
							part: { inlineData: { mimeType: "application/pdf", data: content.input_pdf.data } },
							filename: `C:\\user_uploaded_document_${mediaCount}.pdf`
						});
					}
				}
			}

			if (mediaParts.length > 0) {
				const messages: GeminiFormattedMessage[] = [];
				const toolIdBase = crypto.randomUUID().replace(/-/g, "").substring(0, 8);
				
				// 1. Dummy User message to satisfy the API rule that conversation must start with 'user'
				messages.push({ role: "user", parts: [{ text: "I have uploaded some files. Please use your tools to view them." }] });
				
				// 2. Fake Assistant message with functionCall for each media file
				messages.push({
					role: "model",
					parts: mediaParts.map((media, index) => ({
						functionCall: {
							id: `${toolIdBase}${index}`,
							name: "view_file",
							args: { 
								AbsolutePath: media.filename,
								toolAction: "Viewing uploaded file",
								toolSummary: "File viewing"
							}
						},
						thoughtSignature: "skip_thought_signature_validator"
					}))
				});
				
				// 3. Fake User message with functionResponse AND actual user text
				const combinedUserParts: GeminiPart[] = mediaParts.map((media, index) => ({
					functionResponse: {
						id: `${toolIdBase}${index}`,
						name: "view_file",
						response: { output: "The following is the entire, complete content of the requested file." },
						parts: [media.part]
					}
				}));

				if (textParts.length > 0) {
					combinedUserParts.push(...textParts);
				} else {
					combinedUserParts.push({ text: "Please analyze the uploaded files." });
				}

				messages.push({
					role: "user",
					parts: combinedUserParts
				});
				
				return messages;
			}

			return { role, parts: textParts };
		}

		// Fallback for unexpected content format
		return {
			role,
			parts: [{ text: String(msg.content) }]
		};
	}

	/**
	 * Stream content from Gemini API.
	 */
	async *streamContent(
		modelId: string,
		systemPrompt: string,
		messages: ChatMessage[],
		options?: GeminiCompletionOptions
	): AsyncGenerator<StreamChunk> {
		await this.authManager.initializeAuth(modelId);
		await this.fetchDynamicModels();
		const projectId = await this.discoverProjectId();

		// Preprocess messages to handle parallel tool calls
		const preprocessedMessages = this.preprocessMessages(messages);
		const contentsPromises = preprocessedMessages.map((msg) => this.messageToGeminiFormat(msg));
		const contentsArrays = await Promise.all(contentsPromises);
		const contents = contentsArrays.flat();

		// Gemini API strictly requires that conversations start and end with the 'user' role.
		if (contents.length > 0) {
			if (contents[0].role === "model") {
				contents.unshift({ role: "user", parts: [{ text: "Begin conversation." }] });
			}
			if (contents[contents.length - 1].role === "model") {
				contents.push({ role: "user", parts: [{ text: "Continue." }] });
			}
		}

		// Check if this is a thinking model and which thinking mode to use
		const isThinkingModel = geminiCliModels[modelId]?.thinking || false;
		const isRealThinkingEnabled = this.env.ENABLE_REAL_THINKING === "true";
		const isFakeThinkingEnabled = this.env.ENABLE_FAKE_THINKING === "true";
		const streamThinkingAsContent = this.env.STREAM_THINKING_AS_CONTENT === "true";
		const includeReasoning = options?.includeReasoning || false;
		const showReasoning = options?.showReasoning ?? true;

		// Native tools integration
		const nativeToolsManager = new NativeToolsManager(this.env);
		const nativeToolsParams = this.extractNativeToolsParams(options as Record<string, unknown>);
		const toolConfig = nativeToolsManager.determineToolConfiguration(options?.tools || [], nativeToolsParams, modelId);

		// Configure request based on tool strategy
		const { tools, toolConfig: finalToolConfig } = GenerationConfigValidator.createFinalToolConfiguration(
			toolConfig,
			options
		);

		// For thinking models with fake thinking (fallback when real thinking is not enabled or not requested)
		let needsThinkingClose = false;
		if (showReasoning && isThinkingModel && isFakeThinkingEnabled && !includeReasoning) {
			yield* this.generateReasoningOutput(messages, streamThinkingAsContent);
			needsThinkingClose = streamThinkingAsContent; // Only need to close if we streamed as content
		}

		let finalModelEnum = "MODEL_PLACEHOLDER_M0";
		let customThinkingBudget = 1000;
		let finalModelId = modelId;
		let hasDynamicThinkingBudget = false;

		if (this.dynamicModelMap) {
			const dModel = this.dynamicModelMap[modelId];
			if (dModel) {
				if (dModel.model) finalModelEnum = dModel.model;
				if (dModel.thinkingBudget !== undefined) {
					customThinkingBudget = dModel.thinkingBudget;
					hasDynamicThinkingBudget = true;
				}
			} else {
				for (const [key, value] of Object.entries(this.dynamicModelMap)) {
					if ((value as any).displayName === modelId) {
						finalModelId = key;
						if ((value as any).model) finalModelEnum = (value as any).model;
						if ((value as any).thinkingBudget !== undefined) {
							customThinkingBudget = (value as any).thinkingBudget;
							hasDynamicThinkingBudget = true;
						}
						// Skip deprecated models like M37 and keep looking for the active one (M16)
						if (finalModelEnum !== "MODEL_PLACEHOLDER_M37") {
							break;
						}
					}
				}
			}
		} else {
			// Hardcoded fallbacks based on recent Antigravity 2.0 reverse engineering
			// Match both internal keys and IDE displayNames
			if (modelId === "claude-opus-4-6-thinking" || modelId === "Claude Opus 4.6 (Thinking)") {
				finalModelId = "claude-opus-4-6-thinking";
				finalModelEnum = "MODEL_PLACEHOLDER_M26";
				customThinkingBudget = 1024;
				hasDynamicThinkingBudget = true;
			} else if (modelId === "claude-sonnet-4-6-thinking" || modelId === "Claude Sonnet 4.6 (Thinking)") {
				finalModelId = "claude-sonnet-4-6-thinking";
				finalModelEnum = "MODEL_PLACEHOLDER_M35";
				customThinkingBudget = 1024;
				hasDynamicThinkingBudget = true;
			} else if (modelId === "gemini-3.5-flash-extra-low" || modelId === "Gemini 3.5 Flash (Low)") {
				finalModelId = "gemini-3.5-flash-extra-low";
				finalModelEnum = "MODEL_PLACEHOLDER_M187";
				customThinkingBudget = 4000; // Matches dump budget for Low
				hasDynamicThinkingBudget = true;
			} else if (modelId === "gemini-3.5-flash-medium" || modelId === "Gemini 3.5 Flash (Medium)") {
				finalModelId = "gemini-3.5-flash-low"; // Key from dynamic dump
				finalModelEnum = "MODEL_PLACEHOLDER_M20";
				customThinkingBudget = 4000;
				hasDynamicThinkingBudget = true;
			} else if (modelId === "gemini-3.5-flash-high" || modelId === "Gemini 3.5 Flash (High)") {
				finalModelId = "gemini-3-flash-agent";
				finalModelEnum = "MODEL_PLACEHOLDER_M132";
				customThinkingBudget = 10000;
				hasDynamicThinkingBudget = true;
			} else if (modelId === "gpt-oss-120b-medium" || modelId === "GPT-OSS 120B (Medium)") {
				finalModelId = "gpt-oss-120b-medium";
				finalModelEnum = "MODEL_OPENAI_GPT_OSS_120B_MEDIUM";
				customThinkingBudget = 8192;
			}
		}

		const isNewModel = finalModelId.includes("3.6") || finalModelId.includes("3.7") || finalModelEnum === "MODEL_PLACEHOLDER_M73";
		const defaultMaxTokens = geminiCliModels[finalModelId]?.maxTokens ?? (isNewModel ? 65536 : 16384);
		const generationConfig = GenerationConfigValidator.createValidatedConfig(
			finalModelId,
			options,
			isRealThinkingEnabled,
			includeReasoning,
			selectSchemaProtocol(finalModelId, this.env.GEMINI_RESPONSE_SCHEMA_MODE)
		);
		const finalGenerationConfig: Record<string, unknown> = {
			...generationConfig,
			maxOutputTokens: Math.min(options?.max_completion_tokens ?? options?.max_tokens ?? defaultMaxTokens, defaultMaxTokens)
		};
		const requestedReasoningEffort =
			options?.reasoning_effort || options?.extra_body?.reasoning_effort || options?.model_params?.reasoning_effort;
		const hasExplicitThinkingControl = options?.thinking_budget !== undefined || requestedReasoningEffort !== undefined;
		const supportsThinking = geminiCliModels[finalModelId]?.thinking === true || hasDynamicThinkingBudget;
		const fallbackThinkingBudget =
			options?.thinking_budget ??
			(requestedReasoningEffort && GenerationConfigValidator.isValidEffortLevel(requestedReasoningEffort)
				? GenerationConfigValidator.mapEffortToThinkingBudget(requestedReasoningEffort, finalModelId)
				: customThinkingBudget);

		if (!includeReasoning || !isRealThinkingEnabled || !supportsThinking) {
			delete finalGenerationConfig.thinkingConfig;
		} else if (!hasExplicitThinkingControl || finalGenerationConfig.thinkingConfig === undefined) {
			finalGenerationConfig.thinkingConfig = {
				includeThoughts: true,
				thinkingBudget: fallbackThinkingBudget
			};
		}

		const conversationId = crypto.randomUUID();
		const trajectoryId = crypto.randomUUID();

		const streamRequest: {
			model: string;
			project?: string;
			userAgent: string;
			requestId: string;
			requestType: string;
			request: {
				contents: unknown;
				systemInstruction?: unknown;
				generationConfig: unknown;
				tools?: unknown;
				toolConfig?: unknown;
				safetySettings?: unknown;
				labels: Record<string, string>;
				sessionId: string;
			};
		} = {
			model: finalModelId,
			userAgent: "antigravity",
			requestId: `agent/${conversationId}/${Date.now()}/${trajectoryId}/2`,
			requestType: "agent",
			request: {
				contents: contents,
				generationConfig: finalGenerationConfig,
				labels: {
					"last_step_index": "1",
					"model_enum": finalModelEnum,
					"request_id": `${trajectoryId}-0`,
					"trajectory_id": trajectoryId,
					"used_claude": finalModelId.includes("claude") ? "true" : "false",
					"used_claude_conservative": "false",
					"used_non_gemini_model": "false"
				},
				sessionId: String(Math.floor(Math.random() * Number.MAX_SAFE_INTEGER))
			}
		};

		let finalTools = tools ? [...(tools as any[])] : [];
		let resolvedToolConfig = finalToolConfig;

		// If we gaslighted the model with view_file tool calls for images, we MUST declare the tool
		const hasGaslightedTools = contents.some((msg: any) => 
			msg.parts?.some((p: any) => p.functionCall?.name === "view_file" || p.functionResponse?.name === "view_file")
		);

		if (hasGaslightedTools) {
			const viewFileDecl = {
				name: "view_file",
				description: "View the contents of a file from the local filesystem. This tool supports text files and following binary files: image, pdf, video, audio.",
				parameters: {
					type: "OBJECT",
					properties: {
						AbsolutePath: {
							type: "STRING",
							description: "Path to file to view. Must be an absolute path."
						},
						toolAction: {
							type: "STRING",
							description: "Brief 2-5 word summary of what this tool is doing. Capitalize like a sentence."
						},
						toolSummary: {
							type: "STRING",
							description: "Brief 2-5 word noun phrase describing what this tool call is about. Capitalize like a sentence."
						}
					},
					required: [
						"AbsolutePath",
						"toolSummary",
						"toolAction"
					]
				}
			};

			const existingFuncDeclObj = finalTools.find((t: any) => Array.isArray(t.functionDeclarations));
			if (existingFuncDeclObj) {
				if (!existingFuncDeclObj.functionDeclarations.some((f: any) => f.name === "view_file")) {
					existingFuncDeclObj.functionDeclarations.push(viewFileDecl);
				}
			} else {
				finalTools.push({ functionDeclarations: [viewFileDecl] });
			}

			resolvedToolConfig = resolvedToolConfig || { functionCallingConfig: { mode: "VALIDATED" } };
		}

		if (finalTools.length > 0) {
			streamRequest.request.tools = finalTools;
			if (resolvedToolConfig) {
				streamRequest.request.toolConfig = resolvedToolConfig;
			}
		}

		if (projectId) {
			streamRequest.project = projectId;
		} else {
			// Force the default shadow project if missing, because Pro models with projectRestrictions 
			// will silently close the 200 OK stream if the project is omitted.
			streamRequest.project = "carbon-reporter-5sf6z";
		}

		if (systemPrompt) {
			streamRequest.request.systemInstruction = {
				role: "user",
				parts: [{ text: systemPrompt }]
			};
		}

		const safetySettings = GenerationConfigValidator.createSafetySettings(this.env);
		if (safetySettings.length > 0) {
			streamRequest.request.safetySettings = safetySettings;
		}

		yield* this.performStreamRequest(
			streamRequest,
			needsThinkingClose,
			false,
			includeReasoning && streamThinkingAsContent,
			showReasoning,
			modelId,
			nativeToolsManager,
			0,
			options?.signal
		);
	}

	/**
	 * Generates reasoning output for thinking models.
	 */
	private async *generateReasoningOutput(
		messages: ChatMessage[],
		streamAsContent: boolean = false
	): AsyncGenerator<StreamChunk> {
		// Get the last user message to understand what the model should think about
		const lastUserMessage = messages.filter((msg) => msg.role === "user").pop();
		let userContent = "";

		if (lastUserMessage) {
			if (typeof lastUserMessage.content === "string") {
				userContent = lastUserMessage.content;
			} else if (Array.isArray(lastUserMessage.content)) {
				userContent = lastUserMessage.content
					.filter(isTextContent)
					.map((c) => c.text)
					.join(" ");
			}
		}

		// Generate reasoning text based on the user's question using constants
		const requestPreview = userContent.substring(0, 100) + (userContent.length > 100 ? "..." : "");

		if (streamAsContent) {
			// DeepSeek R1 style: stream thinking as content with <think> tags
			yield {
				type: "thinking_content",
				data: "<think>\n"
			};

			// Add a small delay after opening tag
			await new Promise((resolve) => setTimeout(resolve, REASONING_CHUNK_DELAY)); // Stream reasoning content in smaller chunks for more realistic streaming
			const reasoningTexts = REASONING_MESSAGES.map((msg) => msg.replace("{requestPreview}", requestPreview));
			const fullReasoningText = reasoningTexts.join("");

			// Split into smaller chunks for more realistic streaming
			// Try to split on word boundaries when possible for better readability
			const chunks: string[] = [];
			let remainingText = fullReasoningText;

			while (remainingText.length > 0) {
				if (remainingText.length <= THINKING_CONTENT_CHUNK_SIZE) {
					chunks.push(remainingText);
					break;
				}

				// Try to find a good break point (space, newline, punctuation)
				let chunkEnd = THINKING_CONTENT_CHUNK_SIZE;
				const searchSpace = remainingText.substring(0, chunkEnd + 10); // Look a bit ahead
				const goodBreaks = [" ", "\n", ".", ",", "!", "?", ";", ":"];

				for (const breakChar of goodBreaks) {
					const lastBreak = searchSpace.lastIndexOf(breakChar);
					if (lastBreak > THINKING_CONTENT_CHUNK_SIZE * 0.7) {
						// Don't make chunks too small
						chunkEnd = lastBreak + 1;
						break;
					}
				}

				chunks.push(remainingText.substring(0, chunkEnd));
				remainingText = remainingText.substring(chunkEnd);
			}

			for (const chunk of chunks) {
				yield {
					type: "thinking_content",
					data: chunk
				};

				// Add small delay between chunks
				await new Promise((resolve) => setTimeout(resolve, 50));
			}

			// Note: We don't close the thinking tag here - it will be closed when real content starts
		} else {
			// Original mode: stream as reasoning field
			const reasoningTexts = REASONING_MESSAGES.map((msg) => msg.replace("{requestPreview}", requestPreview));

			// Stream the reasoning text in chunks
			for (const reasoningText of reasoningTexts) {
				const reasoningData: ReasoningData = { reasoning: reasoningText };
				yield {
					type: "reasoning",
					data: reasoningData
				};

				// Add a small delay to simulate thinking time
				await new Promise((resolve) => setTimeout(resolve, REASONING_CHUNK_DELAY));
			}
		}
	}

	/**
	 * Performs the actual stream request with retry logic for 401 errors and auto model switching for rate limits.
	 */
	private async *performStreamRequest(
		streamRequest: unknown,
		needsThinkingClose: boolean = false,
		isRetry: boolean = false,
		realThinkingAsContent: boolean = false,
		showReasoning: boolean = true, // Add showReasoning parameter, default to true
		originalModel?: string,
		nativeToolsManager?: NativeToolsManager,
		retryCount: number = 0,
		signal?: AbortSignal
	): AsyncGenerator<StreamChunk> {
		const citationsProcessor = new CitationsProcessor(this.env);
		// Keep the abort scope alive for the whole SSE body so client
		// disconnects immediately cancel the upstream generation.
		const abortScope = createAbortScope(this.generationTimeoutMs(), signal);
		let response: Response;
		try {
			response = await fetch(
				`${CODE_ASSIST_ENDPOINT}/${CODE_ASSIST_API_VERSION}:streamGenerateContent?alt=sse`,
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						"Authorization": `Bearer ${this.authManager.getAccessToken()}`,
						"User-Agent": "antigravity/cli/1.0.2 windows/amd64"
					},
					body: JSON.stringify(streamRequest),
					signal: abortScope.signal
				}
			);
		} catch (error: unknown) {
			abortScope.cleanup();
			if (signal?.aborted) throw error;
			if (isAbortError(error)) {
				throw new UpstreamApiError(
					504,
					"Upstream generation timed out",
					undefined,
					{ code: "upstream_timeout" }
				);
			}
			throw error;
		}

		if (!response.ok) {
			const errorText = await response.text();
			abortScope.cleanup();
			const requestId = getUpstreamRequestId(response.headers, errorText);
			const retryAfterMs = parseRetryAfterMs(response.headers, errorText);

			if (response.status === 401 && !isRetry) {
				console.log("Got 401 error in stream request, refreshing credentials once...");
				await this.authManager.clearTokenCache();
				await this.authManager.initializeAuth(originalModel);
				yield* this.performStreamRequest(
					streamRequest,
					needsThinkingClose,
					true,
					realThinkingAsContent,
					showReasoning,
					originalModel,
					nativeToolsManager,
					retryCount,
					signal
				); // Retry once
				return;
			}

			// Handle rate limiting with auto model switching
			if (this.autoSwitchHelper.isRateLimitStatus(response.status) && originalModel) {
				// Parse the error to see if it's a hard quota exhaustion
				let isQuotaExhausted = false;
				let quotaResetDelaySeconds = 300; // default 5 mins

				try {
					const errorJson = JSON.parse(errorText);
					const details = errorJson?.error?.details || [];
					for (const detail of details) {
						if (detail.reason === "QUOTA_EXHAUSTED") {
							isQuotaExhausted = true;
							if (detail.metadata?.quotaResetDelay) {
								// e.g., "157h14m28.152758s"
								const delayStr = detail.metadata.quotaResetDelay;
								let totalSeconds = 0;
								
								const hoursMatch = delayStr.match(/(\d+)h/);
								const minutesMatch = delayStr.match(/(\d+)m/);
								const secondsMatch = delayStr.match(/([\d.]+)s/);
								
								if (hoursMatch) totalSeconds += parseInt(hoursMatch[1]) * 3600;
								if (minutesMatch) totalSeconds += parseInt(minutesMatch[1]) * 60;
								if (secondsMatch) totalSeconds += parseFloat(secondsMatch[1]);
								
								if (totalSeconds > 0) quotaResetDelaySeconds = totalSeconds;
							}
							break;
						}
					}
				} catch (e) {
					// Fallback if parsing fails
				}

				if (isQuotaExhausted) {
					if (isRetry) {
						// If we already hot-swapped and the new account is ALSO exhausted, the pool is dead.
						console.error(`[GeminiAPI] FATAL: All accounts in pool exhausted for ${originalModel}.`);
						throw new UpstreamApiError(
							429,
							"All configured accounts have exhausted their quotas for this model.",
							safeLogText(errorText, 2000),
							{ requestId, retryAfterMs, code: "quota_exhausted" }
						);
					}

					console.log(`[GeminiAPI] QUOTA EXHAUSTED for model ${originalModel}. Reset in ${quotaResetDelaySeconds}s.`);
					// Tell AuthManager to ban this account for this provider
					await this.authManager.markAccountExhausted(originalModel, quotaResetDelaySeconds);
					
					// Hot-swap retry: call initializeAuth again to pick a new account, then retry
					// This happens silently without injecting text into the user's chat.
					console.log(`[GeminiAPI] Hot-swapping to a new account from the pool silently...`);

					await this.authManager.initializeAuth(originalModel);
					
					yield* this.performStreamRequest(
						streamRequest,
						needsThinkingClose,
						true, // count as retry so we don't infinitely loop on 401s
						realThinkingAsContent,
						showReasoning,
						originalModel,
						nativeToolsManager,
						retryCount,
						signal
					);
					return;
				}

				// If not a hard quota exhaustion, attempt auto model switching
				const fallbackModel = this.autoSwitchHelper.getFallbackModel(originalModel);
				if (fallbackModel && this.autoSwitchHelper.isEnabled() && !isRetry) {
					console.log(
						`Got ${response.status} error for model ${originalModel}, switching to fallback model: ${fallbackModel}`
					);

					// Create new request with fallback model
					const fallbackRequest = {
						...(streamRequest as Record<string, unknown>),
						model: fallbackModel
					};

					// Add a notification chunk about the model switch
					yield {
						type: "text",
						data: this.autoSwitchHelper.createSwitchNotification(originalModel, fallbackModel)
					};

					yield* this.performStreamRequest(
						fallbackRequest,
						needsThinkingClose,
						true,
						realThinkingAsContent,
						showReasoning,
						originalModel,
						nativeToolsManager,
						0, // Reset retry count for the new model
						signal
					);
					return;
				}
			}

			// Handle transient 429/499 rate limits with delay (Agent/Tool call rapid requests)
			if ((response.status === 429 || response.status === 499) && retryCount < MAX_RATE_LIMIT_RETRIES) {
				const delayMs = Math.min(
					retryAfterMs ?? Math.pow(2, retryCount) * 1000,
					MAX_RETRY_DELAY_MS
				);
				console.log(
					`[GeminiAPI] Rate limited (${response.status}). Waiting ${delayMs}ms before retrying (Attempt ${retryCount + 1}/${MAX_RATE_LIMIT_RETRIES})...`
				);
				await sleep(delayMs);
				
				yield* this.performStreamRequest(
					streamRequest,
					needsThinkingClose,
					isRetry,
					realThinkingAsContent,
					showReasoning,
					originalModel,
					nativeToolsManager,
					retryCount + 1,
					signal
				);
				return;
			}

			console.error(
				`[GeminiAPI] Stream request failed: ${response.status}`,
				safeLogText(errorText, 2000)
			);
			throw new UpstreamApiError(
				response.status,
				`Upstream request failed with status ${response.status}`,
				safeLogText(errorText, 2000),
				{ requestId, retryAfterMs }
			);
		}

		if (!response.body) {
			abortScope.cleanup();
			throw new UpstreamApiError(502, "Upstream response has no body", undefined, {
				code: "empty_upstream_body"
			});
		}

		let hasClosedThinking = false;
		let hasStartedThinking = false;
		let isThinking = false; // Track if we are currently in a thinking block

		try {
		for await (const jsonData of this.parseSSEStream(response.body)) {
			if (jsonData.error) {
				const upstreamStatus =
					typeof jsonData.error.code === "number" && jsonData.error.code >= 400
						? jsonData.error.code
						: 502;
				throw new UpstreamApiError(
					upstreamStatus,
					jsonData.error.message || "Upstream stream returned an error",
					undefined,
					{
						code: jsonData.error.status || "upstream_stream_error",
						requestId: jsonData.traceId
					}
				);
			}

			const candidate = jsonData.response?.candidates?.[0];

			if (candidate?.content?.parts) {
				for (const part of candidate.content.parts as GeminiPart[]) {
					// Handle real thinking content from Gemini
					if (part.thought === true && part.text) {
						if (showReasoning) {
							const thinkingText = part.text;

							if (realThinkingAsContent) {
								// Stream as content with <think> tags (DeepSeek R1 style)
								if (!hasStartedThinking) {
									yield {
										type: "thinking_content",
										data: "<think>\n"
									};
									hasStartedThinking = true;
								}

								yield {
									type: "thinking_content",
									data: thinkingText
								};
							} else {
								// Stream as separate reasoning field
								yield {
									type: "real_thinking",
									data: thinkingText
								};
							}
						}
						continue; // Skip yielding this part as text if it's a thought part
					}

					// Check if text content contains <think> tags (e.g. for models that don't use part.thought)
					if (part.text && (part.text.includes("<think>") || isThinking)) {
						let text = part.text;
						
						if (part.text.includes("<think>")) {
							isThinking = true;
						}

						if (showReasoning) {
							if (realThinkingAsContent) {
								// Extract thinking content and convert to our format
								const thinkingMatch = text.match(/<think>(.*?)<\/think>/s);
								if (thinkingMatch) {
									if (!hasStartedThinking) {
										yield {
											type: "thinking_content",
											data: "<think>\n"
										};
										hasStartedThinking = true;
									}

									yield {
										type: "thinking_content",
										data: thinkingMatch[1]
									};
									isThinking = false;
								} else if (text.includes("<think>")) {
									// Start of thinking but no end tag yet
									if (!hasStartedThinking) {
										yield {
											type: "thinking_content",
											data: "<think>\n"
										};
										hasStartedThinking = true;
									}
									yield {
										type: "thinking_content",
										data: text.split("<think>")[1]
									};
								} else {
									// Continuation of thinking
									yield {
										type: "thinking_content",
										data: text
									};
								}

								// Extract any non-thinking content if the block closed in this chunk
								if (text.includes("</think>")) {
									isThinking = false;
									const nonThinkingContent = text.split("</think>")[1].trim();
									if (nonThinkingContent) {
										if (hasStartedThinking && !hasClosedThinking) {
											yield {
												type: "thinking_content",
										data: "\n</think>\n\n"
											};
											hasClosedThinking = true;
										}
										yield { type: "text", data: nonThinkingContent };
									}
								}
							} else {
								// Stream thinking as separate reasoning field
								const thinkingMatch = text.match(/<think>(.*?)<\/think>/s);
								if (thinkingMatch) {
									yield {
										type: "real_thinking",
										data: thinkingMatch[1]
									};
									isThinking = false;
								} else if (text.includes("<think>")) {
									yield {
										type: "real_thinking",
										data: text.split("<think>")[1]
									};
								} else {
									yield {
										type: "real_thinking",
										data: text
									};
								}

								// Stream non-thinking content as regular text if block closed
								if (text.includes("</think>")) {
									isThinking = false;
									const nonThinkingContent = text.split("</think>")[1].trim();
									if (nonThinkingContent) {
										yield { type: "text", data: nonThinkingContent };
									}
								}
							}
						} else {
							// If showReasoning is false, we still need to check for the closing tag
							if (text.includes("</think>")) {
								isThinking = false;
								const nonThinkingContent = text.split("</think>")[1].trim();
								if (nonThinkingContent) {
									yield { type: "text", data: nonThinkingContent };
								}
							}
						}
						continue;
					}

					// Handle regular content
					if (part.text) {
						// Close thinking tag before first real content if needed
						if ((needsThinkingClose || (realThinkingAsContent && hasStartedThinking)) && !hasClosedThinking) {
							yield {
								type: "thinking_content",
								data: "\n</think>\n\n"
							};
							hasClosedThinking = true;
						}

						let processedText = part.text;
						if (nativeToolsManager) {
							processedText = citationsProcessor.processChunk(
								part.text,
								jsonData.response?.candidates?.[0]?.groundingMetadata
							);
						}
						yield { type: "text", data: processedText };
					}
					// Handle function calls from Gemini
					else if (part.functionCall) {
						// Close thinking tag before function call if needed
						if ((needsThinkingClose || (realThinkingAsContent && hasStartedThinking)) && !hasClosedThinking) {
							yield {
								type: "thinking_content",
									data: "\n</think>\n\n"
							};
							hasClosedThinking = true;
						}

						const functionCallData: GeminiFunctionCall = {
							id: part.functionCall.id,
							name: part.functionCall.name,
							args: part.functionCall.args ?? {}
						};

						yield {
							type: "tool_code",
							data: functionCallData
						};
					}
					// Note: Skipping unknown part structures
				}
			}

			if (candidate?.finishReason) {
				yield { type: "finish", data: { reason: mapGeminiFinishReason(candidate.finishReason) } };
			} else if (jsonData.response?.promptFeedback?.blockReason) {
				const mapped = mapGeminiFinishReason(
					jsonData.response.promptFeedback.blockReason
				);
				yield {
					type: "finish",
					data: { reason: mapped === "stop" ? "content_filter" : mapped }
				};
			}

			if (jsonData.response?.usageMetadata) {
				const usage = jsonData.response.usageMetadata;
				const usageData: UsageData = {
					inputTokens: usage.promptTokenCount || 0,
					outputTokens: (usage.candidatesTokenCount || 0) + (usage.thoughtsTokenCount || 0),
					reasoningTokens: usage.thoughtsTokenCount || undefined,
					cachedTokens: usage.cachedContentTokenCount || undefined,
					toolUseTokens: usage.toolUsePromptTokenCount || undefined
				};
				yield {
					type: "usage",
					data: usageData
				};
			}
		}
		} catch (error: unknown) {
			if (!signal?.aborted && abortScope.didTimeout()) {
				throw new UpstreamApiError(
					504,
					"Upstream generation timed out",
					undefined,
					{ code: "upstream_timeout" }
				);
			}
			throw error;
		} finally {
			abortScope.cleanup();
		}
	}

	/**
	 * Get a complete response from Gemini API (non-streaming).
	 */
	async getCompletion(
		modelId: string,
		systemPrompt: string,
		messages: ChatMessage[],
		options?: GeminiCompletionOptions
	): Promise<{
		content: string;
		usage?: UsageData;
		tool_calls?: Array<{ id: string; type: "function"; function: { name: string; arguments: string } }>;
		reasoning?: string;
		reasoning_content?: string;
		finish_reason?: ChatCompletionFinishReason;
	}> {
		try {
			let content = "";
			let reasoningContent = "";
			let usage: UsageData | undefined;
			let finishReason: ChatCompletionFinishReason | undefined;
			const tool_calls: Array<{ id: string; type: "function"; function: { name: string; arguments: string } }> = [];

			// Collect all chunks from the stream. Thinking/reasoning chunks are
			// kept separate from assistant content.
			for await (const chunk of this.streamContent(modelId, systemPrompt, messages, options)) {
				if (chunk.type === "text" && typeof chunk.data === "string") {
					content += chunk.data;
				} else if (
					(chunk.type === "real_thinking" || chunk.type === "thinking_content") &&
					typeof chunk.data === "string"
				) {
					reasoningContent += chunk.data;
				} else if (chunk.type === "reasoning" && typeof chunk.data === "object" && chunk.data !== null) {
					const reasoningData = chunk.data as ReasoningData;
					if (typeof reasoningData.reasoning === "string") {
						reasoningContent += reasoningData.reasoning;
					}
				} else if (chunk.type === "usage" && typeof chunk.data === "object") {
					usage = chunk.data as UsageData;
				} else if (chunk.type === "tool_code" && typeof chunk.data === "object") {
					const toolData = chunk.data as GeminiFunctionCall;
					tool_calls.push({
						id: toolData.id || `call_${crypto.randomUUID()}`,
						type: "function",
						function: {
							name: toolData.name,
							arguments: JSON.stringify(toolData.args ?? {})
						}
					});
				} else if (chunk.type === "finish" && typeof chunk.data === "object" && "reason" in chunk.data) {
					finishReason = chunk.data.reason;
				}
			}

			// Clean any inline <think> tags from content in non-streaming mode.
			if (content.includes("<think>")) {
				const thinkMatches = Array.from(content.matchAll(/<think>([\s\S]*?)<\/think>/g));
				if (thinkMatches.length > 0 && !reasoningContent) {
					reasoningContent = thinkMatches.map((match) => match[1].trim()).join("\n\n");
				}
				content = content.replace(/<think>[\s\S]*?<\/think>\s*/g, "").trim();
			}

			// Also clean any unclosed <think> or leftover tags.
			if (content.includes("<think>")) {
				const parts = content.split("<think>");
				content = parts[0].trim();
				if (!reasoningContent && parts[1]) {
					reasoningContent = parts[1].replace(/<\/think>/g, "").trim();
				}
			}

			const resolvedFinishReason =
				finishReason === "length" || finishReason === "content_filter"
					? finishReason
					: tool_calls.length > 0
						? "tool_calls"
						: finishReason ?? "stop";

			const cleanedReasoning = reasoningContent
				.replace(/<\/?think>/gi, "")
				.trim();

			return {
				content,
				reasoning: cleanedReasoning || undefined,
				reasoning_content: cleanedReasoning || undefined,
				usage,
				tool_calls: tool_calls.length > 0 ? tool_calls : undefined,
				finish_reason: resolvedFinishReason
			};
		} catch (error: unknown) {
			// Handle rate limiting for non-streaming requests
			if (this.autoSwitchHelper.isRateLimitError(error)) {
				const fallbackResult = await this.autoSwitchHelper.handleNonStreamingFallback(
					modelId,
					systemPrompt,
					messages,
					options,
					this.streamContent.bind(this)
				);
				if (fallbackResult) {
					return fallbackResult;
				}
			}

			// Re-throw if not a rate limit error or fallback not available
			throw error;
		}
	}

	private extractNativeToolsParams(options?: Record<string, unknown>): NativeToolsRequestParams {
		return {
			enableSearch: this.extractBooleanParam(options, "enable_search"),
			enableUrlContext: this.extractBooleanParam(options, "enable_url_context"),
			enableNativeTools: this.extractBooleanParam(options, "enable_native_tools"),
			nativeToolsPriority: this.extractStringParam(
				options,
				"native_tools_priority",
				(v): v is "native" | "custom" | "mixed" => ["native", "custom", "mixed"].includes(v)
			)
		};
	}

	private extractBooleanParam(options: Record<string, unknown> | undefined, key: string): boolean | undefined {
		const value =
			options?.[key] ??
			(options?.extra_body as Record<string, unknown>)?.[key] ??
			(options?.model_params as Record<string, unknown>)?.[key];
		return typeof value === "boolean" ? value : undefined;
	}

	private extractStringParam<T extends string>(
		options: Record<string, unknown> | undefined,
		key: string,
		guard: (v: string) => v is T
	): T | undefined {
		const value =
			options?.[key] ??
			(options?.extra_body as Record<string, unknown>)?.[key] ??
			(options?.model_params as Record<string, unknown>)?.[key];
		if (typeof value === "string" && guard(value)) {
			return value;
		}
		return undefined;
	}
}
