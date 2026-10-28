import { Env, OAuth2Credentials, AccountData } from "./types";
import {
	CODE_ASSIST_ENDPOINT,
	CODE_ASSIST_API_VERSION,
	OAUTH_CLIENT_ID,
	OAUTH_CLIENT_SECRET,
	OAUTH_REFRESH_URL,
	TOKEN_BUFFER_TIME,
	KV_TOKEN_KEY,
	TOKEN_REFRESH_TIMEOUT_MS,
	TOKEN_REFRESH_MAX_ATTEMPTS,
	MAX_RETRY_DELAY_MS
} from "./config";
import {
	envTimeoutMs,
	fetchTextWithTimeout,
	getUpstreamRequestId,
	parseRetryAfterMs,
	sleep
} from "./utils/http";
import { safeLogText } from "./utils/redact";
import { UpstreamApiError } from "./helpers/openai-compat";

interface TokenRefreshResponse {
	access_token: string;
	expires_in: number;
}

interface CallEndpointOptions {
	isRetry?: boolean;
	timeoutMs?: number;
	signal?: AbortSignal;
}

export class AuthError extends Error {
	constructor(
		public readonly status: number,
		public readonly code: string,
		message: string
	) {
		super(message);
		this.name = "AuthError";
	}
}

let legacyMemoryToken:
	| { refreshToken: string; accessToken: string; expiryDate: number }
	| undefined;
let legacyRefreshInFlight: Promise<TokenRefreshResponse> | undefined;
let legacyForceRefresh = false;
const accountRefreshInFlight = new Map<string, Promise<void>>();

export class AuthManager {
	private env: Env;
	private currentAccountId: string | null = null;
	private currentAccessToken: string | null = null;

	constructor(env: Env) {
		this.env = env;
	}

	public getCurrentAccountId(): string | null {
		return this.currentAccountId;
	}

	private tokenRefreshTimeoutMs(): number {
		return envTimeoutMs(
			this.env as unknown as Record<string, unknown>,
			"GEMINI_TOKEN_REFRESH_TIMEOUT_MS",
			TOKEN_REFRESH_TIMEOUT_MS
		);
	}

	/**
	 * Determines which provider pool to check for quota exhaustion.
	 */
	private getProviderForModel(modelId: string): "gemini" | "partner" {
		const lower = modelId.toLowerCase();
		if (lower.includes("claude") || lower.includes("gpt")) {
			return "partner";
		}
		return "gemini";
	}

	/**
	 * Find an available account from the KV pool that isn't currently rate-limited for the requested provider.
	 * Implements Randomized Round-Robin with Priority routing.
	 */
	public async getAvailableAccount(modelId: string): Promise<string> {
		const provider = this.getProviderForModel(modelId);
		const now = Date.now();

		if (this.env.ACCOUNTS_KV) {
			try {
				const keys = await this.env.ACCOUNTS_KV.list();
				const healthyAccountsByPriority = new Map<number, string[]>();
				let fallbackAccountId: string | null = null;

				for (const key of keys.keys) {
					const accountData = await this.env.ACCOUNTS_KV.get<AccountData>(key.name, "json");
					if (!accountData || accountData.is_invalid) continue;

					if (!fallbackAccountId) {
						fallbackAccountId = key.name;
					}

					const exhaustedUntil = provider === "gemini" ? accountData.exhausted_gemini_until : accountData.exhausted_partner_until;

					if (!exhaustedUntil || exhaustedUntil < now) {
						// Healthy account! Group it by priority (defaulting to 100)
						const priority = accountData.priority !== undefined ? accountData.priority : 100;
						if (!healthyAccountsByPriority.has(priority)) {
							healthyAccountsByPriority.set(priority, []);
						}
						healthyAccountsByPriority.get(priority)!.push(key.name);
					}
				}

				if (healthyAccountsByPriority.size > 0) {
					// Find the group with the lowest priority number (e.g., 1 goes before 100)
					const priorities = Array.from(healthyAccountsByPriority.keys()).sort((a, b) => a - b);
					const lowestPriority = priorities[0];
					const bestAccounts = healthyAccountsByPriority.get(lowestPriority)!;
					
					// Randomly select one account from the best priority group to distribute load (Serverless Round Robin)
					const selectedAccountId = bestAccounts[Math.floor(Math.random() * bestAccounts.length)];
					
					console.log(`[MultiAuth] Selected healthy account: ${selectedAccountId} (Priority: ${lowestPriority}) for provider ${provider}`);
					return selectedAccountId;
				}

				if (fallbackAccountId) {
					console.warn(`[MultiAuth] ALL accounts exhausted for provider ${provider}. Falling back to ${fallbackAccountId} and hoping for the best.`);
					return fallbackAccountId;
				}
				
				console.warn("[MultiAuth] ACCOUNTS_KV is bound but empty. Falling back to GCP_SERVICE_ACCOUNT env var.");
			} catch (e) {
				console.error("[MultiAuth] Failed to read from ACCOUNTS_KV:", e);
			}
		}

		// Legacy fallback
		return "legacy_single_account";
	}

	/**
	 * Initializes authentication for a specific model request.
	 * It finds a healthy account from the pool and retrieves/refreshes its access token.
	 */
	public async initializeAuth(modelId: string = "gemini-3.5-flash"): Promise<void> {
		const accountId = await this.getAvailableAccount(modelId);
		this.currentAccountId = accountId;

		try {
			if (accountId === "legacy_single_account") {
				await this.initializeLegacyAuth();
				return;
			}

			// Multi-account logic
			const accountData = await this.env.ACCOUNTS_KV!.get<AccountData>(accountId, "json");
			if (!accountData) throw new Error(`Account data for ${accountId} not found in KV`);

			const timeUntilExpiry = (accountData.expiry_date || 0) - Date.now();
			if (accountData.access_token && timeUntilExpiry > TOKEN_BUFFER_TIME) {
				this.currentAccessToken = accountData.access_token;
				console.log(`[MultiAuth] Using cached token for ${accountId}, valid for ${Math.floor(timeUntilExpiry / 1000)}s`);
				return;
			}

			console.log(`[MultiAuth] Token for ${accountId} expired. Refreshing...`);
			await this.refreshAccountToken(accountId, accountData);

		} catch (e: unknown) {
			console.error(`[MultiAuth] Failed to initialize auth for ${accountId}:`, safeLogText(String(e)));
			if (e instanceof AuthError || e instanceof UpstreamApiError) throw e;
			const errorMessage = e instanceof Error ? e.message : String(e);
			throw new AuthError(500, "authentication_failed", "Authentication failed: " + errorMessage);
		}
	}

	private async refreshAccountToken(accountId: string, accountData: AccountData): Promise<void> {
		const existing = accountRefreshInFlight.get(accountId);
		if (existing) return existing;

		const refresh = this.performAccountRefresh(accountId, accountData).finally(() => {
			accountRefreshInFlight.delete(accountId);
		});
		accountRefreshInFlight.set(accountId, refresh);
		return refresh;
	}

	private async performAccountRefresh(accountId: string, accountData: AccountData): Promise<void> {
		if (!accountData.refresh_token) {
			accountData.is_invalid = true;
			await this.env.ACCOUNTS_KV?.put(accountId, JSON.stringify(accountData));
			throw new AuthError(401, "invalid_grant", "Account has no refresh token");
		}

		for (let attempt = 1; attempt <= TOKEN_REFRESH_MAX_ATTEMPTS; attempt++) {
			try {
				const params = new URLSearchParams({
					client_id: OAUTH_CLIENT_ID,
					refresh_token: accountData.refresh_token,
					grant_type: "refresh_token"
				});
				if (OAUTH_CLIENT_SECRET) params.append("client_secret", OAUTH_CLIENT_SECRET);

				const { response: refreshResponse, text: responseText } =
					await fetchTextWithTimeout(
						OAUTH_REFRESH_URL,
						{
							method: "POST",
							headers: { "Content-Type": "application/x-www-form-urlencoded" },
							body: params
						},
						this.tokenRefreshTimeoutMs()
					);

				if (refreshResponse.ok) {
					let refreshData: TokenRefreshResponse;
					try {
						refreshData = JSON.parse(responseText) as TokenRefreshResponse;
					} catch {
						throw new AuthError(502, "token_refresh_invalid_response", "Token refresh returned invalid JSON");
					}
					this.currentAccessToken = refreshData.access_token;
					accountData.access_token = refreshData.access_token;
					accountData.expiry_date = Date.now() + refreshData.expires_in * 1000;
					try {
						await this.env.ACCOUNTS_KV!.put(accountId, JSON.stringify(accountData));
					} catch {
						console.warn(`[MultiAuth] Failed to cache refreshed token for ${accountId}`);
					}
					console.log(`[MultiAuth] Token for ${accountId} refreshed successfully.`);
					return;
				}

				console.error(
					`[MultiAuth] Token refresh failed for ${accountId} (HTTP ${refreshResponse.status}):`,
					safeLogText(responseText)
				);

				if (/invalid_grant/i.test(responseText)) {
					accountData.is_invalid = true;
					try {
						await this.env.ACCOUNTS_KV!.put(accountId, JSON.stringify(accountData));
					} catch {
						console.warn(`[MultiAuth] Failed to mark account ${accountId} as invalid`);
					}
					console.error(`[MultiAuth] Account ${accountId} marked as permanently INVALID.`);
					throw new AuthError(401, "invalid_grant", "Refresh token is no longer valid");
				}
				if (refreshResponse.status === 401 || refreshResponse.status === 403) {
					throw new AuthError(
						refreshResponse.status,
						"permission_denied",
						"Refresh token lacks permission to obtain an access token"
					);
				}

				const retryable =
					refreshResponse.status === 429 || refreshResponse.status >= 500;
				if (!retryable || attempt === TOKEN_REFRESH_MAX_ATTEMPTS) {
					throw new AuthError(
						refreshResponse.status >= 500 ? 502 : refreshResponse.status,
						"token_refresh_failed",
						`Token refresh failed with upstream status ${refreshResponse.status}`
					);
				}

				const retryAfter = parseRetryAfterMs(refreshResponse.headers, responseText);
				const delay = Math.min(
					retryAfter ?? Math.pow(2, attempt - 1) * 1000,
					MAX_RETRY_DELAY_MS
				);
				await sleep(delay);
			} catch (error: unknown) {
				if (error instanceof AuthError) throw error;
				const message = error instanceof Error ? error.message : String(error);
				if (attempt === TOKEN_REFRESH_MAX_ATTEMPTS) {
					throw new AuthError(
						502,
						"token_refresh_network_error",
						`Token refresh failed after ${attempt} attempts: ${safeLogText(message)}`
					);
				}
				await sleep(Math.min(Math.pow(2, attempt - 1) * 1000, MAX_RETRY_DELAY_MS));
			}
		}
	}

	private async initializeLegacyAuth(): Promise<void> {
		if (!this.env.GCP_SERVICE_ACCOUNT) {
			throw new AuthError(
				500,
				"missing_credentials",
				"`GCP_SERVICE_ACCOUNT` environment variable not set. Please provide OAuth2 credentials JSON."
			);
		}

		let oauth2Creds: OAuth2Credentials;
		try {
			oauth2Creds = JSON.parse(this.env.GCP_SERVICE_ACCOUNT) as OAuth2Credentials;
		} catch {
			throw new AuthError(500, "invalid_credentials", "GCP_SERVICE_ACCOUNT is not valid JSON");
		}
		if (!oauth2Creds.refresh_token) {
			throw new AuthError(500, "invalid_credentials", "GCP_SERVICE_ACCOUNT has no refresh_token");
		}

		if (!legacyForceRefresh) {
			let cachedTokenData = null;
			try {
				if (this.env.ACCOUNTS_KV) {
					const cachedToken = await this.env.ACCOUNTS_KV.get(KV_TOKEN_KEY, "json");
					if (cachedToken) cachedTokenData = cachedToken as { access_token?: string; expiry_date?: number };
				}
			} catch (e) { /* ignore cache errors */ }

			if (
				cachedTokenData?.access_token &&
				(cachedTokenData.expiry_date || 0) - Date.now() > TOKEN_BUFFER_TIME
			) {
				this.currentAccessToken = cachedTokenData.access_token;
				return;
			}

			if (
				legacyMemoryToken?.refreshToken === oauth2Creds.refresh_token &&
				legacyMemoryToken.expiryDate - Date.now() > TOKEN_BUFFER_TIME
			) {
				this.currentAccessToken = legacyMemoryToken.accessToken;
				return;
			}

			const timeUntilExpiry = (oauth2Creds.expiry_date || 0) - Date.now();
			if (oauth2Creds.access_token && timeUntilExpiry > TOKEN_BUFFER_TIME) {
				this.currentAccessToken = oauth2Creds.access_token;
				legacyMemoryToken = {
					refreshToken: oauth2Creds.refresh_token,
					accessToken: oauth2Creds.access_token,
					expiryDate: oauth2Creds.expiry_date
				};
				await this.cacheLegacyTokenInKV(oauth2Creds.access_token, oauth2Creds.expiry_date);
				return;
			}
		}

		console.log("[LegacyAuth] Refreshing access token...");
		const refreshData = await this.refreshLegacyToken(oauth2Creds.refresh_token);
		const expiryDate = Date.now() + refreshData.expires_in * 1000;
		this.currentAccessToken = refreshData.access_token;
		legacyMemoryToken = {
			refreshToken: oauth2Creds.refresh_token,
			accessToken: refreshData.access_token,
			expiryDate
		};
		legacyForceRefresh = false;
		await this.cacheLegacyTokenInKV(refreshData.access_token, expiryDate);
	}

	private async refreshLegacyToken(refreshToken: string): Promise<TokenRefreshResponse> {
		if (legacyRefreshInFlight) return legacyRefreshInFlight;

		legacyRefreshInFlight = (async () => {
			for (let attempt = 1; attempt <= TOKEN_REFRESH_MAX_ATTEMPTS; attempt++) {
				try {
					const params = new URLSearchParams({
						client_id: OAUTH_CLIENT_ID,
						refresh_token: refreshToken,
						grant_type: "refresh_token"
					});
					if (OAUTH_CLIENT_SECRET) params.append("client_secret", OAUTH_CLIENT_SECRET);

					const { response: refreshResponse, text: responseText } =
						await fetchTextWithTimeout(
							OAUTH_REFRESH_URL,
							{
								method: "POST",
								headers: { "Content-Type": "application/x-www-form-urlencoded" },
								body: params
							},
							this.tokenRefreshTimeoutMs()
						);

					if (refreshResponse.ok) {
						try {
							return JSON.parse(responseText) as TokenRefreshResponse;
						} catch {
							throw new AuthError(502, "token_refresh_invalid_response", "Token refresh returned invalid JSON");
						}
					}

					if (/invalid_grant/i.test(responseText)) {
						throw new AuthError(401, "invalid_grant", "Refresh token is no longer valid");
					}
					if (refreshResponse.status === 401 || refreshResponse.status === 403) {
						throw new AuthError(
							refreshResponse.status,
							"permission_denied",
							"Refresh token lacks permission to obtain an access token"
						);
					}

					const retryable = refreshResponse.status === 429 || refreshResponse.status >= 500;
					if (!retryable || attempt === TOKEN_REFRESH_MAX_ATTEMPTS) {
						throw new AuthError(
							refreshResponse.status >= 500 ? 502 : refreshResponse.status,
							"token_refresh_failed",
							`Token refresh failed with upstream status ${refreshResponse.status}`
						);
					}
					const retryAfter = parseRetryAfterMs(refreshResponse.headers, responseText);
					await sleep(
						Math.min(retryAfter ?? Math.pow(2, attempt - 1) * 1000, MAX_RETRY_DELAY_MS)
					);
				} catch (error: unknown) {
					if (error instanceof AuthError) throw error;
					if (attempt === TOKEN_REFRESH_MAX_ATTEMPTS) {
						throw new AuthError(
							502,
							"token_refresh_network_error",
							`Token refresh failed after ${attempt} attempts: ${safeLogText(String(error))}`
						);
					}
					await sleep(Math.min(Math.pow(2, attempt - 1) * 1000, MAX_RETRY_DELAY_MS));
				}
			}
			throw new AuthError(502, "token_refresh_failed", "Token refresh failed");
		})().finally(() => {
			legacyRefreshInFlight = undefined;
		});

		return legacyRefreshInFlight;
	}

	private async cacheLegacyTokenInKV(accessToken: string, expiryDate: number): Promise<void> {
		try {
			if (this.env.ACCOUNTS_KV) {
				const ttlSeconds = Math.floor((expiryDate - Date.now()) / 1000) - 300;
				if (ttlSeconds >= 60) {
					await this.env.ACCOUNTS_KV.put(KV_TOKEN_KEY, JSON.stringify({ access_token: accessToken, expiry_date: expiryDate }), {
						expirationTtl: ttlSeconds
					});
				}
			}
		} catch (e) { /* ignore */ }
	}

	/**
	 * Marks the current account as exhausted for a specific provider.
	 * This triggers the round-robin to pick a different account on the next attempt.
	 */
	public async markAccountExhausted(modelId: string, resetDelaySeconds: number): Promise<void> {
		if (!this.currentAccountId || this.currentAccountId === "legacy_single_account" || !this.env.ACCOUNTS_KV) {
			console.warn("[MultiAuth] Cannot mark account exhausted: not using KV pool or no account selected.");
			return;
		}

		const provider = this.getProviderForModel(modelId);
		const unbanTimestamp = Date.now() + (resetDelaySeconds * 1000);
		
		try {
			const accountData = await this.env.ACCOUNTS_KV.get<AccountData>(this.currentAccountId, "json");
			if (accountData) {
				if (provider === "gemini") {
					accountData.exhausted_gemini_until = unbanTimestamp;
				} else {
					accountData.exhausted_partner_until = unbanTimestamp;
				}
				await this.env.ACCOUNTS_KV.put(this.currentAccountId, JSON.stringify(accountData));
				console.log(`[MultiAuth] 🛑 Account ${this.currentAccountId} marked EXHAUSTED for ${provider} until ${new Date(unbanTimestamp).toISOString()}`);
			}
		} catch (e) {
			console.error("[MultiAuth] Failed to update exhaustion state:", e);
		}
	}

	public async clearTokenCache(): Promise<void> {
		try {
			if (this.currentAccountId && this.currentAccountId !== "legacy_single_account" && this.env.ACCOUNTS_KV) {
				const accountData = await this.env.ACCOUNTS_KV.get<AccountData>(this.currentAccountId, "json");
				if (accountData) {
					accountData.access_token = undefined;
					accountData.expiry_date = undefined;
					await this.env.ACCOUNTS_KV.put(this.currentAccountId, JSON.stringify(accountData));
				}
			} else {
				legacyMemoryToken = undefined;
				legacyForceRefresh = true;
				if (this.env.ACCOUNTS_KV) {
					await this.env.ACCOUNTS_KV.delete(KV_TOKEN_KEY);
				}
			}
		} catch (e) { /* ignore */ }
	}

	public async callEndpoint(
		method: string,
		body: Record<string, unknown>,
		retryOrOptions: boolean | CallEndpointOptions = false
	): Promise<unknown> {
		const options: CallEndpointOptions =
			typeof retryOrOptions === "boolean" ? { isRetry: retryOrOptions } : retryOrOptions;

		if (!this.currentAccessToken) {
			await this.initializeAuth();
		}
		if (!this.currentAccessToken) {
			throw new AuthError(401, "missing_access_token", "Unable to obtain an access token");
		}

		const { response, text: responseText } = await fetchTextWithTimeout(
			`${CODE_ASSIST_ENDPOINT}/${CODE_ASSIST_API_VERSION}:${method}`,
			{
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${this.currentAccessToken}`,
					"User-Agent": "antigravity/cli/1.0.2 windows/amd64"
				},
				body: JSON.stringify(body)
			},
			options.timeoutMs ?? 30_000,
			options.signal
		);

		if (!response.ok) {
			const errorText = responseText;
			const requestId = getUpstreamRequestId(response.headers, errorText);

			if (response.status === 401 && !options.isRetry) {
				console.log("Got 401 from Code Assist endpoint, refreshing credentials once...");
				this.currentAccessToken = null;
				await this.clearTokenCache();
				await this.initializeAuth();
				return this.callEndpoint(method, body, { ...options, isRetry: true });
			}

			throw new UpstreamApiError(
				response.status,
				`API call failed with status ${response.status}`,
				safeLogText(errorText, 2000),
				{
					requestId,
					retryAfterMs: parseRetryAfterMs(response.headers, errorText)
				}
			);
		}

		try {
			return JSON.parse(responseText) as unknown;
		} catch {
			throw new UpstreamApiError(
				502,
				"Upstream API returned invalid JSON",
				safeLogText(responseText, 2000),
				{ requestId: getUpstreamRequestId(response.headers, responseText) }
			);
		}
	}

	public getAccessToken(): string | null {
		return this.currentAccessToken;
	}
}
