// --- Google Code Assist API Constants ---
export const CODE_ASSIST_ENDPOINT = "https://daily-cloudcode-pa.googleapis.com";
export const CODE_ASSIST_API_VERSION = "v1internal";

// --- OAuth2 Configuration ---
export const OAUTH_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com";
export const OAUTH_CLIENT_SECRET = "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";
export const OAUTH_REFRESH_URL = "https://oauth2.googleapis.com/token";

// --- Token Management ---
export const TOKEN_BUFFER_TIME = 5 * 60 * 1000; // 5 minutes in milliseconds
export const KV_TOKEN_KEY = "oauth_token_cache";

// --- Upstream timeouts and retry limits ---
export const TOKEN_REFRESH_TIMEOUT_MS = 15 * 1000;
export const TOKEN_REFRESH_MAX_ATTEMPTS = 3;
export const MODEL_DISCOVERY_TIMEOUT_MS = 10 * 1000;
export const PROJECT_DISCOVERY_TIMEOUT_MS = 10 * 1000;
export const GENERATION_TIMEOUT_MS = 10 * 60 * 1000;
export const DYNAMIC_MODEL_CACHE_TTL_MS = 5 * 60 * 1000;
export const DYNAMIC_MODEL_STALE_TTL_MS = 24 * 60 * 60 * 1000;
export const PROJECT_ID_CACHE_TTL_MS = 30 * 60 * 1000;
export const MAX_RATE_LIMIT_RETRIES = 3;
export const MAX_RETRY_DELAY_MS = 30 * 1000;

// --- OpenAI API Constants ---
export const OPENAI_CHAT_COMPLETION_OBJECT = "chat.completion.chunk";
export const OPENAI_MODEL_OWNER = "google-gemini-cli";
