const SECRET_KEY_PATTERN =
	/((?:authorization|proxy-authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|password|secret|code))(\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,&}]+)/gi;

const JSON_SECRET_PATTERN =
	/("(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|password|secret|code)"\s*:\s*)"[^"]*"/gi;

const DATA_URL_PATTERN = /data:([a-zA-Z0-9.+-]+\/[a-zA-Z0-9.+-]+);base64,[A-Za-z0-9+/=\s]+/g;
const LONG_BASE64_PATTERN = /("(?:data|inlineData|input_audio|input_video|input_pdf|url)"\s*:\s*)"[A-Za-z0-9+/=\r\n]{200,}"/gi;

export function redactSensitiveText(input: string): string {
	return input
		.replace(DATA_URL_PATTERN, "data:$1;base64,[REDACTED_MEDIA]")
		.replace(/(authorization\s*[:=]\s*)(?:Bearer\s+)?[^\s,}]+/gi, "$1[REDACTED]")
		.replace(JSON_SECRET_PATTERN, '$1"[REDACTED]"')
		.replace(SECRET_KEY_PATTERN, "$1$2[REDACTED]")
		.replace(LONG_BASE64_PATTERN, '$1"[REDACTED_MEDIA]"');
}

export function safeLogText(input: string, maxLength = 500): string {
	const redacted = redactSensitiveText(input);
	return redacted.length > maxLength ? `${redacted.slice(0, maxLength)}...` : redacted;
}

export function redactHeaders(headers: Record<string, string> | Headers): Record<string, string> {
	const source = headers instanceof Headers ? Object.fromEntries(headers.entries()) : headers;
	const redacted: Record<string, string> = {};
	for (const [key, value] of Object.entries(source)) {
		redacted[key] = /authorization|api-key|token|secret|cookie/i.test(key) ? "[REDACTED]" : value;
	}
	return redacted;
}

export function safeErrorMessage(error: unknown): string {
	if (error instanceof Error) return redactSensitiveText(error.message);
	return redactSensitiveText(String(error));
}
