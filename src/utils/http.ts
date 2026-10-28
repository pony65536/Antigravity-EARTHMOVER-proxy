export class RequestTimeoutError extends Error {
	constructor(public readonly timeoutMs: number) {
		super(`Upstream request timed out after ${timeoutMs}ms`);
		this.name = "RequestTimeoutError";
	}
}

export function isAbortError(error: unknown): boolean {
	if (typeof error !== "object" || error === null) return false;
	const candidate = error as { name?: unknown; code?: unknown };
	return candidate.name === "AbortError" || candidate.code === "ABORT_ERR" || candidate.name === "TimeoutError";
}

export function envTimeoutMs(env: Record<string, unknown> | undefined, key: string, fallback: number): number {
	const raw = env?.[key];
	if (typeof raw !== "string" && typeof raw !== "number") return fallback;
	const value = Number(raw);
	if (!Number.isFinite(value) || value <= 0) return fallback;
	return value;
}

/**
 * Combines an optional caller signal with a per-operation timeout. The returned
 * cleanup function must be called once the operation settles.
 */
export function createAbortScope(
	timeoutMs: number,
	parentSignal?: AbortSignal,
): { signal: AbortSignal; cleanup: () => void; didTimeout: () => boolean } {
	const controller = new AbortController();
	let timedOut = false;
	let timer: ReturnType<typeof setTimeout> | undefined;

	const onParentAbort = () => {
		controller.abort(parentSignal?.reason);
	};

	if (parentSignal) {
		if (parentSignal.aborted) {
			controller.abort(parentSignal.reason);
		} else {
			parentSignal.addEventListener("abort", onParentAbort, { once: true });
		}
	}

	if (timeoutMs > 0) {
		timer = setTimeout(() => {
			timedOut = true;
			controller.abort(new RequestTimeoutError(timeoutMs));
		}, timeoutMs);
	}

	return {
		signal: controller.signal,
		cleanup: () => {
			if (timer) clearTimeout(timer);
			parentSignal?.removeEventListener("abort", onParentAbort);
		},
		didTimeout: () => timedOut,
	};
}

export async function fetchWithTimeout(
	input: RequestInfo | URL,
	init: RequestInit,
	timeoutMs: number,
	parentSignal?: AbortSignal,
): Promise<Response> {
	const scope = createAbortScope(timeoutMs, parentSignal);
	try {
		return await fetch(input, { ...init, signal: scope.signal });
	} catch (error: unknown) {
		if (isAbortError(error)) {
			if (parentSignal?.aborted) throw error;
			if (scope.didTimeout()) throw new RequestTimeoutError(timeoutMs);
		}
		throw error;
	} finally {
		scope.cleanup();
	}
}

export async function fetchTextWithTimeout(
	input: RequestInfo | URL,
	init: RequestInit,
	timeoutMs: number,
	parentSignal?: AbortSignal,
): Promise<{ response: Response; text: string }> {
	const scope = createAbortScope(timeoutMs, parentSignal);
	try {
		const response = await fetch(input, { ...init, signal: scope.signal });
		const text = await response.text();
		return { response, text };
	} catch (error: unknown) {
		if (isAbortError(error)) {
			if (parentSignal?.aborted) throw error;
			if (scope.didTimeout()) throw new RequestTimeoutError(timeoutMs);
		}
		throw error;
	} finally {
		scope.cleanup();
	}
}

export function parseRetryAfterMs(headers: Headers, bodyText = ""): number | undefined {
	const retryAfter = headers.get("retry-after");
	if (retryAfter) {
		const seconds = Number(retryAfter);
		if (Number.isFinite(seconds) && seconds >= 0) return Math.ceil(seconds * 1000);
		const date = Date.parse(retryAfter);
		if (Number.isFinite(date)) return Math.max(0, date - Date.now());
	}

	const match = bodyText.match(/(?:retry in|reset after|retry after)\s+([\d.]+)\s*s/i);
	if (match?.[1]) {
		const seconds = Number(match[1]);
		if (Number.isFinite(seconds) && seconds >= 0) return Math.ceil(seconds * 1000);
	}
	return undefined;
}

export function getUpstreamRequestId(headers: Headers, bodyText = ""): string | undefined {
	const headerId =
		headers.get("x-request-id") ||
		headers.get("x-goog-request-id") ||
		headers.get("x-correlation-id") ||
		headers.get("traceparent");
	if (headerId) return headerId;

	const match = bodyText.match(/"(?:requestId|request_id|traceId|trace_id)"\s*:\s*"([^"]+)"/i);
	return match?.[1];
}

export function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));
}
