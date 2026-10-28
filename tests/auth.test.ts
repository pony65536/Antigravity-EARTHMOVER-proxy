import assert from "node:assert/strict";
import test from "node:test";
import { AuthError, AuthManager } from "../src/auth";
import { Env } from "../src/types";

function credentials(expiryOffsetMs = -60_000): string {
  return JSON.stringify({
    access_token: "expired-access-token",
    refresh_token: "refresh-token",
    scope: "",
    token_type: "Bearer",
    id_token: "",
    expiry_date: Date.now() + expiryOffsetMs,
  });
}

function authEnv(extra: Record<string, string> = {}): Env {
  return {
    GCP_SERVICE_ACCOUNT: credentials(),
    ...extra,
  } as Env;
}

function installFetchMock(
  handler: (url: string) => Response | Promise<Response>,
): { calls: string[]; restore: () => void } {
  const original = globalThis.fetch;
  const calls: string[] = [];
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    calls.push(url);
    return handler(url);
  }) as typeof fetch;
  return {
    calls,
    restore: () => {
      globalThis.fetch = original;
    },
  };
}

test.beforeEach(async () => {
  const auth = new AuthManager({} as Env);
  await auth.clearTokenCache();
});

test("invalid_grant is terminal and is not retried", async () => {
  const mock = installFetchMock(() =>
    new Response('{"error":"invalid_grant"}', { status: 400 }),
  );
  try {
    const auth = new AuthManager(authEnv());
    await assert.rejects(
      auth.initializeAuth(),
      (error: unknown) =>
        error instanceof AuthError &&
        error.status === 401 &&
        error.code === "invalid_grant",
    );
    assert.equal(mock.calls.length, 1);
  } finally {
    mock.restore();
  }
});

test("permission errors are reported without retrying", async () => {
  const mock = installFetchMock(() =>
    new Response('{"error":"insufficient_scope"}', { status: 403 }),
  );
  try {
    const auth = new AuthManager(authEnv());
    await assert.rejects(
      auth.initializeAuth(),
      (error: unknown) =>
        error instanceof AuthError &&
        error.status === 403 &&
        error.code === "permission_denied",
    );
    assert.equal(mock.calls.length, 1);
  } finally {
    mock.restore();
  }
});

test("transient token refresh failures retry with Retry-After", async () => {
  let attempts = 0;
  const mock = installFetchMock(() => {
    attempts += 1;
    if (attempts === 1) {
      return new Response("temporarily unavailable", {
        status: 503,
        headers: { "Retry-After": "0" },
      });
    }
    return Response.json({
      access_token: "fresh-access-token",
      expires_in: 3600,
    });
  });

  try {
    const auth = new AuthManager(authEnv());
    await auth.initializeAuth();
    assert.equal(auth.getAccessToken(), "fresh-access-token");
    assert.equal(attempts, 2);
  } finally {
    mock.restore();
  }
});

test("clearTokenCache forces refresh even when the static token looks fresh", async () => {
  let attempts = 0;
  const mock = installFetchMock(() => {
    attempts += 1;
    return Response.json({
      access_token: `fresh-${attempts}`,
      expires_in: 3600,
    });
  });

  try {
    const auth = new AuthManager(authEnv());
    await auth.clearTokenCache();
    await auth.initializeAuth();
    assert.equal(auth.getAccessToken(), "fresh-1");
    assert.equal(attempts, 1);
  } finally {
    mock.restore();
  }
});
