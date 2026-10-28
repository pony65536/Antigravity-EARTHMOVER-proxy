import { Hono } from "hono";
import { Env } from "../types";
import { AuthManager } from "../auth";
import { GeminiApiClient } from "../gemini-client";
import { safeLogText } from "../utils/redact";

/**
 * Debug and testing routes for troubleshooting authentication and API functionality.
 */
export const DebugRoute = new Hono<{ Bindings: Env }>();

// Check KV cache status
DebugRoute.get("/cache", async (c) => {
	return c.json({
		status: "error",
		message: "Not implemented in Multi-Account architecture"
	}, 501);
});

// Simple token test endpoint
DebugRoute.post("/token-test", async (c) => {
	try {
		console.log("Token test endpoint called");
		const authManager = new AuthManager(c.env);

		// Test authentication only
		await authManager.initializeAuth();
		console.log("Token test passed");

		return c.json({
			status: "ok",
			message: "Token authentication successful"
		});
	} catch (e: unknown) {
		const errorMessage = safeLogText(e instanceof Error ? e.message : String(e), 1000);
		console.error("Token test error:", errorMessage);
		return c.json(
			{
				status: "error",
				message: errorMessage
				// Removed stack trace for security
			},
			500
		);
	}
});

// Full functionality test endpoint
DebugRoute.post("/test", async (c) => {
	try {
		console.log("Test endpoint called");
		const authManager = new AuthManager(c.env);
		const geminiClient = new GeminiApiClient(c.env, authManager);

		// Test authentication
		await authManager.initializeAuth();
		console.log("Auth test passed");

		// Test project discovery
		const projectId = await geminiClient.discoverProjectId();
		console.log("Project discovery test passed");

		return c.json({
			status: "ok",
			message: "Authentication and project discovery successful",
			project_available: !!projectId
			// Removed actual projectId for security
		});
	} catch (e: unknown) {
		const errorMessage = safeLogText(e instanceof Error ? e.message : String(e), 1000);
		console.error("Test endpoint error:", errorMessage);
		return c.json(
			{
				status: "error",
				message: errorMessage
				// Removed stack trace and detailed error message for security
			},
			500
		);
	}
});
