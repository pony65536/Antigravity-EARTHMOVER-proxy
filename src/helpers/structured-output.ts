type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export type SchemaProtocol = "responseSchema" | "responseJsonSchema";

export class SchemaConversionError extends Error {
	constructor(
		message: string,
		public readonly code: string = "unsupported_json_schema",
		public readonly path: string = "$",
	) {
		super(message);
		this.name = "SchemaConversionError";
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

const METADATA_KEYS = new Set([
	"$schema",
	"$id",
	"$comment",
	"title",
	"default",
	"examples",
	"example",
	"deprecated",
	"readOnly",
	"writeOnly",
]);

const REJECTED_SCHEMA_KEYS = new Set([
	"$ref",
	"$defs",
	"definitions",
	"allOf",
	"oneOf",
	"not",
	"if",
	"then",
	"else",
	"dependentRequired",
	"dependentSchemas",
	"patternProperties",
	"unevaluatedProperties",
	"unevaluatedItems",
	"contains",
	"minContains",
	"maxContains",
	"propertyNames",
	"contentEncoding",
	"contentMediaType",
	"contentSchema",
]);

const JSON_SCHEMA_ONLY_KEYS = new Set([
	"exclusiveMinimum",
	"exclusiveMaximum",
	"multipleOf",
	"uniqueItems",
	"minProperties",
	"maxProperties",
]);

const SUPPORTED_KEYS = new Set([
	"type",
	"enum",
	"const",
	"anyOf",
	"nullable",
	"description",
	"format",
	"properties",
	"required",
	"items",
	"additionalProperties",
	"minimum",
	"maximum",
	"minLength",
	"maxLength",
	"pattern",
	"minItems",
	"maxItems",
	"propertyOrdering",
	"exclusiveMinimum",
	"exclusiveMaximum",
	"multipleOf",
	"uniqueItems",
	"minProperties",
	"maxProperties",
]);

export function selectSchemaProtocol(
	modelId: string,
	requested?: string,
): SchemaProtocol {
	if (requested === "responseSchema" || requested === "responseJsonSchema") {
		return requested;
	}
	// Gemini 3.x accepts the newer full JSON Schema field. Older models keep
	// using the OpenAPI-style responseSchema field.
	return /gemini[\s_-]*3(?:[.\s_-])/i.test(modelId)
		? "responseJsonSchema"
		: "responseSchema";
}

function assertNoUnsupportedKey(key: string, protocol: SchemaProtocol, path: string): void {
	if (METADATA_KEYS.has(key)) return;
	if (REJECTED_SCHEMA_KEYS.has(key)) {
		throw new SchemaConversionError(
			`JSON schema keyword '${key}' is not supported by Gemini structured output`,
			"unsupported_json_schema",
			`${path}.${key}`,
		);
	}
	if (JSON_SCHEMA_ONLY_KEYS.has(key) && protocol === "responseSchema") {
		throw new SchemaConversionError(
			`JSON schema keyword '${key}' is not supported by Gemini responseSchema`,
			"unsupported_json_schema",
			`${path}.${key}`,
		);
	}
	if (!SUPPORTED_KEYS.has(key)) {
		throw new SchemaConversionError(
			`JSON schema keyword '${key}' is not supported`,
			"unsupported_json_schema",
			`${path}.${key}`,
		);
	}
}

function inferTypeFromEnum(values: unknown[]): string | undefined {
	if (values.length === 0) return undefined;
	if (values.every((value) => typeof value === "string")) return "string";
	if (values.every((value) => typeof value === "number")) return "number";
	if (values.every((value) => typeof value === "boolean")) return "boolean";
	return undefined;
}

function convertType(type: unknown, protocol: SchemaProtocol, path: string): JsonValue {
	if (typeof type === "string") {
		return protocol === "responseSchema" ? type.toUpperCase() : type;
	}
	if (Array.isArray(type)) {
		if (protocol === "responseSchema") {
			const nonNull = type.filter((entry) => entry !== "null");
			if (nonNull.length !== 1) {
				throw new SchemaConversionError(
					"Union types must be expressed with anyOf when using Gemini responseSchema",
					"unsupported_json_schema",
					`${path}.type`,
				);
			}
			return String(nonNull[0]).toUpperCase();
		}
		return type.map((entry) => String(entry));
	}
	throw new SchemaConversionError("Schema 'type' must be a string or array", "invalid_json_schema", `${path}.type`);
}

function convertNode(
	node: unknown,
	protocol: SchemaProtocol,
	path: string,
	depth: number,
): JsonValue {
	if (depth > 64) {
		throw new SchemaConversionError("JSON schema nesting is too deep", "invalid_json_schema", path);
	}
	if (typeof node === "boolean") {
		throw new SchemaConversionError(
			"Boolean JSON schemas are not supported",
			"unsupported_json_schema",
			path,
		);
	}
	if (!isRecord(node)) {
		throw new SchemaConversionError("Schema nodes must be objects", "invalid_json_schema", path);
	}

	const responseSchema = protocol === "responseSchema";
	let typeValue: unknown = node.type;
	let enumValue: unknown = node.enum;
	let anyOfValue: unknown = node.anyOf;
	let nullable = false;

	if (responseSchema) {
		if (node.nullable === true) nullable = true;

		if (Array.isArray(typeValue) && typeValue.includes("null")) {
			nullable = true;
			const nonNull = typeValue.filter((entry) => entry !== "null");
			if (nonNull.length !== 1) {
				throw new SchemaConversionError(
					"Union types must be expressed with anyOf when using Gemini responseSchema",
					"unsupported_json_schema",
					`${path}.type`,
				);
			}
			typeValue = nonNull[0];
		}

		if (Array.isArray(anyOfValue)) {
			const hasNullBranch = anyOfValue.some(
				(branch) => isRecord(branch) && branch.type === "null",
			);
			if (hasNullBranch) {
				nullable = true;
				anyOfValue = anyOfValue.filter(
					(branch) => !(isRecord(branch) && branch.type === "null"),
				);
				if ((anyOfValue as unknown[]).length === 0) {
					throw new SchemaConversionError(
						"anyOf must contain at least one non-null schema",
						"invalid_json_schema",
						`${path}.anyOf`,
					);
				}
			}
		}

		if (Array.isArray(enumValue) && enumValue.includes(null)) {
			nullable = true;
			enumValue = enumValue.filter((entry) => entry !== null);
		}
	} else if (node.nullable === true) {
		if (typeof typeValue === "string") {
			typeValue = [typeValue, "null"];
		} else if (Array.isArray(typeValue)) {
			typeValue = typeValue.includes("null") ? typeValue : [...typeValue, "null"];
		} else if (Array.isArray(anyOfValue)) {
			anyOfValue = [...anyOfValue, { type: "null" }];
		} else {
			throw new SchemaConversionError(
				"nullable requires a type or anyOf schema when using responseJsonSchema",
				"invalid_json_schema",
				`${path}.nullable`,
			);
		}
	}

	const converted: { [key: string]: JsonValue } = {};
	const handled = new Set<string>();

	for (const [key, value] of Object.entries(node)) {
		if (handled.has(key)) continue;
		assertNoUnsupportedKey(key, protocol, path);

		if (METADATA_KEYS.has(key)) continue;

		switch (key) {
			case "type":
				converted.type = convertType(typeValue, protocol, path);
				handled.add(key);
				break;
			case "nullable":
				if (responseSchema && nullable) converted.nullable = true;
				handled.add(key);
				break;
			case "enum": {
				if (!Array.isArray(enumValue)) {
					throw new SchemaConversionError("Schema 'enum' must be an array", "invalid_json_schema", `${path}.enum`);
				}
				if (!typeValue && !anyOfValue) {
					const inferred = inferTypeFromEnum(enumValue);
					if (inferred) converted.type = convertType(inferred, protocol, path);
				}
				converted.enum = enumValue as JsonValue[];
				handled.add(key);
				break;
			}
			case "const":
				if (responseSchema) {
					if (!converted.enum) converted.enum = [value as JsonValue];
				} else {
					converted.const = value as JsonValue;
				}
				handled.add(key);
				break;
			case "anyOf": {
				if (!Array.isArray(anyOfValue) || anyOfValue.length === 0) {
					throw new SchemaConversionError(
						"Schema 'anyOf' must be a non-empty array",
						"invalid_json_schema",
						`${path}.anyOf`,
					);
				}
				converted.anyOf = (anyOfValue as unknown[]).map((branch, index) =>
					convertNode(branch, protocol, `${path}.anyOf[${index}]`, depth + 1),
				);
				handled.add(key);
				break;
			}
			case "properties": {
				if (!isRecord(value)) {
					throw new SchemaConversionError(
						"Schema 'properties' must be an object",
						"invalid_json_schema",
						`${path}.properties`,
					);
				}
				const properties: { [key: string]: JsonValue } = {};
				for (const [propertyName, propertySchema] of Object.entries(value)) {
					properties[propertyName] = convertNode(
						propertySchema,
						protocol,
						`${path}.properties.${propertyName}`,
						depth + 1,
					);
				}
				converted.properties = properties;
				handled.add(key);
				break;
			}
			case "required": {
				if (!Array.isArray(value) || !value.every((entry) => typeof entry === "string")) {
					throw new SchemaConversionError(
						"Schema 'required' must be an array of strings",
						"invalid_json_schema",
						`${path}.required`,
					);
				}
				converted.required = value as JsonValue[];
				handled.add(key);
				break;
			}
			case "items": {
				if (Array.isArray(value)) {
					throw new SchemaConversionError(
						"Tuple-style array items are not supported",
						"unsupported_json_schema",
						`${path}.items`,
					);
				}
				converted.items = convertNode(value, protocol, `${path}.items`, depth + 1);
				handled.add(key);
				break;
			}
			case "additionalProperties": {
				if (responseSchema) {
					if (value !== false) {
						throw new SchemaConversionError(
							"Open additional properties are not supported by Gemini responseSchema",
							"unsupported_json_schema",
							`${path}.additionalProperties`,
						);
					}
					// Gemini object schemas are closed by default; dropping false is
					// equivalent and avoids silently accepting an open object.
				} else if (typeof value === "boolean") {
					converted.additionalProperties = value;
				} else {
					converted.additionalProperties = convertNode(
						value,
						protocol,
						`${path}.additionalProperties`,
						depth + 1,
					);
				}
				handled.add(key);
				break;
			}
			case "propertyOrdering": {
				if (!Array.isArray(value) || !value.every((entry) => typeof entry === "string")) {
					throw new SchemaConversionError(
						"Schema 'propertyOrdering' must be an array of strings",
						"invalid_json_schema",
						`${path}.propertyOrdering`,
					);
				}
				converted.propertyOrdering = value as JsonValue[];
				handled.add(key);
				break;
			}
			default:
				if (JSON_SCHEMA_ONLY_KEYS.has(key)) {
					converted[key] = value as JsonValue;
				} else {
					converted[key] = value as JsonValue;
				}
				handled.add(key);
		}
	}

	if (responseSchema && nullable) converted.nullable = true;
	return converted;
}

export function convertSchemaForGemini(
	schema: unknown,
	protocol: SchemaProtocol,
): Record<string, unknown> {
	if (!isRecord(schema)) {
		throw new SchemaConversionError("Schema must be an object", "invalid_json_schema", "$");
	}
	const converted = convertNode(schema, protocol, "$", 0);
	if (!isRecord(converted)) {
		throw new SchemaConversionError("Schema conversion failed", "invalid_json_schema", "$");
	}
	return converted;
}

export function assertSupportedSchema(
	schema: unknown,
	protocol: SchemaProtocol = "responseSchema",
): Record<string, unknown> {
	return convertSchemaForGemini(schema, protocol);
}

function typeOfJson(value: unknown): string {
	if (value === null) return "null";
	if (Array.isArray(value)) return "array";
	if (typeof value === "number") return Number.isInteger(value) ? "integer" : "number";
	return typeof value;
}

function matchesType(value: unknown, type: string): boolean {
	if (type === "integer") return typeof value === "number" && Number.isInteger(value);
	if (type === "number") return typeof value === "number" && Number.isFinite(value);
	if (type === "object") return isRecord(value);
	if (type === "array") return Array.isArray(value);
	if (type === "null") return value === null;
	return typeof value === type;
}

export function validateJsonValue(
	value: unknown,
	schema: unknown,
	path = "$",
): string[] {
	if (schema === true || schema === undefined) return [];
	if (schema === false) return [`${path} must not be present`];
	if (!isRecord(schema)) return [`${path} has an invalid schema`];

	const issues: string[] = [];
	const nullable = schema.nullable === true;
	if (value === null && nullable) return issues;

	if (schema.type !== undefined) {
		const types = Array.isArray(schema.type) ? schema.type : [schema.type];
		if (!types.some((type) => typeof type === "string" && matchesType(value, type))) {
			const actual = typeOfJson(value);
			return [`${path} must be of type ${types.join(" or ")}, got ${actual}`];
		}
	}

	if (Array.isArray(schema.enum) && !schema.enum.some((entry) => Object.is(entry, value))) {
		issues.push(`${path} must be one of the allowed enum values`);
	}
	if ("const" in schema && !Object.is(schema.const, value)) {
		issues.push(`${path} must equal the const value`);
	}

	if (Array.isArray(schema.anyOf)) {
		const anyOfIssues = schema.anyOf.map((branch, index) =>
			validateJsonValue(value, branch, `${path}.anyOf[${index}]`),
		);
		if (!anyOfIssues.some((branchIssues) => branchIssues.length === 0)) {
			issues.push(`${path} does not match any allowed schema variant`);
		}
	}

	if (Array.isArray(schema.allOf)) {
		for (const branch of schema.allOf) issues.push(...validateJsonValue(value, branch, path));
	}

	if (Array.isArray(schema.oneOf)) {
		const matches = schema.oneOf.filter(
			(branch) => validateJsonValue(value, branch, path).length === 0,
		).length;
		if (matches !== 1) issues.push(`${path} must match exactly one schema variant`);
	}

	if (isRecord(value)) {
		if (Array.isArray(schema.required)) {
			for (const key of schema.required) {
				if (typeof key === "string" && !(key in value)) {
					issues.push(`${path}.${key} is required`);
				}
			}
		}
		if (isRecord(schema.properties)) {
			for (const [key, propertySchema] of Object.entries(schema.properties)) {
				if (key in value) {
					issues.push(...validateJsonValue(value[key], propertySchema, `${path}.${key}`));
				}
			}
		}
		if (schema.additionalProperties === false) {
			const allowedProperties = isRecord(schema.properties)
				? schema.properties
				: {};
			for (const key of Object.keys(value)) {
				if (!(key in allowedProperties)) {
					issues.push(`${path}.${key} is not an allowed property`);
				}
			}
		} else if (isRecord(schema.additionalProperties)) {
			for (const [key, propertyValue] of Object.entries(value)) {
				if (!isRecord(schema.properties) || !(key in schema.properties)) {
					issues.push(
						...validateJsonValue(propertyValue, schema.additionalProperties, `${path}.${key}`),
					);
				}
			}
		}
		const propertyCount = Object.keys(value).length;
		if (typeof schema.minProperties === "number" && propertyCount < schema.minProperties) {
			issues.push(`${path} must have at least ${schema.minProperties} properties`);
		}
		if (typeof schema.maxProperties === "number" && propertyCount > schema.maxProperties) {
			issues.push(`${path} must have at most ${schema.maxProperties} properties`);
		}
	}

	if (Array.isArray(value)) {
		if (schema.items !== undefined) {
			for (let index = 0; index < value.length; index++) {
				issues.push(...validateJsonValue(value[index], schema.items, `${path}[${index}]`));
			}
		}
		if (typeof schema.minItems === "number" && value.length < schema.minItems) {
			issues.push(`${path} must contain at least ${schema.minItems} items`);
		}
		if (typeof schema.maxItems === "number" && value.length > schema.maxItems) {
			issues.push(`${path} must contain at most ${schema.maxItems} items`);
		}
		if (schema.uniqueItems === true) {
			const serialized = value.map((entry) => JSON.stringify(entry));
			if (new Set(serialized).size !== serialized.length) {
				issues.push(`${path} must contain unique items`);
			}
		}
	}

	if (typeof value === "number") {
		if (typeof schema.minimum === "number" && value < schema.minimum) {
			issues.push(`${path} must be >= ${schema.minimum}`);
		}
		if (typeof schema.maximum === "number" && value > schema.maximum) {
			issues.push(`${path} must be <= ${schema.maximum}`);
		}
		if (typeof schema.exclusiveMinimum === "number" && value <= schema.exclusiveMinimum) {
			issues.push(`${path} must be > ${schema.exclusiveMinimum}`);
		}
		if (typeof schema.exclusiveMaximum === "number" && value >= schema.exclusiveMaximum) {
			issues.push(`${path} must be < ${schema.exclusiveMaximum}`);
		}
		if (typeof schema.multipleOf === "number" && schema.multipleOf > 0) {
			const quotient = value / schema.multipleOf;
			if (Math.abs(quotient - Math.round(quotient)) > 1e-9) {
				issues.push(`${path} must be a multiple of ${schema.multipleOf}`);
			}
		}
	}

	if (typeof value === "string") {
		if (typeof schema.minLength === "number" && value.length < schema.minLength) {
			issues.push(`${path} must have at least ${schema.minLength} characters`);
		}
		if (typeof schema.maxLength === "number" && value.length > schema.maxLength) {
			issues.push(`${path} must have at most ${schema.maxLength} characters`);
		}
		if (typeof schema.pattern === "string") {
			try {
				if (!new RegExp(schema.pattern).test(value)) {
					issues.push(`${path} must match pattern ${schema.pattern}`);
				}
			} catch {
				issues.push(`${path} has an invalid pattern schema`);
			}
		}
	}

	return issues;
}

export interface StructuredOutputError {
	message: string;
	path?: string;
}

/**
 * Some Gemini models wrap JSON in a Markdown code fence even when the
 * response MIME type is application/json. Only structured-output callers use
 * this helper, so removing the fence cannot affect normal text responses.
 */
export function normalizeStructuredJsonText(text: string): string {
	const trimmed = text.trim();
	const openFence = trimmed.search(/```(?:json)?[^\S\r\n]*\r?\n?/i);
	if (openFence === -1) return trimmed;

	const withoutOpeningFence = trimmed
		.slice(openFence)
		.replace(/^```(?:json)?[^\S\r\n]*\r?\n?/i, "");
	const closingFence = withoutOpeningFence.lastIndexOf("```");
	if (closingFence === -1) return trimmed;

	return withoutOpeningFence.slice(0, closingFence).trim();
}

export function validateStructuredOutputText(
	text: string,
	options: { schema?: unknown; requireObject?: boolean } = {},
):
	| { ok: true; value: unknown; text: string }
	| { ok: false; error: StructuredOutputError } {
	const normalized = normalizeStructuredJsonText(text);
	if (!normalized) {
		return { ok: false, error: { message: "Upstream returned an empty structured response" } };
	}

	let parsed: unknown;
	try {
		parsed = JSON.parse(normalized);
	} catch {
		return {
			ok: false,
			error: { message: "Upstream response is not valid JSON", path: "$" },
		};
	}

	if (options.requireObject && (!isRecord(parsed))) {
		return {
			ok: false,
			error: { message: "Upstream response must be a JSON object", path: "$" },
		};
	}

	if (options.schema !== undefined) {
		const issues = validateJsonValue(parsed, options.schema);
		if (issues.length > 0) {
			return {
				ok: false,
				error: {
					message: `Upstream response does not match the requested schema: ${issues[0]}`,
					path: issues[0],
				},
			};
		}
	}

	return { ok: true, value: parsed, text: normalized };
}
