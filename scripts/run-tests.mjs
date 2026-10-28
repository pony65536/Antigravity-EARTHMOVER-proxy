import { spawnSync } from "node:child_process";
import { mkdir, readFile, readdir } from "node:fs/promises";
import { build } from "esbuild";

const live = process.argv.includes("--live");
const outputDirectory = ".test-dist";

await mkdir(outputDirectory, { recursive: true });

const testFiles = (await readdir("tests"))
  .filter((name) => name.endsWith(".test.ts"))
  .filter((name) => (live ? name === "live.test.ts" : name !== "live.test.ts"))
  .sort();

if (testFiles.length === 0) {
  console.error("No test files found.");
  process.exitCode = 1;
  process.exit();
}

const outputs = [];
for (const file of testFiles) {
  const outputFile = `${outputDirectory}/${file.replace(/\.ts$/, ".mjs")}`;
  await build({
    entryPoints: [`tests/${file}`],
    bundle: true,
    platform: "node",
    format: "esm",
    target: "node24",
    outfile: outputFile,
    sourcemap: "inline",
  });
  outputs.push(outputFile);
}

const env = { ...process.env };
if (live) {
  env.LIVE = "1";
  try {
    const devVars = await readFile(".dev.vars", "utf8");
    for (const rawLine of devVars.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#")) continue;
      const separator = line.indexOf("=");
      if (separator === -1) continue;
      const key = line.slice(0, separator).trim();
      let value = line.slice(separator + 1).trim();
      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }
      env[key] = value;
    }
  } catch {
    console.error("test:live requires a readable .dev.vars file.");
    process.exitCode = 1;
    process.exit();
  }
}

const result = spawnSync(process.execPath, ["--test", ...outputs], {
  stdio: "inherit",
  env,
});

process.exitCode = result.status ?? 1;
