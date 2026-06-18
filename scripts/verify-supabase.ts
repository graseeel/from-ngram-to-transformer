import { readFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";

const API_BASE = "https://api.supabase.com/v1";
const REPO_SLUG = "from-ngram-to-transformer";
const PROJECT_PREFIX = `${REPO_SLUG}-`;

type JsonObject = Record<string, unknown>;

function env(name: string): string | undefined {
  const value = process.env[name]?.trim();
  return value && value.length > 0 ? value : undefined;
}

async function apiFetch<T>(path: string): Promise<T> {
  const token = env("SUPABASE_ACCESS_TOKEN");
  if (!token) {
    throw new Error("SUPABASE_ACCESS_TOKEN is required to verify remote project ownership");
  }
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Supabase API ${response.status} ${response.statusText}: ${body}`);
  }
  return (await response.json()) as T;
}

function projectsFromAuthenticatedCli(): Array<JsonObject> {
  const output = execFileSync("supabase", ["projects", "list", "--output", "json"], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  return JSON.parse(output) as Array<JsonObject>;
}

async function readManifest(): Promise<JsonObject | null> {
  try {
    return JSON.parse(await readFile(".supabase-project.json", "utf8")) as JsonObject;
  } catch {
    return null;
  }
}

function refOf(project: JsonObject): string | undefined {
  const ref = project.ref ?? project.id;
  return typeof ref === "string" ? ref : undefined;
}

async function main(): Promise<void> {
  const expectedRef = env("SUPABASE_PROJECT_REF");
  if (!expectedRef) {
    throw new Error("SUPABASE_PROJECT_REF is required");
  }

  const projects = env("SUPABASE_ACCESS_TOKEN")
    ? await apiFetch<Array<JsonObject>>("/projects")
    : projectsFromAuthenticatedCli();
  const project = projects.find((item) => refOf(item) === expectedRef);
  if (!project) {
    throw new Error(`Project ref ${expectedRef} was not found in this Supabase account`);
  }

  const name = project.name;
  if (typeof name !== "string" || !name.startsWith(PROJECT_PREFIX)) {
    throw new Error(`Project ${expectedRef} is named ${String(name)} and does not match ${PROJECT_PREFIX}*`);
  }

  const organizationId = env("SUPABASE_ORG_ID");
  if (organizationId && project.organization_id && project.organization_id !== organizationId) {
    throw new Error(`Project ${expectedRef} does not belong to SUPABASE_ORG_ID=${organizationId}`);
  }

  const manifest = await readManifest();
  if (manifest) {
    if (manifest.repo !== REPO_SLUG) {
      throw new Error(".supabase-project.json belongs to a different repository");
    }
    if (manifest.project_ref !== expectedRef) {
      throw new Error(".supabase-project.json project_ref does not match SUPABASE_PROJECT_REF");
    }
    if (manifest.project_name !== name) {
      throw new Error(".supabase-project.json project_name does not match the remote project");
    }
  }

  console.log(`Verified ${expectedRef} (${name}) as an isolated project for ${REPO_SLUG}.`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
