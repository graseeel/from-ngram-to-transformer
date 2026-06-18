import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";

const API_BASE = "https://api.supabase.com/v1";
const REPO_SLUG = "from-ngram-to-transformer";
const PROJECT_PREFIX = `${REPO_SLUG}-`;

type JsonObject = Record<string, unknown>;

type SupabaseProject = {
  ref: string;
  name: string;
  organizationId?: string;
};

type ProjectManifest = {
  repo: string;
  projectName: string;
  projectRef: string;
};

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

function stringProperty(source: JsonObject, key: string, context: string): string {
  const value = source[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${context}.${key} must be a non-empty string`);
  }
  return value;
}

function optionalStringProperty(source: JsonObject, key: string): string | undefined {
  const value = source[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function normalizeProject(raw: JsonObject): SupabaseProject {
  const ref = optionalStringProperty(raw, "ref") ?? optionalStringProperty(raw, "id");
  if (!ref) {
    throw new Error("project.ref or project.id must be a non-empty string");
  }
  return {
    ref,
    name: stringProperty(raw, "name", "project"),
    organizationId: optionalStringProperty(raw, "organization_id"),
  };
}

function normalizeProjects(rawProjects: Array<JsonObject>): SupabaseProject[] {
  return rawProjects.map(normalizeProject);
}

function projectsFromAuthenticatedCli(): SupabaseProject[] {
  const output = execFileSync("supabase", ["projects", "list", "--output", "json"], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  return normalizeProjects(JSON.parse(output) as Array<JsonObject>);
}

function normalizeManifest(raw: JsonObject): ProjectManifest {
  return {
    repo: stringProperty(raw, "repo", "manifest"),
    projectName: stringProperty(raw, "project_name", "manifest"),
    projectRef: stringProperty(raw, "project_ref", "manifest"),
  };
}

async function readManifest(): Promise<ProjectManifest | null> {
  let content: string;
  try {
    content = await readFile(".supabase-project.json", "utf8");
  } catch (error: unknown) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") {
      return null;
    }
    throw error;
  }
  return normalizeManifest(JSON.parse(content) as JsonObject);
}

async function main(): Promise<void> {
  const expectedRef = env("SUPABASE_PROJECT_REF");
  if (!expectedRef) {
    throw new Error("SUPABASE_PROJECT_REF is required");
  }

  const projects: SupabaseProject[] = env("SUPABASE_ACCESS_TOKEN")
    ? normalizeProjects(await apiFetch<Array<JsonObject>>("/projects"))
    : projectsFromAuthenticatedCli();
  const project = projects.find((item) => item.ref === expectedRef);
  if (!project) {
    throw new Error(`Project ref ${expectedRef} was not found in this Supabase account`);
  }

  if (!project.name.startsWith(PROJECT_PREFIX)) {
    throw new Error(`Project ${expectedRef} is named ${project.name} and does not match ${PROJECT_PREFIX}*`);
  }

  const organizationId = env("SUPABASE_ORG_ID");
  if (organizationId && project.organizationId && project.organizationId !== organizationId) {
    throw new Error(`Project ${expectedRef} does not belong to SUPABASE_ORG_ID=${organizationId}`);
  }

  const manifest = await readManifest();
  if (manifest) {
    if (manifest.repo !== REPO_SLUG) {
      throw new Error(".supabase-project.json belongs to a different repository");
    }
    if (manifest.projectRef !== expectedRef) {
      throw new Error(".supabase-project.json project_ref does not match SUPABASE_PROJECT_REF");
    }
    if (manifest.projectName !== project.name) {
      throw new Error(".supabase-project.json project_name does not match the remote project");
    }
  }

  console.log(`Verified ${expectedRef} (${project.name}) as an isolated project for ${REPO_SLUG}.`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
