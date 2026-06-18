import { randomBytes } from "node:crypto";
import { writeFile } from "node:fs/promises";

const API_BASE = "https://api.supabase.com/v1";
const REPO_SLUG = "from-ngram-to-transformer";
const PROJECT_PREFIX = `${REPO_SLUG}-`;

type JsonObject = Record<string, unknown>;

function env(name: string): string | undefined {
  const value = process.env[name]?.trim();
  return value && value.length > 0 ? value : undefined;
}

function missing(names: string[]): string[] {
  return names.filter((name) => !env(name));
}

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = env("SUPABASE_ACCESS_TOKEN");
  if (!token) {
    throw new Error("SUPABASE_ACCESS_TOKEN is required");
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Supabase API ${response.status} ${response.statusText}: ${body}`);
  }
  return (await response.json()) as T;
}

async function resolveOrganizationId(): Promise<string> {
  const orgId = env("SUPABASE_ORG_ID");
  if (orgId) {
    return orgId;
  }
  const slug = env("SUPABASE_ORG_SLUG");
  if (!slug) {
    throw new Error("Set SUPABASE_ORG_ID or SUPABASE_ORG_SLUG");
  }
  const organizations = await apiFetch<Array<JsonObject>>("/organizations");
  const organization = organizations.find((item) => item.slug === slug);
  if (!organization || typeof organization.id !== "string") {
    throw new Error(`No organization found for slug ${slug}`);
  }
  return organization.id;
}

function projectRef(project: JsonObject): string | undefined {
  const ref = project.ref ?? project.id;
  return typeof ref === "string" ? ref : undefined;
}

async function main(): Promise<void> {
  const missingRequired = missing(["SUPABASE_ACCESS_TOKEN", "SUPABASE_DB_PASSWORD"]);
  if (!env("SUPABASE_ORG_ID") && !env("SUPABASE_ORG_SLUG")) {
    missingRequired.push("SUPABASE_ORG_ID or SUPABASE_ORG_SLUG");
  }
  if (missingRequired.length > 0) {
    console.error("Cannot provision remote Supabase project. Missing:");
    for (const name of missingRequired) {
      console.error(`- ${name}`);
    }
    console.error("No existing Supabase project was selected or modified.");
    process.exitCode = 2;
    return;
  }

  const organizationId = await resolveOrganizationId();
  const projectName = env("SUPABASE_PROJECT_NAME") ?? `${PROJECT_PREFIX}${randomBytes(4).toString("hex")}`;
  if (!projectName.startsWith(PROJECT_PREFIX)) {
    throw new Error(`Project name must start with ${PROJECT_PREFIX}`);
  }

  const projects = await apiFetch<Array<JsonObject>>("/projects");
  const collision = projects.find((project) => project.name === projectName);
  if (collision) {
    throw new Error(
      `A project named ${projectName} already exists. Choose a different SUPABASE_PROJECT_NAME.`,
    );
  }

  const created = await apiFetch<JsonObject>("/projects", {
    method: "POST",
    body: JSON.stringify({
      organization_id: organizationId,
      name: projectName,
      region: env("SUPABASE_REGION") ?? "us-east-1",
      db_pass: env("SUPABASE_DB_PASSWORD"),
      plan: env("SUPABASE_PLAN") ?? "free",
    }),
  });
  const ref = projectRef(created);
  if (!ref) {
    throw new Error(`Supabase create response did not include a project ref: ${JSON.stringify(created)}`);
  }

  const manifest = {
    repo: REPO_SLUG,
    project_name: projectName,
    project_ref: ref,
    organization_id: organizationId,
    created_at: new Date().toISOString(),
  };
  await writeFile(".supabase-project.json", `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  console.log("Created isolated Supabase project for this repository.");
  console.log(`SUPABASE_PROJECT_NAME=${projectName}`);
  console.log(`SUPABASE_PROJECT_REF=${ref}`);
  console.log("Wrote .supabase-project.json without secrets.");
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
