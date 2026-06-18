# Supabase Setup

This repository requires a new, isolated Supabase project for remote demos. Never reuse an existing project, database, bucket, Edge Function, credential, URL, or project ref.

## Local Development

```bash
supabase start
supabase db reset
```

The local stack uses `supabase/config.toml`, `supabase/migrations`, and `supabase/seed.sql`. Tables in the public schema use explicit grants because new Supabase projects no longer automatically expose new tables to the Data API. RLS still controls row visibility after the tables are accessible.

## Remote Provisioning

Set the required environment variables:

```bash
export SUPABASE_ACCESS_TOKEN=...
export SUPABASE_ORG_ID=...
export SUPABASE_REGION=us-east-1
export SUPABASE_DB_PASSWORD='use-a-strong-generated-password'
```

Then run:

```bash
npm install
npm run supabase:provision
```

The script:

- lists existing projects only to avoid a name collision;
- creates a new project named `from-ngram-to-transformer-<random>`;
- writes `.supabase-project.json` without secrets;
- prints the new `SUPABASE_PROJECT_REF`.

It exits without creating or modifying anything if required credentials are missing.

## Verification

Set `SUPABASE_PROJECT_REF` to the new ref and run:

```bash
npm run supabase:verify
```

The verification script checks that the ref is present in the authenticated Supabase account, the project name starts with `from-ngram-to-transformer-`, and the local manifest matches when `.supabase-project.json` exists.

## Application Environment

Use the local values from `supabase start` or the remote values from the new project:

```bash
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_ANON_KEY=...
```

Only server-side code may use `SUPABASE_SERVICE_ROLE_KEY`. The demo uses user access tokens for history writes so RLS remains active.
