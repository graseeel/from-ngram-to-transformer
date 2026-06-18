create extension if not exists pgcrypto;

create schema if not exists private;
revoke all on schema private from public, anon, authenticated;

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.experiments (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  description text,
  model_type text not null check (model_type in ('ngram', 'transformer')),
  config jsonb not null,
  metrics jsonb,
  source_corpus text not null,
  source_license text not null,
  is_public boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.model_versions (
  id uuid primary key default gen_random_uuid(),
  experiment_id uuid not null references public.experiments(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  model_type text not null check (model_type in ('ngram', 'transformer')),
  version_label text not null,
  config_hash text not null,
  config jsonb not null,
  metrics jsonb not null default '{}'::jsonb,
  parameter_count integer,
  artifact_bucket text,
  artifact_path text,
  tokenizer_path text,
  checkpoint_path text,
  created_at timestamptz not null default now(),
  unique (owner_id, version_label)
);

create table public.generations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  model_version_id uuid references public.model_versions(id) on delete set null,
  model_type text not null check (model_type in ('ngram', 'transformer')),
  model_version_label text not null,
  prompt text not null check (length(prompt) between 1 and 2000),
  generated_text text not null,
  generation_params jsonb not null,
  seed integer,
  created_at timestamptz not null default now()
);

create table public.evaluation_reports (
  id uuid primary key default gen_random_uuid(),
  experiment_id uuid not null references public.experiments(id) on delete cascade,
  model_version_id uuid references public.model_versions(id) on delete set null,
  owner_id uuid not null references auth.users(id) on delete cascade,
  report_bucket text not null default 'evaluation-reports',
  report_path text,
  metrics jsonb not null,
  sample_table jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

create index experiments_owner_id_idx on public.experiments(owner_id);
create index experiments_public_idx on public.experiments(is_public) where is_public;
create index model_versions_owner_id_idx on public.model_versions(owner_id);
create index model_versions_experiment_id_idx on public.model_versions(experiment_id);
create index generations_user_id_created_at_idx on public.generations(user_id, created_at desc);
create index evaluation_reports_owner_id_idx on public.evaluation_reports(owner_id);
create index evaluation_reports_experiment_id_idx on public.evaluation_reports(experiment_id);

alter table public.profiles enable row level security;
alter table public.experiments enable row level security;
alter table public.model_versions enable row level security;
alter table public.generations enable row level security;
alter table public.evaluation_reports enable row level security;

grant usage on schema public to anon, authenticated;
grant select, insert, update, delete on public.profiles to authenticated;
grant select, insert, update, delete on public.experiments to authenticated;
grant select, insert, update, delete on public.model_versions to authenticated;
grant select, insert, delete on public.generations to authenticated;
grant select, insert, update, delete on public.evaluation_reports to authenticated;

create policy "profiles are visible to the owning user"
on public.profiles
for select
to authenticated
using (id = (select auth.uid()));

create policy "profiles are inserted by the owning user"
on public.profiles
for insert
to authenticated
with check (id = (select auth.uid()));

create policy "profiles are updated by the owning user"
on public.profiles
for update
to authenticated
using (id = (select auth.uid()))
with check (id = (select auth.uid()));

create policy "experiments are visible to owners or when published"
on public.experiments
for select
to authenticated
using (owner_id = (select auth.uid()) or is_public);

create policy "experiments are inserted by owners"
on public.experiments
for insert
to authenticated
with check (owner_id = (select auth.uid()));

create policy "experiments are updated by owners"
on public.experiments
for update
to authenticated
using (owner_id = (select auth.uid()))
with check (owner_id = (select auth.uid()));

create policy "experiments are deleted by owners"
on public.experiments
for delete
to authenticated
using (owner_id = (select auth.uid()));

create policy "model versions follow experiment visibility"
on public.model_versions
for select
to authenticated
using (
  owner_id = (select auth.uid())
  or exists (
    select 1
    from public.experiments
    where experiments.id = model_versions.experiment_id
      and experiments.is_public
  )
);

create policy "model versions are inserted by owners"
on public.model_versions
for insert
to authenticated
with check (owner_id = (select auth.uid()));

create policy "model versions are updated by owners"
on public.model_versions
for update
to authenticated
using (owner_id = (select auth.uid()))
with check (owner_id = (select auth.uid()));

create policy "model versions are deleted by owners"
on public.model_versions
for delete
to authenticated
using (owner_id = (select auth.uid()));

create policy "generations are visible to their user"
on public.generations
for select
to authenticated
using (user_id = (select auth.uid()));

create policy "generations are inserted by their user"
on public.generations
for insert
to authenticated
with check (user_id = (select auth.uid()));

create policy "generations are deleted by their user"
on public.generations
for delete
to authenticated
using (user_id = (select auth.uid()));

create policy "evaluation reports follow experiment visibility"
on public.evaluation_reports
for select
to authenticated
using (
  owner_id = (select auth.uid())
  or exists (
    select 1
    from public.experiments
    where experiments.id = evaluation_reports.experiment_id
      and experiments.is_public
  )
);

create policy "evaluation reports are inserted by owners"
on public.evaluation_reports
for insert
to authenticated
with check (owner_id = (select auth.uid()));

create policy "evaluation reports are updated by owners"
on public.evaluation_reports
for update
to authenticated
using (owner_id = (select auth.uid()))
with check (owner_id = (select auth.uid()));

create policy "evaluation reports are deleted by owners"
on public.evaluation_reports
for delete
to authenticated
using (owner_id = (select auth.uid()));

create policy "private model artifacts are owned by uploader"
on storage.objects
for all
to authenticated
using (bucket_id = 'model-artifacts' and owner_id = (select auth.uid())::text)
with check (bucket_id = 'model-artifacts' and owner_id = (select auth.uid())::text);

create policy "private evaluation reports are owned by uploader"
on storage.objects
for all
to authenticated
using (bucket_id = 'evaluation-reports' and owner_id = (select auth.uid())::text)
with check (bucket_id = 'evaluation-reports' and owner_id = (select auth.uid())::text);

create policy "public demo assets are readable"
on storage.objects
for select
to anon, authenticated
using (bucket_id = 'public-demo-assets');

create policy "public demo assets are written by authenticated uploaders"
on storage.objects
for insert
to authenticated
with check (bucket_id = 'public-demo-assets' and owner_id = (select auth.uid())::text);

create or replace function private.touch_updated_at()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger profiles_touch_updated_at
before update on public.profiles
for each row execute function private.touch_updated_at();

create trigger experiments_touch_updated_at
before update on public.experiments
for each row execute function private.touch_updated_at();

create or replace function private.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, display_name)
  values (new.id, coalesce(new.raw_user_meta_data ->> 'display_name', split_part(new.email, '@', 1)))
  on conflict (id) do nothing;
  return new;
end;
$$;

create trigger on_auth_user_created
after insert on auth.users
for each row execute function private.handle_new_user();
