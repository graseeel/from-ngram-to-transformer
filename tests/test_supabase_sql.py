from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = next((ROOT / "supabase" / "migrations").glob("*_initial_schema.sql"))


def test_migration_enables_rls_for_exposed_tables() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    for table in [
        "profiles",
        "experiments",
        "model_versions",
        "generations",
        "evaluation_reports",
    ]:
        assert f"alter table public.{table} enable row level security;" in sql


def test_generation_policy_is_user_isolated() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "generations are visible to their user" in sql
    assert "using (user_id = (select auth.uid()))" in sql
    assert "generations are deleted by their user" in sql


def test_public_experiments_require_explicit_flag() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "is_public boolean not null default false" in sql
    assert "owner_id = (select auth.uid()) or is_public" in sql


def test_private_security_definer_functions_are_not_in_public_schema() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "create schema if not exists private;" in sql
    assert "create or replace function private.handle_new_user()" in sql
    assert "security definer" in sql


def test_seed_defines_requested_buckets() -> None:
    seed = (ROOT / "supabase" / "seed.sql").read_text(encoding="utf-8")
    assert "'model-artifacts'" in seed
    assert "'evaluation-reports'" in seed
    assert "'public-demo-assets'" in seed
