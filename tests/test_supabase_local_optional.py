import os
import subprocess

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_SUPABASE_LOCAL_TESTS") != "1",
    reason="set RUN_SUPABASE_LOCAL_TESTS=1 to run local Supabase CLI integration checks",
)
def test_supabase_migrations_apply_locally() -> None:
    result = subprocess.run(
        ["supabase", "db", "reset"],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr + result.stdout
