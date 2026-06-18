from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    anon_key: str

    @classmethod
    def from_env(cls) -> SupabaseSettings | None:
        url = os.getenv("SUPABASE_URL")
        anon_key = os.getenv("SUPABASE_ANON_KEY")
        if not url or not anon_key:
            return None
        return cls(url=url.rstrip("/"), anon_key=anon_key)


class SupabaseGateway:
    """Thin HTTP adapter for Auth and PostgREST requests under user JWT RLS."""

    def __init__(self, settings: SupabaseSettings, timeout_seconds: float = 10.0) -> None:
        self.settings = settings
        self.timeout_seconds = timeout_seconds

    def _headers(self, access_token: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.settings.anon_key,
            "Content-Type": "application/json",
        }
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        return headers

    async def sign_up(self, email: str, password: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.url}/auth/v1/signup",
                headers=self._headers(),
                json={"email": email, "password": password},
            )
        response.raise_for_status()
        return response.json()

    async def sign_in(self, email: str, password: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.url}/auth/v1/token?grant_type=password",
                headers=self._headers(),
                json={"email": email, "password": password},
            )
        response.raise_for_status()
        return response.json()

    async def save_generation(self, access_token: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.url}/rest/v1/generations",
                headers={**self._headers(access_token), "Prefer": "return=representation"},
                json=payload,
            )
        response.raise_for_status()
        rows = response.json()
        return rows[0] if rows else {}

    async def list_generations(self, access_token: str) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                f"{self.settings.url}/rest/v1/generations"
                "?select=id,model_type,model_version_label,prompt,generated_text,"
                "generation_params,seed,created_at&order=created_at.desc",
                headers=self._headers(access_token),
            )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError("Supabase generations response was not a list")
        return data

    async def delete_generation(self, access_token: str, generation_id: str) -> None:
        encoded_id = quote(generation_id, safe="")
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.delete(
                f"{self.settings.url}/rest/v1/generations?id=eq.{encoded_id}",
                headers=self._headers(access_token),
            )
        response.raise_for_status()
