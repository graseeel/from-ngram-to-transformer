from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote

import httpx

JsonObject = dict[str, object]


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


@dataclass(frozen=True)
class AuthSession:
    raw: JsonObject
    access_token: str | None

    @classmethod
    def from_payload(cls, payload: JsonObject) -> AuthSession:
        token = payload.get("access_token")
        if token is not None and not isinstance(token, str):
            raise ValueError("Supabase auth response access_token must be a string")
        return cls(raw=payload, access_token=token)


@dataclass(frozen=True)
class GenerationInsert:
    model_type: str
    model_version_label: str
    prompt: str
    generated_text: str
    generation_params: JsonObject
    seed: int | None

    def to_payload(self) -> JsonObject:
        return {
            "model_type": self.model_type,
            "model_version_label": self.model_version_label,
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "generation_params": self.generation_params,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class GenerationRecord:
    id: str
    model_type: str
    model_version_label: str
    prompt: str
    generated_text: str
    generation_params: JsonObject
    seed: int | None
    created_at: str
    raw: JsonObject

    @classmethod
    def from_payload(cls, payload: JsonObject) -> GenerationRecord:
        params = payload.get("generation_params")
        seed = payload.get("seed")
        return cls(
            id=_string_field(payload, "id"),
            model_type=_string_field(payload, "model_type"),
            model_version_label=_string_field(payload, "model_version_label"),
            prompt=_string_field(payload, "prompt"),
            generated_text=_string_field(payload, "generated_text"),
            generation_params=params if isinstance(params, dict) else {},
            seed=seed if isinstance(seed, int) else None,
            created_at=_string_field(payload, "created_at"),
            raw=payload,
        )


def _json_object(response: httpx.Response) -> JsonObject:
    data: object = response.json()
    if not isinstance(data, dict):
        raise ValueError("Supabase response must be a JSON object")
    return data


def _json_object_list(response: httpx.Response) -> list[JsonObject]:
    data: object = response.json()
    if not isinstance(data, list):
        raise ValueError("Supabase response must be a JSON list")
    rows: list[JsonObject] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Supabase list response contained a non-object row")
        rows.append(item)
    return rows


def _string_field(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Supabase row field {key!r} must be a string")
    return value


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

    async def sign_up(self, email: str, password: str) -> AuthSession:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.url}/auth/v1/signup",
                headers=self._headers(),
                json={"email": email, "password": password},
            )
        response.raise_for_status()
        return AuthSession.from_payload(_json_object(response))

    async def sign_in(self, email: str, password: str) -> AuthSession:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.url}/auth/v1/token?grant_type=password",
                headers=self._headers(),
                json={"email": email, "password": password},
            )
        response.raise_for_status()
        return AuthSession.from_payload(_json_object(response))

    async def save_generation(
        self,
        access_token: str,
        generation: GenerationInsert,
    ) -> GenerationRecord:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.url}/rest/v1/generations",
                headers={**self._headers(access_token), "Prefer": "return=representation"},
                json=generation.to_payload(),
            )
        response.raise_for_status()
        rows = _json_object_list(response)
        if not rows:
            raise ValueError("Supabase did not return the inserted generation")
        return GenerationRecord.from_payload(rows[0])

    async def list_generations(self, access_token: str) -> list[GenerationRecord]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                f"{self.settings.url}/rest/v1/generations"
                "?select=id,model_type,model_version_label,prompt,generated_text,"
                "generation_params,seed,created_at&order=created_at.desc",
                headers=self._headers(access_token),
            )
        response.raise_for_status()
        return [GenerationRecord.from_payload(row) for row in _json_object_list(response)]

    async def delete_generation(self, access_token: str, generation_id: str) -> None:
        encoded_id = quote(generation_id, safe="")
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.delete(
                f"{self.settings.url}/rest/v1/generations?id=eq.{encoded_id}",
                headers=self._headers(access_token),
            )
        response.raise_for_status()
