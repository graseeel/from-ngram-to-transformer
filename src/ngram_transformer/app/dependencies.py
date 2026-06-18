from __future__ import annotations

import os
from functools import lru_cache

from fastapi import HTTPException, status

from ngram_transformer.app.model_service import TextGenerationService
from ngram_transformer.infra.supabase_gateway import SupabaseGateway, SupabaseSettings


def bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token")
    return token


@lru_cache(maxsize=1)
def get_model_service() -> TextGenerationService:
    return TextGenerationService.from_config(os.getenv("MODEL_CONFIG_PATH", "configs/default.yaml"))


@lru_cache(maxsize=1)
def get_supabase_gateway() -> SupabaseGateway | None:
    settings = SupabaseSettings.from_env()
    return SupabaseGateway(settings) if settings else None


def require_gateway(gateway: SupabaseGateway | None) -> SupabaseGateway:
    if gateway is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase is not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY.",
        )
    return gateway


def require_access_token(authorization: str | None) -> str:
    token = bearer_token(authorization)
    if token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="login required")
    return token
