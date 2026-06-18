from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status

from ngram_transformer.app.dependencies import (
    get_model_service,
    get_supabase_gateway,
    require_access_token,
    require_gateway,
)
from ngram_transformer.app.generation_history import save_generation_result
from ngram_transformer.app.gradio_ui import mount_gradio
from ngram_transformer.app.model_service import TextGenerationService
from ngram_transformer.app.schemas import AuthRequest, GenerateRequest, GenerateResponse, ModelInfo
from ngram_transformer.infra.supabase_gateway import JsonObject, SupabaseGateway


def _http_error(exc: httpx.HTTPStatusError) -> HTTPException:
    return HTTPException(status_code=exc.response.status_code, detail=exc.response.text)


def create_app() -> FastAPI:
    app = FastAPI(title="From N-gram to Transformer", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/models", response_model=list[ModelInfo])
    def models(
        service: Annotated[TextGenerationService, Depends(get_model_service)],
    ) -> list[ModelInfo]:
        return service.list_models()

    @app.post("/auth/signup")
    async def signup(
        request: AuthRequest,
        gateway: Annotated[SupabaseGateway | None, Depends(get_supabase_gateway)],
    ) -> JsonObject:
        try:
            return (await require_gateway(gateway).sign_up(request.email, request.password)).raw
        except httpx.HTTPStatusError as exc:
            raise _http_error(exc) from exc

    @app.post("/auth/login")
    async def login(
        request: AuthRequest,
        gateway: Annotated[SupabaseGateway | None, Depends(get_supabase_gateway)],
    ) -> JsonObject:
        try:
            return (await require_gateway(gateway).sign_in(request.email, request.password)).raw
        except httpx.HTTPStatusError as exc:
            raise _http_error(exc) from exc

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(
        request: GenerateRequest,
        service: Annotated[TextGenerationService, Depends(get_model_service)],
        gateway: Annotated[SupabaseGateway | None, Depends(get_supabase_gateway)],
        authorization: Annotated[str | None, Header()] = None,
    ) -> GenerateResponse:
        try:
            result = service.generate(request)
            if not request.save:
                return result
            token = require_access_token(authorization)
            return await save_generation_result(require_gateway(gateway), token, request, result)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise _http_error(exc) from exc

    @app.get("/generations")
    async def generations(
        gateway: Annotated[SupabaseGateway | None, Depends(get_supabase_gateway)],
        authorization: Annotated[str | None, Header()] = None,
    ) -> list[JsonObject]:
        try:
            records = await require_gateway(gateway).list_generations(
                require_access_token(authorization),
            )
        except httpx.HTTPStatusError as exc:
            raise _http_error(exc) from exc
        return [record.raw for record in records]

    @app.delete("/generations/{generation_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_generation(
        generation_id: str,
        gateway: Annotated[SupabaseGateway | None, Depends(get_supabase_gateway)],
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        try:
            await require_gateway(gateway).delete_generation(
                require_access_token(authorization),
                generation_id,
            )
        except httpx.HTTPStatusError as exc:
            raise _http_error(exc) from exc

    mount_gradio(app)
    return app
