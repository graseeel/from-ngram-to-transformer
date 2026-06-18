from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, Any

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status

from ngram_transformer.app.model_service import TextGenerationService
from ngram_transformer.app.schemas import AuthRequest, GenerateRequest, GenerateResponse, ModelInfo
from ngram_transformer.infra.supabase_gateway import SupabaseGateway, SupabaseSettings


def _bearer_token(authorization: str | None) -> str | None:
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


def _require_gateway(gateway: SupabaseGateway | None) -> SupabaseGateway:
    if gateway is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase is not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY.",
        )
    return gateway


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
    ) -> dict[str, Any]:
        try:
            return await _require_gateway(gateway).sign_up(request.email, request.password)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=exc.response.text,
            ) from exc

    @app.post("/auth/login")
    async def login(
        request: AuthRequest,
        gateway: Annotated[SupabaseGateway | None, Depends(get_supabase_gateway)],
    ) -> dict[str, Any]:
        try:
            return await _require_gateway(gateway).sign_in(request.email, request.password)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=exc.response.text,
            ) from exc

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(
        request: GenerateRequest,
        service: Annotated[TextGenerationService, Depends(get_model_service)],
        gateway: Annotated[SupabaseGateway | None, Depends(get_supabase_gateway)],
        authorization: Annotated[str | None, Header()] = None,
    ) -> GenerateResponse:
        try:
            result = service.generate(request)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        token = _bearer_token(authorization)
        if request.save:
            if token is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="login required to save",
                )
            try:
                row = await _require_gateway(gateway).save_generation(
                    token,
                    {
                        "model_type": result.model_name,
                        "model_version_label": result.model_version_label,
                        "prompt": result.prompt,
                        "generated_text": result.generated_text,
                        "generation_params": result.generation_params,
                        "seed": request.seed,
                    },
                )
                result.saved_generation_id = row.get("id")
            except httpx.HTTPStatusError as exc:
                raise HTTPException(
                    status_code=exc.response.status_code,
                    detail=exc.response.text,
                ) from exc
        return result

    @app.get("/generations")
    async def generations(
        gateway: Annotated[SupabaseGateway | None, Depends(get_supabase_gateway)],
        authorization: Annotated[str | None, Header()] = None,
    ) -> list[dict[str, Any]]:
        token = _bearer_token(authorization)
        if token is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="login required")
        try:
            return await _require_gateway(gateway).list_generations(token)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=exc.response.text,
            ) from exc

    @app.delete("/generations/{generation_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_generation(
        generation_id: str,
        gateway: Annotated[SupabaseGateway | None, Depends(get_supabase_gateway)],
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        token = _bearer_token(authorization)
        if token is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="login required")
        try:
            await _require_gateway(gateway).delete_generation(token, generation_id)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=exc.response.text,
            ) from exc

    mount_gradio(app)
    return app


def mount_gradio(app: FastAPI) -> None:
    import gradio as gr

    service = get_model_service()

    async def login(email: str, password: str) -> tuple[str, str]:
        gateway = get_supabase_gateway()
        if gateway is None:
            return "", "Supabase is not configured."
        try:
            payload = await gateway.sign_in(email, password)
        except httpx.HTTPError as exc:
            return "", f"Login failed: {exc}"
        return payload.get("access_token", ""), "Logged in."

    async def signup(email: str, password: str) -> str:
        gateway = get_supabase_gateway()
        if gateway is None:
            return "Supabase is not configured."
        try:
            await gateway.sign_up(email, password)
        except httpx.HTTPError as exc:
            return f"Signup failed: {exc}"
        return "Account created. You can log in now."

    async def generate_ui(
        token: str,
        model_name: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: float,
        top_p: float,
        seed: float,
        save: bool,
    ) -> tuple[str, str]:
        request = GenerateRequest(
            model_name=model_name,  # type: ignore[arg-type]
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=int(top_k) if top_k > 0 else None,
            top_p=top_p if top_p > 0 else None,
            seed=int(seed) if seed >= 0 else None,
            save=save,
        )
        try:
            response = service.generate(request)
        except ValueError as exc:
            return "", str(exc)
        if save:
            gateway = get_supabase_gateway()
            if not token or gateway is None:
                return (
                    response.generated_text,
                    "Generated but not saved: login and Supabase config required.",
                )
            try:
                row = await gateway.save_generation(
                    token,
                    {
                        "model_type": response.model_name,
                        "model_version_label": response.model_version_label,
                        "prompt": response.prompt,
                        "generated_text": response.generated_text,
                        "generation_params": response.generation_params,
                        "seed": request.seed,
                    },
                )
            except httpx.HTTPError as exc:
                return response.generated_text, f"Generated but save failed: {exc}"
            return response.generated_text, f"Saved generation {row.get('id')}"
        return response.generated_text, response.model_version_label

    async def history(token: str) -> str:
        gateway = get_supabase_gateway()
        if not token or gateway is None:
            return "Login and Supabase config required."
        try:
            rows = await gateway.list_generations(token)
        except httpx.HTTPError as exc:
            return f"History failed: {exc}"
        return "\n\n".join(
            f"{row['created_at']} | {row['model_version_label']}\n{row['generated_text']}"
            for row in rows
        )

    with gr.Blocks(title="From N-gram to Transformer") as demo:
        token_state = gr.State("")
        gr.Markdown("# From N-gram to Transformer")
        with gr.Row():
            email = gr.Textbox(label="Email")
            password = gr.Textbox(label="Password", type="password")
        with gr.Row():
            signup_button = gr.Button("Create account")
            login_button = gr.Button("Login")
        auth_status = gr.Textbox(label="Auth status", interactive=False)
        model = gr.Dropdown(choices=["ngram", "transformer"], value="ngram", label="Model")
        prompt = gr.Textbox(label="Prompt", value="The old model")
        with gr.Row():
            max_tokens = gr.Slider(1, 300, value=120, step=1, label="Max new tokens")
            temperature = gr.Slider(0.1, 2.0, value=0.9, step=0.1, label="Temperature")
            top_k = gr.Slider(0, 100, value=20, step=1, label="Top-k (0 disables)")
            top_p = gr.Slider(0, 1, value=0.95, step=0.05, label="Top-p (0 disables)")
            seed = gr.Number(value=1337, label="Seed (-1 disables)")
        save = gr.Checkbox(value=True, label="Save generation")
        generate_button = gr.Button("Generate")
        generated = gr.Textbox(label="Generated text", lines=8)
        generation_status = gr.Textbox(label="Generation status", interactive=False)
        history_button = gr.Button("Refresh history")
        history_box = gr.Textbox(label="History", lines=10)

        signup_button.click(signup, inputs=[email, password], outputs=[auth_status])
        login_button.click(login, inputs=[email, password], outputs=[token_state, auth_status])
        generate_button.click(
            generate_ui,
            inputs=[token_state, model, prompt, max_tokens, temperature, top_k, top_p, seed, save],
            outputs=[generated, generation_status],
        )
        history_button.click(history, inputs=[token_state], outputs=[history_box])

    gr.mount_gradio_app(app, demo, path="/demo")


app = create_app()


def run() -> None:
    uvicorn.run(
        "ngram_transformer.app.main:app",
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=os.getenv("APP_ENV") == "development",
    )


if __name__ == "__main__":
    run()
