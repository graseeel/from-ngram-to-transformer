from __future__ import annotations

import httpx
from fastapi import FastAPI

from ngram_transformer.app.dependencies import get_model_service, get_supabase_gateway
from ngram_transformer.app.generation_history import format_history, save_generation_result
from ngram_transformer.app.schemas import GenerateRequest, ModelName, parse_model_name


def _generation_request(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: float,
    top_p: float,
    seed: float,
    save: bool,
) -> GenerateRequest:
    return GenerateRequest(
        model_name=parse_model_name(model_name),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=int(top_k) if top_k > 0 else None,
        top_p=top_p if top_p > 0 else None,
        seed=int(seed) if seed >= 0 else None,
        save=save,
    )


def mount_gradio(app: FastAPI) -> None:
    import gradio as gr

    service = get_model_service()
    choices: list[ModelName] = ["ngram", "transformer"]

    async def login(email: str, password: str) -> tuple[str, str]:
        gateway = get_supabase_gateway()
        if gateway is None:
            return "", "Supabase is not configured."
        try:
            session = await gateway.sign_in(email, password)
        except httpx.HTTPError as exc:
            return "", f"Login failed: {exc}"
        return session.access_token or "", "Logged in."

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
        try:
            request = _generation_request(
                model_name,
                prompt,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                seed,
                save,
            )
            response = service.generate(request)
        except ValueError as exc:
            return "", str(exc)
        if not save:
            return response.generated_text, response.model_version_label
        gateway = get_supabase_gateway()
        if not token or gateway is None:
            return (
                response.generated_text,
                "Generated but not saved: login and Supabase config required.",
            )
        try:
            saved = await save_generation_result(gateway, token, request, response)
        except httpx.HTTPError as exc:
            return response.generated_text, f"Generated but save failed: {exc}"
        return saved.generated_text, f"Saved generation {saved.saved_generation_id}"

    async def history(token: str) -> str:
        gateway = get_supabase_gateway()
        if not token or gateway is None:
            return "Login and Supabase config required."
        try:
            return format_history(await gateway.list_generations(token))
        except httpx.HTTPError as exc:
            return f"History failed: {exc}"

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
        model = gr.Dropdown(choices=choices, value="ngram", label="Model")
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
