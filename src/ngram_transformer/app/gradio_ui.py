from __future__ import annotations

from collections.abc import Callable

import httpx
from fastapi import FastAPI

from ngram_transformer.app.dependencies import get_model_service, get_supabase_gateway
from ngram_transformer.app.generation_history import format_history, save_generation_result
from ngram_transformer.app.gradio_design import (
    DEMO_CSS,
    HERO_HTML,
    PROMPT_PRESETS,
    model_status_html,
)
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


def _prompt_preset(prompt: str) -> Callable[[], str]:
    def select_prompt() -> str:
        return prompt

    return select_prompt


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

    model_status = model_status_html(service.list_models())

    with gr.Blocks(
        title="From N-gram to Transformer",
        css=DEMO_CSS,
        fill_width=True,
    ) as demo:
        token_state = gr.State("")
        with gr.Column(elem_id="ngt-shell"):
            gr.HTML(HERO_HTML, padding=False)
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, elem_classes="ngt-panel"):
                    gr.Markdown(
                        "### Prompt lab\nWrite a seed text, choose a model, and tune the sampler.",
                        elem_classes="ngt-section-title",
                    )
                    model = gr.Dropdown(choices=choices, value="ngram", label="Model")
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="The old model",
                        lines=4,
                        max_lines=8,
                        elem_classes="ngt-prompt",
                    )
                    with gr.Row():
                        for label, preset in PROMPT_PRESETS:
                            preset_button = gr.Button(
                                label,
                                size="sm",
                                elem_classes="ngt-secondary",
                            )
                            preset_button.click(_prompt_preset(preset), outputs=[prompt])
                    with gr.Row():
                        max_tokens = gr.Slider(
                            1,
                            300,
                            value=120,
                            step=1,
                            label="Max new tokens",
                        )
                        temperature = gr.Slider(
                            0.1,
                            2.0,
                            value=0.9,
                            step=0.1,
                            label="Temperature",
                        )
                    with gr.Row():
                        top_k = gr.Slider(0, 100, value=20, step=1, label="Top-k")
                        top_p = gr.Slider(0, 1, value=0.95, step=0.05, label="Top-p")
                        seed = gr.Number(value=1337, label="Seed")
                    save = gr.Checkbox(value=False, label="Save generation to Supabase history")
                    generate_button = gr.Button(
                        "Generate text",
                        variant="primary",
                        elem_classes="ngt-primary",
                    )
                with gr.Column(scale=1, elem_classes="ngt-panel"):
                    gr.Markdown(
                        "### Run state\nModel readiness and optional Supabase session.",
                        elem_classes="ngt-section-title",
                    )
                    gr.HTML(model_status, padding=False)
                    email = gr.Textbox(label="Email", type="email")
                    password = gr.Textbox(label="Password", type="password")
                    with gr.Row():
                        signup_button = gr.Button(
                            "Create account",
                            elem_classes="ngt-secondary",
                        )
                        login_button = gr.Button("Login", elem_classes="ngt-primary")
                    auth_status = gr.Markdown(
                        "Not logged in.",
                        elem_classes="ngt-status-box",
                    )
            with gr.Tabs():
                with gr.Tab("Generated text"):
                    generated = gr.Textbox(
                        label="Generated text",
                        lines=10,
                        elem_classes="ngt-output",
                    )
                    generation_status = gr.Markdown(
                        "No generation yet.",
                        elem_classes="ngt-status-box",
                    )
                with gr.Tab("History"):
                    history_button = gr.Button(
                        "Refresh history",
                        elem_classes="ngt-secondary",
                    )
                    history_box = gr.Textbox(
                        label="History",
                        lines=12,
                        elem_classes="ngt-history",
                    )

        signup_button.click(signup, inputs=[email, password], outputs=[auth_status])
        login_button.click(login, inputs=[email, password], outputs=[token_state, auth_status])
        generate_button.click(
            generate_ui,
            inputs=[token_state, model, prompt, max_tokens, temperature, top_k, top_p, seed, save],
            outputs=[generated, generation_status],
        )
        history_button.click(history, inputs=[token_state], outputs=[history_box])

    gr.mount_gradio_app(app, demo, path="/demo")
