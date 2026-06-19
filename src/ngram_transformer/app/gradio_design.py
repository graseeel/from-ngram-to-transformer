from __future__ import annotations

from html import escape

from ngram_transformer.app.schemas import ModelInfo

HERO_HTML = """
<section class="ngt-hero" aria-labelledby="ngt-title">
  <div class="ngt-hero-copy">
    <p class="ngt-kicker">Character language model lab</p>
    <h1 id="ngt-title">From N-gram to Transformer</h1>
    <p class="ngt-lede">
      Compare a smoothed character N-gram with a compact decoder-only Transformer using
      the same prompt, sampler settings, and Supabase-backed generation history.
    </p>
  </div>
  <div class="ngt-hero-facts" aria-label="Project stack">
    <span>FastAPI</span>
    <span>PyTorch</span>
    <span>Supabase</span>
  </div>
</section>
"""

PROMPT_PRESETS = (
    ("Corpus seed", "The old model"),
    ("Transformer seed", "A tiny transformer learns"),
    ("Language modeling", "Language models begin"),
)

DEMO_CSS = """
:root {
  --ngt-bg: oklch(0.09 0 0);
  --ngt-bg-2: oklch(0.13 0.016 188);
  --ngt-surface: oklch(0.18 0.018 188);
  --ngt-surface-2: oklch(0.23 0.02 188);
  --ngt-border: oklch(0.34 0.03 188);
  --ngt-border-strong: oklch(0.51 0.08 188);
  --ngt-text: oklch(0.96 0.006 188);
  --ngt-muted: oklch(0.75 0.03 188);
  --ngt-primary: oklch(0.72 0.1 188);
  --ngt-primary-ink: oklch(0.09 0 0);
  --ngt-accent: oklch(0.7 0.13 64);
  --ngt-danger: oklch(0.66 0.18 25);
  --ngt-success: oklch(0.72 0.12 150);
  --ngt-focus: oklch(0.82 0.11 188);
  --ngt-radius: 14px;
  --ngt-ease-out: cubic-bezier(0.23, 1, 0.32, 1);
}

body,
.gradio-container {
  background:
    radial-gradient(circle at 12% 0%, oklch(0.72 0.1 188 / 0.12), transparent 34rem),
    linear-gradient(145deg, var(--ngt-bg), var(--ngt-bg-2));
  color: var(--ngt-text);
  font-family:
    "IBM Plex Sans",
    Inter,
    ui-sans-serif,
    system-ui,
    -apple-system,
    BlinkMacSystemFont,
    "Segoe UI",
    sans-serif;
}

.gradio-container {
  min-height: 100dvh;
}

.gradio-container .contain {
  max-width: none;
}

#ngt-shell {
  width: min(1160px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 0 0 40px;
}

.ngt-hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 20px 24px;
  border: 1px solid var(--ngt-border);
  border-radius: 16px;
  background:
    linear-gradient(135deg, oklch(0.2 0.025 188 / 0.92), oklch(0.14 0.01 188 / 0.96)),
    var(--ngt-surface);
}

.ngt-hero h1 {
  max-width: none;
  margin: 0;
  color: var(--ngt-text);
  font-size: 2.25rem;
  line-height: 1;
  letter-spacing: -0.03em;
  text-wrap: balance;
}

.ngt-kicker {
  margin: 0 0 10px;
  color: var(--ngt-primary);
  font-size: 0.95rem;
  font-weight: 700;
}

.ngt-lede {
  max-width: 66ch;
  margin: 12px 0 0;
  color: var(--ngt-muted);
  font-size: 0.96rem;
  line-height: 1.55;
  text-wrap: pretty;
}

.ngt-hero-facts {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  min-width: 220px;
}

.ngt-hero-facts span,
.ngt-status-pill {
  min-height: 32px;
  padding: 7px 11px;
  border: 1px solid var(--ngt-border);
  border-radius: 999px;
  color: var(--ngt-text);
  background: oklch(0.24 0.024 188 / 0.78);
  font-size: 0.85rem;
  font-weight: 700;
}

.ngt-panel {
  border: 1px solid var(--ngt-border);
  border-radius: var(--ngt-radius);
  background: oklch(0.16 0.014 188 / 0.94);
}

.ngt-panel,
.ngt-panel > .block {
  overflow: visible;
}

.ngt-section-title h2,
.ngt-section-title h3 {
  margin: 0;
  color: var(--ngt-text);
  font-size: 1rem;
  line-height: 1.3;
  letter-spacing: 0;
}

.ngt-section-title p {
  margin: 8px 0 0;
  color: var(--ngt-muted);
  line-height: 1.55;
}

.ngt-model-status {
  display: grid;
  gap: 10px;
  margin: 0;
  padding: 0;
}

.ngt-model-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 12px;
  border: 1px solid var(--ngt-border);
  border-radius: 12px;
  background: oklch(0.2 0.018 188 / 0.82);
}

.ngt-model-row > div {
  min-width: 0;
}

.ngt-model-row strong {
  color: var(--ngt-text);
}

.ngt-model-row small {
  display: block;
  margin-top: 4px;
  color: var(--ngt-muted);
  line-height: 1.4;
  overflow-wrap: anywhere;
}

.ngt-status-pill {
  flex: 0 0 auto;
}

.ngt-status-pill[data-ready="true"] {
  border-color: oklch(0.72 0.12 150 / 0.72);
  color: oklch(0.9 0.08 150);
}

.ngt-status-pill[data-ready="false"] {
  border-color: oklch(0.7 0.13 64 / 0.72);
  color: oklch(0.9 0.1 64);
}

.ngt-output textarea,
.ngt-history textarea,
.ngt-prompt textarea {
  font-family: ui-monospace, "SFMono-Regular", Menlo, Monaco, Consolas, monospace;
  line-height: 1.55;
}

.ngt-output textarea:disabled,
.ngt-history textarea:disabled {
  color: oklch(0.15 0.014 188) !important;
  -webkit-text-fill-color: oklch(0.15 0.014 188) !important;
  opacity: 1 !important;
}

.ngt-status-box {
  min-height: 44px;
}

button.ngt-primary,
button.ngt-secondary,
.ngt-primary button,
.ngt-secondary button {
  min-height: 44px;
  border-radius: 12px;
  font-weight: 800;
  transition:
    transform 160ms var(--ngt-ease-out),
    background-color 160ms ease,
    border-color 160ms ease;
  touch-action: manipulation;
}

button.ngt-primary,
.ngt-primary button {
  border-color: transparent;
  color: var(--ngt-primary-ink);
  background: var(--ngt-primary);
}

button.ngt-secondary,
.ngt-secondary button {
  border-color: var(--ngt-border);
  color: var(--ngt-text);
  background: var(--ngt-surface-2);
}

button.ngt-primary:active,
button.ngt-secondary:active,
.ngt-primary button:active,
.ngt-secondary button:active {
  transform: scale(0.98);
}

.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container button:focus-visible,
.gradio-container [role="button"]:focus-visible {
  outline: 3px solid var(--ngt-focus);
  outline-offset: 2px;
}

.gradio-container label,
.gradio-container .label-wrap span {
  color: var(--ngt-text);
}

.gradio-container .info,
.gradio-container .prose p,
.gradio-container .prose li {
  color: var(--ngt-muted);
}

@media (hover: hover) and (pointer: fine) {
  button.ngt-primary:hover,
  button.ngt-secondary:hover,
  .ngt-primary button:hover,
  .ngt-secondary button:hover {
    transform: translateY(-1px);
  }
}

@media (max-width: 760px) {
  #ngt-shell {
    width: min(100vw - 20px, 100%);
    padding: 0 0 32px;
  }

  .ngt-hero {
    align-items: flex-start;
    flex-direction: column;
    padding: 22px;
  }

  .ngt-hero h1 {
    max-width: 11ch;
    font-size: 2.15rem;
  }

  .ngt-hero-facts {
    justify-content: flex-start;
    min-width: 0;
  }
}

@media (prefers-reduced-motion: reduce) {
  .ngt-primary button,
  .ngt-secondary button,
  button.ngt-primary,
  button.ngt-secondary {
    transition: background-color 160ms ease, border-color 160ms ease;
  }

  button.ngt-primary:active,
  button.ngt-secondary:active,
  button.ngt-primary:hover,
  button.ngt-secondary:hover,
  .ngt-primary button:active,
  .ngt-secondary button:active,
  .ngt-primary button:hover,
  .ngt-secondary button:hover {
    transform: none;
  }
}

footer,
.gradio-container footer {
  display: none !important;
}
"""


def model_status_html(models: list[ModelInfo]) -> str:
    rows = []
    for model in models:
        readiness = "Ready" if model.ready else "Missing checkpoint"
        version = model.version_label or model.notes
        rows.append(
            '<div class="ngt-model-row">'
            f"<div><strong>{escape(model.name)}</strong><small>{escape(version)}</small></div>"
            f'<span class="ngt-status-pill" data-ready="{str(model.ready).lower()}">'
            f"{escape(readiness)}</span>"
            "</div>",
        )
    return '<div class="ngt-model-status">' + "".join(rows) + "</div>"
