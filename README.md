# From N-gram to Transformer

Practical portfolio repository showing the path from a configurable character N-gram language model to a small decoder-only Transformer. The project is executable, testable, and wired for a Supabase-backed demo.

## Technical Decisions

- Shared preprocessing: both models use the same `.txt` corpus loading, normalization, deterministic split, and character tokenizer.
- Tokenization: character-level today, with a dedicated tokenizer module so subword tokenization can be added without changing model math.
- N-gram: add-k smoothing, configurable `N`, log-likelihood, perplexity, seeded generation, top-k, top-p, temperature, greedy decoding, and unknown-context fallback.
- Transformer: compact decoder-only PyTorch model with token/position embeddings, causal self-attention, residual connections, layer normalization, feed-forward blocks, gradient clipping, checkpoints, and resume support.
- App: FastAPI exposes the API; Gradio is mounted at `/demo` for a lightweight authenticated UI.
- Supabase: auth, generation history, experiments, model versions, reports, and storage buckets are isolated from ML code through an infrastructure gateway.
- Metrics: reports are written only from actual training/evaluation runs. No benchmark numbers are hardcoded.

## Data Model

Supabase tables:

- `profiles`: auth user profile rows created from `auth.users`.
- `experiments`: training runs, config, source corpus/license, metrics, and explicit `is_public` publication flag.
- `model_versions`: checkpoint/config/metric metadata for exact model versions.
- `generations`: user-owned prompts, outputs, parameters, seed, and model version label.
- `evaluation_reports`: metrics and side-by-side sample metadata.

Storage buckets:

- `model-artifacts`: private model/tokenizer/checkpoint artifacts.
- `evaluation-reports`: private report outputs.
- `public-demo-assets`: public assets only when needed by the demo.

RLS is enabled for all public tables. Users can only read/delete their own generations. Public experiment visibility requires `is_public = true`.

## Directory Structure

```text
configs/                 YAML configs for data, N-gram, Transformer, generation
data/corpus/             Tiny CC0 corpus and metadata
docs/                    Architecture and Supabase setup notes
scripts/                 Supabase remote provisioning and verification scripts
src/ngram_transformer/   Python package
supabase/                Local config, migrations, seed data
tests/                   Unit, API, SQL, and optional local Supabase tests
```

## Work Stages

1. Prepare data and tokenizer -> verify encode/decode, persistence, split metadata.
2. Train/evaluate N-gram -> verify probabilities, unknown context fallback, deterministic generation.
3. Train/evaluate Transformer -> verify causal mask, tensor shapes, forward pass, checkpoint resume.
4. Run FastAPI/Gradio demo -> verify generation, auth flow, history save/list/delete.
5. Provision isolated Supabase project -> verify project ref/name match this repository.
6. Run CI checks -> verify Ruff, Mypy, Pytest, TypeScript scripts.

## Completion Criteria

- `uv run ruff check .` passes.
- `uv run mypy src tests` passes.
- `uv run pytest` passes.
- `npm run typecheck:scripts` passes.
- `supabase db reset` applies migrations when local Supabase/Docker is available.
- `npm run supabase:verify` confirms the remote project ref belongs to a new `from-ngram-to-transformer-*` project when credentials are configured.

## Quick Start

```bash
uv sync --extra dev --python 3.12
uv run ngram-transformer-train-ngram --config configs/default.yaml
uv run ngram-transformer-train-transformer --config configs/default.yaml --output-dir artifacts/transformer
uv run ngram-transformer-evaluate
uv run ngram-transformer-api
```

Open `http://127.0.0.1:8000/demo` for the Gradio demo.

## Supabase

Local tests and development use the Supabase CLI:

```bash
supabase start
supabase db reset
```

Remote provisioning requires credentials and always creates a new isolated project:

```bash
npm install
npm run supabase:provision
npm run supabase:verify
```

See [docs/supabase-setup.md](docs/supabase-setup.md).
