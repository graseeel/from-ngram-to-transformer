# Architecture

The repository is split into four layers.

## Domain and ML

`src/ngram_transformer/data` loads text, preserves meaningful punctuation, splits data, and owns tokenization. `src/ngram_transformer/ml` contains model math only: N-gram counting/probabilities, Transformer blocks, causal masking, sampling, and checkpoints.

## Training and Evaluation

`src/ngram_transformer/training` contains executable commands. Training scripts write artifacts and metrics under `artifacts/`; evaluation scripts read those artifacts and generate reports from held-out data.

## Application

`src/ngram_transformer/app` exposes FastAPI routes and mounts Gradio. The default app trains an in-memory N-gram from the small corpus when no artifact exists, so the demo can start without pretraining. The Transformer becomes available after a checkpoint exists.

## Infrastructure

`src/ngram_transformer/infra` contains the Supabase HTTP gateway. It uses the anon key plus the user's access token for PostgREST requests, allowing RLS to enforce row ownership. Service role keys are optional server-only credentials and must never be exposed in the browser.
