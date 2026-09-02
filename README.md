# Lightweight AI Trainer

This repository contains the deployable source bundle and Render configuration for Lightweight AI Trainer. `trainer.zip` remains the original full application source; the Docker build adds a small compatibility/extension layer without removing the existing trainer.

## Render / AI providers

Set `TRAINER_API_KEY` before using a public deployment. Optional server-side AI provider variables are:

- `AI_PROVIDER=gemini|groq|openai|custom`
- `GEMINI_API_KEY`, `GEMINI_MODEL`
- `GROQ_API_KEY`, `GROQ_MODEL`
- `OPENAI_API_KEY`, `OPENAI_MODEL`
- `AI_API_KEY`, `AI_MODEL`, `AI_BASE_URL` for an OpenAI-compatible provider

Users can also supply a provider key per request in the web UI; it is not stored by the application.

The application now includes:

- real dataset analysis, planning, training, prediction, and tiny-transformer text generation
- clear handling when Generate Text is used with a non-language-model experiment
- Gemini, Groq, OpenAI, and custom OpenAI-compatible AI requests
- downloadable trained-model ZIPs containing the latest checkpoints and manifest

Render builds the Docker image directly from this repository and listens on the platform-provided `PORT`.
