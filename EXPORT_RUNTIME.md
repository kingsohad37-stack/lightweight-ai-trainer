# Exported trained model runtime

A tiny-transformer language-model export can be run locally with the bundled runtime.

1. Extract the downloaded model ZIP.
2. Put the bundled `trainer/` package under `runtime/trainer/` (the web app can package this runtime with the export).
3. Install the dependencies from `requirements-runtime.txt`.
4. Run `python run_chat.py`.

The launcher keeps the full conversation in memory and formats turns as `System`, `User`, and `Assistant` so exported language models can be used interactively rather than only inspected as checkpoint files.

Classification/regression/clustering exports remain supported by the existing application download format; this launcher is specifically for tiny-transformer language-model exports.
