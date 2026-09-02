# Run an exported trained model

For a language-model export, extract the ZIP and run:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-runtime.txt
python run_chat.py
```

The export contains the trained checkpoint under `model/` and the required trainer runtime under `runtime/`, so it can run independently of the web application source tree.

For a one-shot prompt:

```bash
python run_chat.py --prompt "Hello"
```

`/clear` resets the conversation and `/exit` quits interactive chat.

Classification/regression exports remain usable for prediction through the web application; the standalone conversational launcher is for `tiny_transformer` language-model exports.
