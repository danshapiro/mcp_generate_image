# Repository Guidelines

## Project Structure & Modules
- `server.py`: MCP stdio server exposing the `image_generate` tool; Gemini client, logging, WebP conversion.
- `requirements.txt`: minimal runtime deps (google-genai, Pillow, fastmcp).
- `README.md`: only end-user doc; keep it the single source for user-facing instructions.
- `outputs/`: default sample output location; other `output_dir` paths must already exist before calls.
- Virtual envs: Windows `./.venv`, WSL `./.venv-wsl`; prefer running commands from WSL/bash.

## Build, Run, and Dev Commands
- Install deps (Windows):  
  ```pwsh
  python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
  ```
- Install deps (WSL):  
  ```bash
  python -m venv .venv-wsl && source .venv-wsl/bin/activate && pip install -r requirements.txt
  ```
- Run the MCP server locally: `python server.py`
- Smoke-test via Codex (from repo):  
  ```bash
  codex exec --skip-git-repo-check "Call the mcp_generate_image.image_generate tool with prompt \"tiny line art\" and output_dir \"outputs\"."
  ```
  If `codex` is not on PATH, try Claude CLI:  
  ```bash
  claude mcp call mcp_generate_image image_generate --param prompt "tiny line art" --param output_dir outputs
  ```

## Coding Style & Naming
- Python, 4-space indents; keep type hints; follow existing helper layout in `server.py`.
- File/name stems derived from prompts: alnum plus `-`/`_`, spaces → `_`, max ~60 chars; collisions append `-001`, `-002`, etc.
- Logging: use `log_event(level, message, **fields)`; keep messages short and include structured context fields.
- Always output lossless WebP; do not hardcode model names in user-facing strings.

## Testing Guidelines
- No formal suite; run `python -m compileall server.py` after changes.
- Generate a quick image into `outputs/` and open with Pillow or an image viewer to confirm headers/decoding.
- If adding functionality, prefer small deterministic prompts for smoke checks.

## Commit & PR Guidelines
- Use concise, imperative commits (e.g., “Remove multi-image option”, “Harden inline decode errors”).
- Describe why a change is needed and how it was validated; link to bug reports or reproduction prompts when relevant.
- Keep end-user wording in `README.md`; agent/internal notes belong in code comments or this guide.

## Security & Configuration Tips
- Auth via env vars: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`/Vertex env). Do not commit keys.
- Ensure `output_dir` exists and is writable; reject calls otherwise.
- If behavior seems stale, restart the MCP process—no background daemon is required.
