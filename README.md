## mcp_generate_image - Let Your Agent Make Art

Give your LLM a paintbrush. This MCP server turns prompts into lossless WebP files and drops them into a folder you choose. You never call it directly; you connect it to your MCP-capable client and let the model drive.

### What it does
- Generates one or more lossless WebP images from descriptive prompts.
- Handles reference images (URLs or local files), aspect ratios, and size presets.
- Can route to FAL models when `FAL_KEY`/`FAL_API_KEY` is set; otherwise uses Gemini 3 Pro Image Preview.

### Quick install for common MCP clients
1) Clone this repo and install deps:
```bash
python -m venv .venv && . .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m venv .venv-wsl && source .venv-wsl/bin/activate && pip install -r requirements.txt  # optional: WSL
```
2) Copy `.env.example` to `.env` and add keys:
```
GEMINI_API_KEY=...      # or GOOGLE_API_KEY / Vertex envs
FAL_KEY=...             # or FAL_API_KEY
```
3) Make sure `outputs/` exists: `mkdir outputs`

#### Claude Code (CLI or Desktop)
Add a stdio server with the built-in MCP helper (server.py self-selects the right venv):  
```bash
claude mcp add --transport stdio mcp_generate_image \
  --env GEMINI_API_KEY=$GEMINI_API_KEY --env FAL_KEY=$FAL_KEY -- python server.py
```
Restart Claude Code/Desktop; the tools appear automatically.

#### Codex CLI / Codex IDE
Codex also ships `codex mcp` helpers. Add this repo as a stdio server:  
```bash
codex mcp add mcp_generate_image \
  --env GEMINI_API_KEY=$GEMINI_API_KEY --env FAL_KEY=$FAL_KEY -- python server.py
```
Then chat in Codex; it will launch the server and call the tool when needed.

#### Cursor IDE
Open Settings -> MCP Servers -> Add:
- Type: `command`
- Command: `python`
- Args: `server.py`
- Env: set `GEMINI_API_KEY` and `FAL_KEY`  
Save and restart Cursor; the model will call the tool directly.

#### Other MCP clients
Any stdio-capable MCP client can launch `python server.py` with the same env vars. Keep `.env` in the project root so your keys load.

### Options the model can use
- `prompt` (required): what to draw; include style, lighting, lenses, text, layout.
- `output_dir` (required): existing writable directory; files are saved as lossless WebP.
- `reference_images` (optional): list of 1-14 local paths or http(s) URLs (PNG/JPEG/WebP, <=20MB each). Describe their content in the prompt, not filenames.
- `aspect_ratio` (optional): 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, 9:21.
- `image_size` (optional, default 1K): 1K | 2K | 4K.
- `model` (optional, shown only when a FAL key is present): `z-image turbo (fast)` | `nano banana pro (best)`. FAL calls force `output_format=webp`; returned WebPs are written as-is, other formats are converted once to lossless WebP.

### Behavior and guarantees
- Output is always WebP; filenames are prompt-sanitized (spaces -> `_`, max ~60 chars, collisions get `-001`, `-002`, ...).
- Validation catches empty prompts, missing/non-directory output paths, oversized/invalid reference images.
- With a FAL key and `model` set, requests route to FAL; otherwise Gemini is used.

### Troubleshooting
- "model is only available..." -> set `FAL_KEY` or `FAL_API_KEY` and retry.
- "output_dir ... does not exist" -> create the folder first.
- If a client can't find the tool schema, restart the client so it reloads `mcp.json`.
