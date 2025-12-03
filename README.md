## mcp_generate_image - Let Your Agent Make Art

Y

### Install (quick)
- Windows PowerShell:
```pwsh
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```
- WSL/bash:
```bash
python -m venv .venv-wsl && source .venv-wsl/bin/activate && pip install -r requirements.txt
```
- Env file: copy `.env.example` to `.env`, then fill:
```
GEMINI_API_KEY=...      # or GOOGLE_API_KEY / Vertex envs
FAL_KEY=...             # or FAL_API_KEY for the FAL models
```
- Create an output folder: `mkdir outputs`

### Generate an image
- Codex CLI:
```bash
codex exec --skip-git-repo-check "Call mcp_generate_image.image_generate with '{\"prompt\":\"line art coffee cup\",\"output_dir\":\"outputs\"}'"
```
- Claude CLI:
```bash
claude "Call mcp_generate_image.image_generate with {\"prompt\":\"tiny line art coffee mug\",\"output_dir\":\"outputs\",\"model\":\"z-image turbo (fast)\"}"
```
Send `"help"` if you want the live schema and option list.

### Options you can set
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

### Fresh sample renders (in `outputs/`)
- `outputs/futuristic_creative_workstation_at_night__laptop_running_an_.webp`
- `outputs/hand-drawn_line_art_schematic_of_an_AI_image_pipeline__arrow.webp`
- `outputs/bold_hero_banner_showing_a_glowing_WebP_badge_hovering_above.webp`

### Troubleshooting
- "model is only available..." -> set `FAL_KEY` or `FAL_API_KEY` and retry.
- "output_dir ... does not exist" -> create the folder first.
- If a client can't find the tool schema, call the tool with `"help"` to refresh it.
