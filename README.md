## Tool: image_generate (single-string interface)

- Call the tool with a single string. If the string does **not** contain `{}` (e.g., just `"help"`), the tool replies with usage instructions and the JSON schema.
- To generate an image, include a JSON object in the string; anything outside the outermost `{ ... }` is ignored. Example (Codex CLI):  
  ```bash
  codex exec --skip-git-repo-check "Call mcp_generate_image.image_generate with '{\"prompt\":\"line art coffee cup\",\"output_dir\":\"outputs\"}'"
  ```

### Request fields (in the JSON object)
- `prompt` (required): what to draw.
- `output_dir` (required): existing directory where the WebP will be written; must already exist and be writable.
- `reference_images` (optional): list of 1–14 local paths or http(s) URLs to PNG/JPEG/WebP. Every attachment is fetched/validated (≤20MB, decodable). In the prompt, describe each reference image and how to use it **without naming files** (e.g., “restyle the dog in the style of the watercolor”).
- `aspect_ratio` (optional): one of 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, 9:21.
- `image_size` (optional, default 2K): 1K | 2K | 4K.

### Behavior
- Uses Gemini 3 Pro Image Preview via `generate_content`.
- Outputs **lossless WebP** only; filenames are sanitized from the prompt (spaces→`_`, max ~60 chars, collisions get `-001`, `-002`, …).
- Fails fast if `output_dir` is missing/not a directory or if any attachment is invalid.
