## Tool: image_generate

This MCP has an unusual interface, so it only uses up a few tokens of context. When the LLM wants to make an image, it calls it with "help" and gets the full json schema. 

It uses Z-image (fast, good, cheap) or nano banana pro (better) to generate images.

Perfect for having your friendly coding agent design a full website, game, or multibillion-dollar SaaS product!



### What you get when you tell it "help"
- `prompt` (required): what to draw.
- `output_dir` (required): existing directory where the WebP will be written; must already exist and be writable.
- `reference_images` (optional): list of 1–14 local paths or http(s) URLs to PNG/JPEG/WebP. Every attachment is fetched/validated (≤20MB, decodable). In the prompt, describe each reference image and how to use it **without naming files** (e.g., “restyle the dog in the style of the watercolor”).
- `aspect_ratio` (optional): one of 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, 9:21.
- `image_size` (optional, default 1K): 1K | 2K | 4K.
- `model` (optional, requires FAL_KEY/FAL_API_KEY): "z-image turbo (fast)" | "nano banana pro (best)". Hidden from help unless a FAL key is set. FAL calls are forced to `output_format=webp` and saved as-is when already WebP.

### Behavior
- Outputs **lossless WebP** only; filenames are sanitized from the prompt (spaces→`_`, max ~60 chars, collisions get `-001`, `-002`, …).
- Fails fast if `output_dir` is missing/not a directory or if any attachment is invalid.
