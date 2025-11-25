## Tool: image_generate
Arguments:
- prompt (required): what to draw.
- output_dir (required): absolute or relative path where the WebP file will be written; the directory must already exist and be writable.
- reference_images (optional): list of 1–14 local file paths or http(s) URLs to images (PNG/JPEG/WebP). Every attachment is validated for existence/decodability and rejected with a clear error if invalid. In your prompt, describe each reference image and how to use it without naming files (e.g., “restyle the dog in the style of the watercolor” instead of citing filenames).
- aspect_ratio (optional): one of 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, 9:21.
- image_size (optional, default 2K): 1K | 2K | 4K.
- seed (optional): reserved; not currently supported by the public preview API.

Behavior:
- Uses Gemini 3 Pro Image Preview via generateContent; model default safety/person settings.
- Prompt enhancement off; watermarking off.
- Saves images as lossless WebP to the specified output_dir using sanitized prompt stems (spaces→underscores, trimmed, deduped with -001, -002) and returns paths/mime types. Image decode failures now raise an error instead of writing invalid files. Reference images are fetched/validated (max 20MB each) before calling the model; invalid attachments fail fast with a clear message.
