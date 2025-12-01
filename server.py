"""
MCP server that exposes a single tool to call the Gemini image generation API
(Gemini 3 Pro Image Preview).

- Transport: stdio (run with `python server.py`)
- Auth: uses GEMINI_API_KEY / GOOGLE_API_KEY if present; otherwise falls back
  to Application Default Credentials when available.
"""

from __future__ import annotations

import base64
import asyncio
import json
import mimetypes
import os
import sys
import time
import urllib.request
from urllib.error import HTTPError
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import google.genai as genai
from google.genai import types
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from PIL import Image

BASE_DIR = Path(__file__).parent
MIN_GENAI_VERSION = (1, 52, 0)
MAX_REFERENCE_IMAGES = 14
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20MB hard cap to match Gemini guidance
ALLOWED_REF_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "JPG": "image/jpeg", "WEBP": "image/webp"}
FAL_ENV_VARS = ("FAL_KEY", "FAL_API_KEY")
MAX_FAL_IMAGE_BYTES = 40 * 1024 * 1024  # allow larger model outputs


def log_event(level: str, message: str, **fields: object) -> None:
    """Emit a plain-English log line to stderr (no key/values)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%fZ")
    extras = ""
    if fields:
        extras = " ".join(str(v) for v in fields.values())
    line = f"[{ts}] {level.upper()} - {message} {extras}".strip()
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _get_fal_api_key() -> Optional[str]:
    """Return the first configured FAL API key, if any."""
    return next((os.getenv(env) for env in FAL_ENV_VARS if os.getenv(env)), None)


def resolve_client() -> genai.Client:
    """Create a Gemini client using the best available credentials."""
    key_candidates = (
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("GENAI_API_KEY"),
    )
    api_key = next((k for k in key_candidates if k), None)

    if api_key:
        log_event("debug", "Using API key auth for Gemini", source="env")
        return genai.Client(api_key=api_key)

    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT") or os.getenv("PROJECT_ID")
    location = os.getenv("GOOGLE_CLOUD_REGION") or "us-central1"
    if project:
        log_event("debug", "Using Vertex-style auth for Gemini", project=project, location=location)
        return genai.Client(vertexai={"project": project, "location": location})

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path:
        log_event("debug", "Attempting ADC auth for Gemini", credentials_file=Path(cred_path).name)
        return genai.Client()  # ADC picked up by default SDK logic

    raise RuntimeError(
        "No Gemini credentials found. Set GEMINI_API_KEY/GOOGLE_API_KEY or Vertex env (GOOGLE_CLOUD_PROJECT & GOOGLE_CLOUD_REGION)."
    )


def _infer_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "image/png"


def _ensure_image_bytes(data: bytes, label: str) -> tuple[bytes, str]:
    """Validate that bytes are a decodable image; may normalize format."""
    if len(data) < 1024:
        raise ValueError(f"Reference image '{label}' is too small to be a valid image.")

    try:
        with Image.open(BytesIO(data)) as im:
            fmt = (im.format or "").upper()
            if fmt in ALLOWED_REF_FORMATS:
                mime = ALLOWED_REF_FORMATS[fmt]
                return data, mime

            # Normalize unsupported formats to PNG.
            im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            buf = BytesIO()
            im.save(buf, format="PNG")
            normalized = buf.getvalue()
            return normalized, "image/png"
    except Exception as e:
        raise ValueError(f"Reference image '{label}' is not a valid decodable image: {e}") from e


def _load_reference_image(source: str) -> tuple[bytes, str]:
    """
    Load a reference image from a local path or HTTP(S) URL.

    Returns (bytes, mime_type). Raises on any validation failure.
    """
    source = source.strip()
    if not source:
        raise ValueError("reference_images entries cannot be empty.")

    is_url = source.startswith(("http://", "https://"))

    if is_url:
        parsed = urlparse(source)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"reference image URL '{source}' must be a valid http(s) URL.")

        req = urllib.request.Request(
            source,
            headers={"User-Agent": "mcp-generate-image/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status >= 400:
                    raise ValueError(f"reference image URL '{source}' returned HTTP {resp.status}.")
                content_length = resp.getheader("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > MAX_IMAGE_BYTES:
                            raise ValueError(f"reference image URL '{source}' exceeds 20MB limit (Content-Length).")
                    except ValueError:
                        pass  # non-integer, fall back to actual read size
                data = resp.read(MAX_IMAGE_BYTES + 1)
        except Exception as e:
            raise ValueError(f"Failed to fetch reference image URL '{source}': {e}") from e

        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"reference image URL '{source}' exceeds 20MB limit.")
        data, mime = _ensure_image_bytes(data, source)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"reference image URL '{source}' exceeds 20MB limit after validation.")
        return data, mime

    # Local file path
    path = Path(source).expanduser()
    if not path.exists():
        raise ValueError(f"reference image path '{source}' does not exist.")
    if not path.is_file():
        raise ValueError(f"reference image path '{source}' is not a file.")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"reference image path '{source}' is empty.")
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"reference image path '{source}' exceeds 20MB limit.")

    data = path.read_bytes()
    data, mime = _ensure_image_bytes(data, source)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"reference image path '{source}' exceeds 20MB limit after validation.")
    return data, mime


def _save_image_bytes(data: bytes, mime_type: str, stem: str, output_dir: Path) -> str:
    ext = ".webp"  # enforce webp output
    target = output_dir / f"{stem}{ext}"
    counter = 1
    base = stem
    while target.exists():
        target = output_dir / f"{base}-{counter:03d}{ext}"
        counter += 1
    target.write_bytes(data)
    return str(target)


def _to_webp_lossless(data: bytes) -> tuple[bytes, str]:
    """
    Convert raw image bytes to lossless WebP.

    Returns (bytes, mime_type). Raises if input is not a decodable image.
    """
    try:
        with Image.open(BytesIO(data)) as im:
            # Preserve alpha if present; otherwise use RGB.
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            buf = BytesIO()
            im.save(buf, format="WEBP", lossless=True, quality=100, method=6)
            return buf.getvalue(), "image/webp"
    except Exception as e:
        raise ValueError(f"inline image bytes are not a valid decodable image: {e}") from e


def _decode_inline_image(payload: object) -> bytes:
    """Decode inline image payload to raw bytes, validating size."""
    data: bytes
    if isinstance(payload, str):
        data = base64.b64decode(payload)
    elif isinstance(payload, bytes):
        # If already looks like an image header, keep as-is; otherwise try base64.
        if payload[:4] in (b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1", b"\x89PNG", b"RIFF"):
            data = payload
        else:
            try:
                data = base64.b64decode(payload, validate=True)
            except Exception:
                data = payload
    else:
        raise TypeError("inline image payload must be str or bytes")

    if len(data) < 1024:
        raise ValueError("Inline image payload too small to be a valid image.")
    return data


def _aspect_ratio_tuple(ratio: str | None) -> tuple[int, int] | None:
    if not ratio:
        return None
    try:
        w_str, h_str = ratio.split(":")
        w, h = int(w_str), int(h_str)
        if w <= 0 or h <= 0:
            return None
        return w, h
    except Exception:
        return None


def _fal_dims_for_size(image_size: str, aspect_ratio: str | None) -> tuple[int, int]:
    """
    Choose width/height for FAL z-image turbo.

    We target the longer side to 1K/2K/4K pixels and derive the short side from the aspect ratio.
    """
    long_side = {"1K": 1024, "2K": 2048, "4K": 4096}.get(image_size, 1024)
    ratio = _aspect_ratio_tuple(aspect_ratio) or (1, 1)
    w_ratio, h_ratio = ratio
    if w_ratio >= h_ratio:
        width = long_side
        height = max(1, int(round(long_side * h_ratio / w_ratio)))
    else:
        height = long_side
        width = max(1, int(round(long_side * w_ratio / h_ratio)))
    return width, height


def _download_image(url: str, max_bytes: int = MAX_FAL_IMAGE_BYTES) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "mcp-generate-image/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"download failed with HTTP {resp.status}")
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise RuntimeError("download exceeded max size")
        mime = resp.getheader("Content-Type") or mimetypes.guess_type(url)[0] or "application/octet-stream"
        mime = mime.split(";")[0].strip().lower()
        return data, mime


def _fal_request_sync(
    req: "ImageRequest",
    prompt: str,
    safe_prompt: str,
    out_dir: Path,
    aspect_ratio: str | None,
    fal_api_key: str,
) -> dict:
    model_choice = req.model or "z-image turbo (fast)"
    if model_choice == "z-image turbo (fast)":
        endpoint = "https://fal.run/fal-ai/z-image/turbo"
        width, height = _fal_dims_for_size(req.image_size, aspect_ratio)
        input_payload: dict[str, object] = {
            "prompt": prompt,
            "output_format": "webp",
            "image_size": {"width": width, "height": height},
        }
        model_id = "fal-ai/z-image/turbo"
    else:
        endpoint = "https://fal.run/fal-ai/nano-banana-pro"
        input_payload = {
            "prompt": prompt,
            "output_format": "webp",
            "resolution": req.image_size,
        }
        if aspect_ratio:
            input_payload["aspect_ratio"] = aspect_ratio
        model_id = "fal-ai/nano-banana-pro"

    # FAL REST endpoints expect the fields at the top level, not nested under "input".
    payload = input_payload

    log_event(
        "info",
        "fal.request",
        model=model_id,
        image_size=req.image_size,
        aspect_ratio=aspect_ratio,
    )

    http_req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Key {fal_api_key}",
            "User-Agent": "mcp-generate-image/1.0",
        },
        method="POST",
    )

    start = time.monotonic()
    try:
        with urllib.request.urlopen(http_req, timeout=60) as resp:
            body = resp.read()
            status = resp.status
    except HTTPError as e:
        err_body = e.read()
        snippet = (
            err_body[:400].decode("utf-8", errors="replace") if isinstance(err_body, (bytes, bytearray)) else str(err_body)
        )
        raise RuntimeError(f"FAL API returned HTTP {e.code}: {snippet}") from None
    if status >= 400:
        snippet = body[:400].decode("utf-8", errors="replace") if isinstance(body, (bytes, bytearray)) else str(body)
        raise RuntimeError(f"FAL API returned HTTP {status}: {snippet}")

    try:
        result = json.loads(body)
    except Exception as e:
        raise RuntimeError(f"Failed to parse FAL response: {e}") from e

    images = result.get("images") or []
    if not images:
        raise RuntimeError("FAL response contained no images.")

    results: list[dict] = []
    for idx, img in enumerate(images):
        url = img.get("url") or img.get("url_public") or img.get("signed_url")
        if not url:
            continue
        data, mime = _download_image(url)
        if mime.lower().startswith("image/webp"):
            webp_bytes, mime_type = data, "image/webp"
        else:
            webp_bytes, mime_type = _to_webp_lossless(data)
        path = _save_image_bytes(webp_bytes, mime_type, safe_prompt, out_dir)
        results.append(
            {
                "path": path,
                "mime_type": mime_type,
                "enhanced_prompt": None,
                "safety": None,
            }
        )

    if not results:
        raise RuntimeError("FAL response did not yield downloadable images.")

    elapsed_ms = int((time.monotonic() - start) * 1000)
    log_event("info", "fal.response", images=len(results), elapsed_ms=elapsed_ms, model=model_id)
    return {"images": results}


async def _generate_with_fal(
    req: "ImageRequest",
    prompt: str,
    safe_prompt: str,
    out_dir: Path,
    aspect_ratio: str | None,
    fal_api_key: str,
) -> dict:
    return await asyncio.to_thread(
        _fal_request_sync,
        req,
        prompt,
        safe_prompt,
        out_dir,
        aspect_ratio,
        fal_api_key,
    )


def _aspect_ratio_tuple(ratio: str | None) -> tuple[int, int] | None:
    if not ratio:
        return None
    try:
        w_str, h_str = ratio.split(":")
        w, h = int(w_str), int(h_str)
        if w <= 0 or h <= 0:
            return None
        return w, h
    except Exception:
        return None


def _fal_dims_for_size(image_size: str, aspect_ratio: str | None) -> tuple[int, int]:
    """
    Choose width/height for FAL z-image turbo.

    We target the longer side to 1K/2K/4K pixels and derive the short side from the aspect ratio.
    """
    long_side = {"1K": 1024, "2K": 2048, "4K": 4096}.get(image_size, 1024)
    ratio = _aspect_ratio_tuple(aspect_ratio) or (1, 1)
    w_ratio, h_ratio = ratio
    if w_ratio >= h_ratio:
        width = long_side
        height = max(1, int(round(long_side * h_ratio / w_ratio)))
    else:
        height = long_side
        width = max(1, int(round(long_side * w_ratio / h_ratio)))
    return width, height


def _download_image(url: str, max_bytes: int = MAX_FAL_IMAGE_BYTES) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "mcp-generate-image/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"download failed with HTTP {resp.status}")
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise RuntimeError("download exceeded max size")
        mime = resp.getheader("Content-Type") or mimetypes.guess_type(url)[0] or "application/octet-stream"
        return data, mime.lower()


def _parse_version(version_str: str | None) -> Optional[tuple[int, int, int]]:
    """Best-effort semantic version parsing; returns None if unknown."""
    if not version_str:
        return None
    parts = version_str.split(".")
    try:
        major, minor, patch = (int(parts[i]) if i < len(parts) else 0 for i in range(3))
        return (major, minor, patch)
    except Exception:
        return None


server = FastMCP(
    name="mcp_generate_image",
    instructions="BEFORE YOU USE THIS, call with the string 'help'.",
)

class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Use long descriptive, narrative framing. Provide technical details, for example specify camera angles, lens types, and lighting for photorealism; art style and background color for assets; font, text content, and layout for typography. Execute edits by ingesting reference images and directing semantic changes: masking for inpainting, merging inputs for composites, or applying style transfers—while explicitly commanding strict preservation of critical details. Structure prompts with hyper-specificity, declared intent, step-by-step logic for complex scenes, and use positive phrasing ('empty street' not 'no cars'). The picture can include lots of text.")
    output_dir: str = Field(
        ...,
        min_length=1,
        description="Directory where lossless WebP files will be saved. Directory must already exist.",
    )
    reference_images: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list (recommend 0-3, max 14) of local file paths or http(s) URLs used for image reference. In the prompt, explain how to use these images, but refer to them by their relevant content not their filenames (e.g., 'restyle the image of the dog in the style of the watercolor')."
        ),
    )
    aspect_ratio: Optional[
        Literal["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9", "9:21"]
    ] = Field(
        default=None,
        description="Optional aspect ratio.",
    )
    image_size: Literal["1K", "2K", "4K"] = Field(
        default="1K",
        description="Render size preset; bigger than 1K and most LLMs will crash on reading.",
    )
    model: Optional[Literal["z-image turbo (fast)", "nano banana pro (best)"]] = Field(
        default=None,
        description="The FAL model is fast and cheap; nano banana pro is better.",
    )


def _help_payload() -> dict:
    instructions = (
        "image_generate takes a single string argument. Call with JSON wrapped in braces to generate an image. "
    )
    schema = ImageRequest.model_json_schema()
    if not _get_fal_api_key():
        # Hide the FAL-only model selector when no FAL key is configured.
        props = dict(schema.get("properties", {}))
        props.pop("model", None)
        schema["properties"] = props
        required = schema.get("required", [])
        if "model" in required:
            schema["required"] = [r for r in required if r != "model"]
    return {"instructions": instructions, "schema": schema}

# Note - this is the definition provided to the caller.
@server.tool(
    name="image_generate",
    title="Generate image",
    description="Single-string entrypoint; call with 'help' to learn how to use it.",
)
async def generate_image(
    request: str,
    ctx: Optional[Context] = None,
) -> dict:
    """
    Generate an image with the configured Gemini image model.

    Single-string entrypoint: send 'help' (or any string without braces) to receive the schema and guidance.
    """

    if "{" not in request or "}" not in request or request.find("{") >= request.rfind("}"):
        return _help_payload()

    try:
        start = request.find("{")
        end = request.rfind("}") + 1
        payload = request[start:end]
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON payload: {e}"}

    fal_api_key = _get_fal_api_key()
    if not fal_api_key and isinstance(data, dict) and "model" in data:
        return {"error": "model is only available when FAL_KEY or FAL_API_KEY is configured in the environment."}

    try:
        req = ImageRequest.model_validate(data)
    except ValidationError as e:
        return {"error": e.errors()}

    prompt = req.prompt.strip()
    if not prompt:
        raise ValueError("prompt is empty or whitespace; please supply descriptive text to generate an image.")

    safe_prompt = (
        "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else " " for ch in prompt)
        .strip()
        .replace(" ", "_")
    )
    safe_prompt = safe_prompt[:60] if safe_prompt else ""
    if not safe_prompt:
        raise ValueError("prompt sanitization produced an empty filename stem; include at least one alphanumeric character.")

    out_dir = Path(req.output_dir).expanduser()
    if not out_dir.exists():
        raise ValueError(f"output_dir '{req.output_dir}' does not exist; please create it first.")
    if not out_dir.is_dir():
        raise ValueError(f"output_dir '{req.output_dir}' is not a directory.")

    allowed_ratios = {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9", "9:21"}
    aspect_ratio = req.aspect_ratio
    if aspect_ratio and aspect_ratio not in allowed_ratios:
        raise ValueError(f"aspect_ratio must be one of {sorted(allowed_ratios)}")

    image_parts: list[types.Part] = []
    reference_images = req.reference_images
    if reference_images:
        if len(reference_images) > MAX_REFERENCE_IMAGES:
            raise ValueError(f"reference_images supports up to {MAX_REFERENCE_IMAGES} items.")
        for src in reference_images:
            data, mime = _load_reference_image(src)
            image_parts.append(types.Part.from_bytes(data=data, mime_type=mime))

    # Route to FAL if a FAL model was requested.
    fal_api_key = _get_fal_api_key()
    if req.model and fal_api_key:
        return await _generate_with_fal(
            req=req,
            prompt=prompt,
            safe_prompt=safe_prompt,
            out_dir=out_dir,
            aspect_ratio=aspect_ratio,
            fal_api_key=fal_api_key,
        )

    client = resolve_client()

    sdk_version_str = getattr(genai, "__version__", None)
    sdk_version = _parse_version(sdk_version_str)
    if sdk_version and sdk_version < MIN_GENAI_VERSION:
        log_event(
            "warning",
            "google_genai_outdated",
            sdk_version=sdk_version_str,
            minimum=".".join(map(str, MIN_GENAI_VERSION)),
        )
        return {
            "error": (
                "Installed google-genai SDK is out of date; suggest upgrading to version 1.52.0 or newer."
            ),
            "sdk_version": sdk_version_str,
            "minimum_required": ".".join(map(str, MIN_GENAI_VERSION)),
        }

    cfg = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=req.image_size,
        ),
    )

    contents = [types.Part.from_text(text=prompt)]
    contents.extend(image_parts)

    start = time.monotonic()

    log_event(
        "info",
        "image_model.request",
        prompt=prompt[:240],
        aspect_ratio=aspect_ratio,
        image_size=req.image_size,
        model=req.model,
        reference_images=len(reference_images or []),
        safety_filter_level="BLOCK_NONE",
        person_generation="ALLOW_ALL",
    )

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=cfg,
        )
    except Exception as e:  # pragma: no cover
        elapsed_ms = int((time.monotonic() - start) * 1000)
        log_event(
            "error",
            "image_model.error",
            elapsed_ms=elapsed_ms,
            retry_after_ms=2000,
            error_type=type(e).__name__,
            error=str(e),
        )
        raise

    candidates = getattr(response, "candidates", None) or []
    images_b64: list[str] = []
    for cand in candidates:
        parts = getattr(cand, "content", cand).parts if hasattr(cand, "content") else getattr(cand, "parts", [])
        for part in parts or []:
            inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
            if inline and getattr(inline, "data", None):
                try:
                    images_b64.append(_decode_inline_image(inline.data))
                except Exception:
                    continue

    if not images_b64:
        raise RuntimeError("No images returned by the image model.")

    results: list[dict] = []

    for idx, data in enumerate(images_b64):
        data, mime = _to_webp_lossless(data)
        stem = safe_prompt
        path = _save_image_bytes(data, mime, stem, out_dir)
        results.append(
            {
                "path": path,
                "mime_type": mime,
                "enhanced_prompt": None,
                "safety": None,
            }
        )

    if ctx:
        # Context.info does not accept extra keyword args; emit plain text.
        await ctx.info(f"Saved {len(results)} image(s) to outputs/")

    log_event("info", "image_model.response", images=len(results))
    elapsed_ms = int((time.monotonic() - start) * 1000)
    log_event("info", "image_model.latency", elapsed_ms=elapsed_ms)
    return {"images": results}


if __name__ == "__main__":
    server.run(transport="stdio")

