"""RunPod serverless worker for Qwen/Qwen-Image-2.1.

Supports text-to-image, image editing (1-10 reference images) and transparent
(RGBA) generation with a single pipeline.
"""

import base64
import binascii
import os
import random
import time
from io import BytesIO

MODEL_ID = os.environ.get("HF_MODEL", "Qwen/Qwen-Image-2.1")

# Filled by RunPod when the endpoint's "Model" field is set: no download, and no billing for download time.
RUNPOD_MODEL_CACHE = "/runpod-volume/huggingface-cache/hub"


def resolve_cached_snapshot(model_id):
    """Return the local snapshot dir of a RunPod-cached model, or None."""
    model_root = os.path.join(RUNPOD_MODEL_CACHE, "models--" + model_id.replace("/", "--"))
    snapshots = os.path.join(model_root, "snapshots")
    ref = os.path.join(model_root, "refs", "main")
    if os.path.isfile(ref):
        with open(ref) as f:
            candidate = os.path.join(snapshots, f.read().strip())
        if os.path.isdir(candidate):
            return candidate
    if os.path.isdir(snapshots):
        versions = sorted(os.listdir(snapshots))
        if versions:
            return os.path.join(snapshots, versions[0])
    return None


# Resolve where weights come from before importing diffusers, since huggingface_hub reads these env vars on import.
# Order: RunPod cached model > explicit HF_HOME > weights baked into the image > network volume > library default.
MODEL_SOURCE = resolve_cached_snapshot(MODEL_ID)
if MODEL_SOURCE:
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    MODEL_SOURCE = MODEL_ID
    if "HF_HOME" not in os.environ:
        if os.path.isdir("/models/huggingface"):
            os.environ["HF_HOME"] = "/models/huggingface"
        elif os.path.isdir("/runpod-volume"):
            os.environ["HF_HOME"] = "/runpod-volume/huggingface"

import requests
import runpod
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

PRECISION = os.environ.get("PRECISION", "bf16").lower()
# On-the-fly quantization of the official BF16 weights: none | int8 | nf4 (4-bit).
QUANTIZATION = os.environ.get("QUANTIZATION", "none").strip().lower()
ENABLE_CPU_OFFLOAD = os.environ.get("ENABLE_CPU_OFFLOAD", "false").lower() in ("1", "true", "yes")
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(2752 * 1536)))
MAX_IMAGES_PER_JOB = int(os.environ.get("MAX_IMAGES_PER_JOB", "1"))
DEFAULT_RESOLUTION = os.environ.get("DEFAULT_RESOLUTION", "1k").lower()
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "40"))
# Optional diffusers attention backend for the cached decode steps, e.g. "_native_cudnn" or "flash_hub".
ATTENTION_BACKEND = os.environ.get("ATTENTION_BACKEND", "").strip()
MAX_REFERENCE_IMAGES = 10
DOWNLOAD_TIMEOUT = 30

# Recommended sizes from the model card.
ASPECT_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

# ~1 MP variants (half of each side, floored to a multiple of 32): ~4x faster than 2K.
ASPECT_RATIOS_1K = {
    "1:1": (1024, 1024),
    "4:3": (1184, 896),
    "3:4": (896, 1184),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
    "16:9": (1376, 768),
    "9:16": (768, 1376),
}

RESOLUTIONS = {"2k": ASPECT_RATIOS, "1k": ASPECT_RATIOS_1K}

RGBA_PREFIX ="This is an RGBA image with transparency."
RGBA_SUFFIX = "The image has alpha channel and the background is transparent."

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def build_quantization_config():
    """Quantize the transformer and text encoder (the VAE is small and stays in full precision)."""
    if QUANTIZATION == "none":
        return None
    if QUANTIZATION not in ("int8", "nf4"):
        raise ValueError(f"QUANTIZATION must be none, int8 or nf4, got {QUANTIZATION!r}")

    from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
    from diffusers.quantizers import PipelineQuantizationConfig
    from transformers import BitsAndBytesConfig as TransformersBnbConfig

    if QUANTIZATION == "int8":
        kwargs = {"load_in_8bit": True}
    else:
        kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": DTYPES[PRECISION],
        }
    # One config per library: passing quant_backend instead makes diffusers require identical
    # BitsAndBytesConfig signatures in diffusers and transformers, which breaks across versions.
    return PipelineQuantizationConfig(
        quant_mapping={
            "transformer": DiffusersBnbConfig(**kwargs),
            "text_encoder": TransformersBnbConfig(**kwargs),
        }
    )


def load_pipeline():
    if PRECISION not in DTYPES:
        raise ValueError(f"PRECISION must be one of {list(DTYPES)}, got {PRECISION!r}")

    started = time.time()
    load_kwargs = {"torch_dtype": DTYPES[PRECISION]}
    quantization_config = build_quantization_config()
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    # bitsandbytes 8-bit weights cannot be moved between CPU and GPU, so quantized models stay on the GPU.
    offload = ENABLE_CPU_OFFLOAD and QUANTIZATION != "int8"
    if ENABLE_CPU_OFFLOAD and not offload:
        print("[init] ENABLE_CPU_OFFLOAD is ignored with QUANTIZATION=int8")
    if offload:
        pipe = QwenImage21Pipeline.from_pretrained(MODEL_SOURCE, **load_kwargs)
        pipe.enable_model_cpu_offload()
    else:
        # Load weights straight onto the GPU instead of staging them in CPU RAM first.
        pipe = QwenImage21Pipeline.from_pretrained(MODEL_SOURCE, device_map="cuda", **load_kwargs)
    if ATTENTION_BACKEND:
        pipe.transformer.set_attention_backend(ATTENTION_BACKEND)
    pipe.set_progress_bar_config(disable=True)
    print(
        f"[init] loaded {MODEL_SOURCE} ({PRECISION}, quantization={QUANTIZATION}, offload={offload}, "
        f"attention={ATTENTION_BACKEND or 'default'}) in {time.time() - started:.1f}s"
    )
    return pipe


PIPE = load_pipeline()


class InputError(ValueError):
    pass


def load_image(source):
    """Load a PIL image from an http(s) URL, a data URI or a raw base64 string."""
    if not isinstance(source, str) or not source:
        raise InputError("Each image must be a non-empty string (URL or base64).")

    if source.startswith(("http://", "https://")):
        resp = requests.get(source, timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        data = resp.content
    else:
        if source.startswith("data:"):
            source = source.split(",", 1)[-1]
        try:
            data = base64.b64decode(source, validate=True)
        except (binascii.Error, ValueError) as e:
            raise InputError(f"Invalid base64 image: {e}") from e

    img = Image.open(BytesIO(data))
    img.load()
    # Keep alpha when present so transparent layers can be edited.
    return img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")


def collect_images(job_input):
    sources = []
    if job_input.get("images"):
        if not isinstance(job_input["images"], list):
            raise InputError("'images' must be a list of URLs or base64 strings.")
        sources.extend(job_input["images"])
    for key in ("image", "image_url", "image_base64"):
        if job_input.get(key):
            sources.append(job_input[key])

    if len(sources) > MAX_REFERENCE_IMAGES:
        raise InputError(f"At most {MAX_REFERENCE_IMAGES} reference images are supported, got {len(sources)}.")
    return [load_image(s) for s in sources]


def resolve_size(job_input, has_images):
    width, height = job_input.get("width"), job_input.get("height")
    aspect_ratio = job_input.get("aspect_ratio")
    resolution = str(job_input.get("resolution", DEFAULT_RESOLUTION)).lower()
    if resolution not in RESOLUTIONS:
        raise InputError(f"'resolution' must be one of {list(RESOLUTIONS)}, got {resolution!r}.")
    sizes = RESOLUTIONS[resolution]

    if aspect_ratio is not None:
        if aspect_ratio not in sizes:
            raise InputError(f"'aspect_ratio' must be one of {list(sizes)}, got {aspect_ratio!r}.")
        width, height = sizes[aspect_ratio]
    elif (width is None) != (height is None):
        raise InputError("Provide both 'width' and 'height', or neither.")
    elif width is None and not has_images:
        width, height = sizes["1:1"]

    # For editing without an explicit size, the pipeline derives it from the last reference image.
    if width is None:
        return None, None

    width, height = int(width), int(height)
    if width < 256 or height < 256:
        raise InputError("'width' and 'height' must be at least 256.")
    if width * height > MAX_PIXELS:
        raise InputError(f"width*height = {width * height} exceeds MAX_PIXELS = {MAX_PIXELS}.")
    return width, height


def build_prompt(prompt, transparent):
    if transparent and RGBA_PREFIX.lower() not in prompt.lower():
        prompt = f"{RGBA_PREFIX} {prompt.strip()} {RGBA_SUFFIX}"
    return prompt


def normalize_format(fmt):
    fmt = str(fmt).lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in ("png", "jpeg", "webp"):
        raise InputError(f"'output_format' must be png, jpeg or webp, got {fmt!r}.")
    return fmt


def encode_image(img, fmt, quality):
    if fmt == "jpeg" and img.mode == "RGBA":
        # JPEG has no alpha: composite over white.
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.getchannel("A"))
        img = background

    buf = BytesIO()
    save_kwargs = {"quality": quality} if fmt in ("jpeg", "webp") else {}
    img.save(buf, format=fmt.upper(), **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(job):
    job_input = job.get("input") or {}
    try:
        prompt = job_input.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise InputError("'prompt' is required and must be a string.")

        images = collect_images(job_input)
        width, height = resolve_size(job_input, has_images=bool(images))

        num_images = int(job_input.get("num_images", 1))
        if not 1 <= num_images <= MAX_IMAGES_PER_JOB:
            raise InputError(f"'num_images' must be between 1 and {MAX_IMAGES_PER_JOB}.")

        steps = int(job_input.get("num_inference_steps", DEFAULT_STEPS))
        if not 1 <= steps <= 100:
            raise InputError("'num_inference_steps' must be between 1 and 100.")

        seed = job_input.get("seed")
        seed = random.randint(0, 2**32 - 1) if seed is None or int(seed) < 0 else int(seed)

        output_format = normalize_format(job_input.get("output_format", "png"))
        quality = int(job_input.get("quality", 95))
        final_prompt = build_prompt(prompt, bool(job_input.get("transparent", False)))

        kwargs = {
            "prompt": final_prompt,
            "num_inference_steps": steps,
            "num_images_per_prompt": num_images,
            "true_cfg_scale": float(job_input.get("true_cfg_scale", 1.0)),
            "output_resolution": int(job_input.get("output_resolution", 1024)),
            "use_kv_cache": bool(job_input.get("use_kv_cache", True)),
            "generator": torch.Generator("cuda").manual_seed(seed),
        }
        if job_input.get("negative_prompt"):
            kwargs["negative_prompt"] = job_input["negative_prompt"]
        if images:
            kwargs["image"] = images if len(images) > 1 else images[0]
        if width is not None:
            kwargs["width"], kwargs["height"] = width, height

        started = time.time()
        with torch.inference_mode():
            result = PIPE(**kwargs).images
        elapsed = time.time() - started

        outputs = []
        for img in result:
            b64 = encode_image(img, output_format, quality)
            outputs.append({"image": b64, "format": output_format, "width": img.width, "height": img.height, "mode": img.mode})

        return {
            "images": outputs,
            "seed": seed,
            "prompt": final_prompt,
            "task": "edit" if images else "text-to-image",
            "inference_time": round(elapsed, 2),
        }
    except InputError as e:
        return {"error": str(e)}
    except requests.RequestException as e:
        return {"error": f"Failed to download image: {e}"}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "CUDA out of memory. Lower the resolution/num_images or set ENABLE_CPU_OFFLOAD=true."}


runpod.serverless.start({"handler": handler})
