"""Call a deployed endpoint and save the returned images.

    export RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...
    python examples/client.py "A red fox in the snow" --aspect-ratio 16:9
    python examples/client.py "Change the background to a sunset beach" --image input.png
    python examples/client.py "A cute cartoon dragon sticker." --transparent
"""

import argparse
import base64
import os
import sys
import time

import requests

API = "https://api.runpod.ai/v2"


def encode_file(path):
    if path.startswith(("http://", "https://")):
        return path
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--image", action="append", default=[], help="Reference image path or URL (repeatable, max 10)")
    parser.add_argument("--aspect-ratio")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--format", default="png", choices=["png", "jpeg", "webp"])
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    api_key = os.environ["RUNPOD_API_KEY"]
    endpoint = os.environ["RUNPOD_ENDPOINT_ID"]
    headers = {"Authorization": f"Bearer {api_key}"}

    payload = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "num_images": args.num_images,
        "transparent": args.transparent,
        "output_format": args.format,
    }
    for key, value in (("aspect_ratio", args.aspect_ratio), ("width", args.width),
                       ("height", args.height), ("seed", args.seed)):
        if value is not None:
            payload[key] = value
    if args.image:
        payload["images"] = [encode_file(p) for p in args.image]

    # Async /run + polling: generation at 2K can exceed the /runsync wait window.
    job = requests.post(f"{API}/{endpoint}/run", json={"input": payload}, headers=headers, timeout=60).json()
    job_id = job["id"]
    print(f"job {job_id} submitted")
    while True:
        status = requests.get(f"{API}/{endpoint}/status/{job_id}", headers=headers, timeout=60).json()
        if status["status"] in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        time.sleep(3)

    output = status.get("output") or {}
    if status["status"] != "COMPLETED" or "error" in output:
        sys.exit(f"{status['status']}: {output.get('error') or status.get('error')}")

    os.makedirs(args.out, exist_ok=True)
    for i, item in enumerate(output["images"]):
        path = os.path.join(args.out, f"{job_id}_{i}.{item['format']}")
        with open(path, "wb") as f:
            f.write(base64.b64decode(item["image"]))
        print(f"saved {path} ({item['width']}x{item['height']}, seed={output['seed']}, {output['inference_time']}s)")


if __name__ == "__main__":
    main()
