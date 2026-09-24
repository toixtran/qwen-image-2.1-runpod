"""Generate one image per line of a prompt file and save it as output/<id>.png.

Prompt file format, one job per line (blank lines and lines starting with # are skipped):

    1|A photorealistic strawberry with a human face, highly detailed
    2|A neon shop sign that reads "QWEN", rainy night

Usage:

    python examples/batch_prompts.py prompts.txt
    python examples/batch_prompts.py prompts.txt --resolution 2k --steps 40 --seed 42 --out output

The API key is read from API_KEY below, then the RUNPOD_API_KEY env var, otherwise asked at startup.
Images that already exist in the output folder are skipped, so an interrupted run can be resumed.
"""

import argparse
import base64
import getpass
import json
import os
import sys
import time
import urllib.error
import urllib.request

API_KEY = ""  # Paste your RunPod API key here, or leave empty to use RUNPOD_API_KEY / the prompt.
ENDPOINT_ID = "rdkyjvit5wpxdn"
API_BASE = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.ai/v2")
POLL_INTERVAL = 5
JOB_TIMEOUT = 900  # seconds, per prompt, including cold start


def parse_prompts(path):
    jobs, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" not in line:
                sys.exit(f"{path}:{lineno}: expected 'id|prompt', got: {line!r}")
            job_id, prompt = (part.strip() for part in line.split("|", 1))
            if not job_id or not prompt:
                sys.exit(f"{path}:{lineno}: id and prompt must both be non-empty")
            if job_id in seen:
                sys.exit(f"{path}:{lineno}: duplicate id {job_id!r}")
            seen.add(job_id)
            jobs.append((job_id, prompt))
    return jobs


def request(method, url, api_key, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e


def run_job(api, api_key, job_input):
    """Submit via /runsync and poll /status if the job outlives the runsync wait window."""
    started = time.time()
    resp = request("POST", f"{api}/runsync", api_key, {"input": job_input})
    while resp.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        if time.time() - started > JOB_TIMEOUT:
            request("POST", f"{api}/cancel/{resp['id']}", api_key)
            raise RuntimeError(f"timed out after {JOB_TIMEOUT}s (job {resp['id']} cancelled)")
        time.sleep(POLL_INTERVAL)
        resp = request("GET", f"{api}/status/{resp['id']}", api_key)
    return resp, time.time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt_file")
    parser.add_argument("--out", default="output", help="output folder (default: output)")
    parser.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID", ENDPOINT_ID))
    parser.add_argument("--resolution", default="1k", choices=["1k", "2k"])
    parser.add_argument("--aspect-ratio", help="1:1, 4:3, 3:4, 3:2, 2:3, 16:9 or 9:16")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="-1 for a random seed per image")
    parser.add_argument("--format", default="png", choices=["png", "jpeg", "webp"])
    parser.add_argument("--overwrite", action="store_true", help="regenerate images that already exist")
    args = parser.parse_args()

    jobs = parse_prompts(args.prompt_file)
    if not jobs:
        sys.exit(f"No prompts found in {args.prompt_file}")

    api_key = API_KEY or os.environ.get("RUNPOD_API_KEY") or getpass.getpass("RunPod API key: ").strip()
    if not api_key:
        sys.exit("An API key is required.")

    api = f"{API_BASE}/{args.endpoint}"
    ext = "jpg" if args.format == "jpeg" else args.format
    os.makedirs(args.out, exist_ok=True)

    done, skipped, failed, timings = 0, 0, [], []
    batch_started = time.time()
    for n, (job_id, prompt) in enumerate(jobs, 1):
        path = os.path.join(args.out, f"{job_id}.{ext}")
        label = f"[{n}/{len(jobs)}] {job_id}"
        if os.path.exists(path) and not args.overwrite:
            print(f"{label}: exists, skipped")
            skipped += 1
            continue

        job_input = {
            "prompt": prompt,
            "resolution": args.resolution,
            "num_inference_steps": args.steps,
            "seed": args.seed,
            "output_format": args.format,
        }
        if args.aspect_ratio:
            job_input["aspect_ratio"] = args.aspect_ratio

        print(f"{label}: {prompt[:70]}{'…' if len(prompt) > 70 else ''}")
        try:
            resp, elapsed = run_job(api, api_key, job_input)
            output = resp.get("output") or {}
            if resp.get("status") != "COMPLETED" or "error" in output:
                raise RuntimeError(output.get("error") or resp.get("error") or resp.get("status"))
            image = output["images"][0]
            with open(path, "wb") as f:
                f.write(base64.b64decode(image["image"]))
            # delayTime = queue + cold start, executionTime = time inside the handler (both in ms).
            delay = (resp.get("delayTime") or 0) / 1000
            execution = (resp.get("executionTime") or 0) / 1000
            inference = output.get("inference_time") or 0
            timings.append((job_id, delay, execution, inference, elapsed))
            print(f"  saved {path} ({image['width']}x{image['height']}, seed={output.get('seed')})")
            print(
                f"  time: inference {inference:.1f}s | execution {execution:.1f}s | "
                f"queue+cold start {delay:.1f}s | total {elapsed:.1f}s"
            )
            done += 1
        except KeyboardInterrupt:
            print("\nInterrupted. Re-run the same command to continue where it stopped.")
            break
        except Exception as e:  # keep going with the remaining prompts
            print(f"  FAILED: {e}")
            failed.append(job_id)

    if timings:
        print(f"\n{'id':<10}{'inference':>11}{'execution':>11}{'queue+cold':>12}{'total':>9}")
        for job_id, delay, execution, inference, elapsed in timings:
            print(f"{job_id:<10}{inference:>10.1f}s{execution:>10.1f}s{delay:>11.1f}s{elapsed:>8.1f}s")
        count = len(timings)
        print(
            f"{'average':<10}{sum(t[3] for t in timings) / count:>10.1f}s"
            f"{sum(t[2] for t in timings) / count:>10.1f}s"
            f"{sum(t[1] for t in timings) / count:>11.1f}s"
            f"{sum(t[4] for t in timings) / count:>8.1f}s"
        )

    wall = time.time() - batch_started
    print(f"\nDone: {done} generated, {skipped} skipped, {len(failed)} failed in {wall // 60:.0f}m {wall % 60:.0f}s")
    if failed:
        print("Failed ids: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
