"""Small local UI for Qwen Image 2.1 through stable-diffusion.cpp."""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
OUTPUTS = Path(os.environ.get("PLAYGROUND_OUTPUT_DIR", ROOT / "outputs"))
MODEL_DIR = Path(os.environ.get("PLAYGROUND_MODEL_DIR", "models"))
BACKEND = os.environ.get("PLAYGROUND_BACKEND", "sd-cpp")
PASSWORD = os.environ.get("PLAYGROUND_PASSWORD", "")
DEFAULTS = {
    "sd_cli": os.environ.get("PLAYGROUND_SD_CLI", "stable-diffusion.cpp/build/bin/sd-cli"),
    "diffusion": os.environ.get("PLAYGROUND_DIFFUSION", str(MODEL_DIR / "qwen_image_2.1-Q4_K.gguf")),
    "encoder": str(MODEL_DIR / "qwen3vl_8b_heretic-Q4_K_M.gguf"),
    "vision": str(MODEL_DIR / "mmproj-qwen3vl_8b_heretic-f16.gguf"),
    "vae": str(MODEL_DIR / "qwen_image_2.1_vae_bf16.safetensors"),
}
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
ACTIVE: str | None = None
MAX_BODY = 24 * 1024 * 1024


def path_for(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def assembled_prompt(data: dict) -> str:
    scene = str(data.get("scene", "")).strip()
    if not scene:
        raise ValueError("Describe the scene or edit first.")
    mode = str(data.get("input_mode", "style"))
    if mode not in {"style", "edit"}:
        raise ValueError("Image input mode must be style or edit.")
    has_image = bool(data.get("reference") or data.get("source_history_id"))
    if mode == "edit" and not has_image:
        raise ValueError("Choose an image to edit.")
    characters = data.get("characters", [])
    if not isinstance(characters, list) or len(characters) > 12:
        raise ValueError("Use at most 12 character cards.")
    parts = (["Edit the supplied image according to these instructions:", scene]
             if mode == "edit" else ["Create a new image of this scene:", scene])
    if characters:
        parts.extend(["", "Characters (keep their appearances distinct; use listed left-to-right order unless a position is specified):"])
        for index, character in enumerate(characters, 1):
            if not isinstance(character, dict):
                raise ValueError("Invalid character card.")
            description = str(character.get("description", "")).strip()
            if not description:
                continue
            name = str(character.get("name", "")).strip() or f"Character {index}"
            position = str(character.get("position", "auto")).strip()
            if position not in {"auto", "left", "center", "right", "foreground", "background"}:
                position = "auto"
            placement = f"; position: {position}" if position != "auto" else ""
            parts.append(f"{index}. {name}{placement}: {description}")
            avoid = str(character.get("avoid", "")).strip()
            if avoid:
                parts.append(f"   For {name}, avoid: {avoid}.")
    if mode == "edit":
        parts.extend(["", "Preserve the original image's elements and composition unless the instructions above change them.",
                      "Keep unchanged details recognizable; do not add unrelated elements."])
    elif has_image:
        parts.extend([
            "", "Use the supplied reference image only as a visual style reference.",
            "Use its linework, brushwork, shading, palette, texture, and rendering technique.",
            "Do not copy its subject, characters, poses, composition, objects, background, camera angle, or text.",
            "Follow the scene and character descriptions above for the content of the new image.",
        ])
    return "\n".join(parts)


def make_command(data: dict, reference: Path | None, output: Path) -> list[str]:
    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("Invalid model settings.")
    def integer(key: str, low: int, high: int, default: int) -> int:
        try:
            value = int(data.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number.") from None
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
        return value
    width = integer("width", 256, 2048, 768)
    height = integer("height", 256, 2048, 768)
    if width % 32 or height % 32:
        raise ValueError("Width and height must be divisible by 32.")
    steps = integer("steps", 1, 100, 30)
    try:
        cfg = float(data.get("cfg", 6))
    except (TypeError, ValueError):
        raise ValueError("CFG must be a number.") from None
    if not 0.1 <= cfg <= 20:
        raise ValueError("CFG must be between 0.1 and 20.")
    seed = integer("seed", 0, 2**32 - 1, 42)
    if BACKEND == "diffusers":
        if data.get("rewrite") and os.environ.get("PLAYGROUND_REWRITER", "0") != "1":
            raise ValueError("Prompt rewriting is disabled. Set PLAYGROUND_REWRITER=1 and restart the Pod.")
        if data.get("rewrite") and data.get("input_mode") == "edit":
            raise ValueError("Disable scene rewriting for image edits.")
        request = {"prompt": assembled_prompt(data), "reference": str(reference) if reference else None,
                   "output": str(output), "width": width, "height": height, "steps": steps,
                   "cfg": cfg, "seed": seed, "negative": str(data.get("negative", "")),
                   "offload": bool(data.get("offload", True)), "rewrite": bool(data.get("rewrite")),
                   "input_mode": data.get("input_mode", "style")}
        request_path = output.parent / "request.json"
        request_path.write_text(json.dumps(request, indent=2))
        return [sys.executable, "-u", str(ROOT / "inference_worker.py"), str(request_path)]
    if data.get("rewrite"):
        raise ValueError("Integrated BF16 rewriting is available on RunPod only.")
    resolved = {}
    for key in ("sd_cli", "diffusion", "encoder", "vae"):
        value = str(settings.get(key) or DEFAULTS[key]).strip()
        file = path_for(value)
        if not file.is_file():
            raise ValueError(f"{key} file was not found: {file}")
        resolved[key] = file
    vision_value = str(settings.get("vision") or DEFAULTS["vision"]).strip()
    if reference:
        if resolved["encoder"].suffix.lower() == ".gguf" and not vision_value:
            raise ValueError("A GGUF encoder needs a vision projector for reference images. Set the vision projector path in Model settings.")
        if vision_value:
            vision = path_for(vision_value)
            if not vision.is_file():
                raise ValueError(f"vision projector file was not found: {vision}")
            resolved["vision"] = vision
    cmd = [str(resolved["sd_cli"]), "--diffusion-model", str(resolved["diffusion"]),
           "--llm", str(resolved["encoder"]), "--vae", str(resolved["vae"])]
    if reference:
        if "vision" in resolved:
            cmd += ["--llm_vision", str(resolved["vision"])]
        cmd += ["-r", str(reference), "--ref-image-args",
                "pass_to_vlm=true,pass_to_dit=true,vlm_resize_mode=longest_side,vlm_max_size=512"]
    cmd += ["-p", assembled_prompt(data), "--seed", str(seed), "--cfg-scale", str(cfg), "--steps", str(steps),
            "--sampling-method", "euler", "--diffusion-fa", "-W", str(width),
            "-H", str(height), "-o", str(output)]
    if data.get("live_preview", True):
        interval = integer("preview_interval", 1, 20, 4)
        cmd += ["--preview", "vae", "--preview-interval", str(interval),
                "--preview-path", str(output.parent / "preview_%03d.png")]
    if data.get("vae_cpu", True):
        cmd += ["--backend", "vae=cpu"]
    negative = str(data.get("negative", "")).strip()
    if negative:
        cmd += ["-n", negative]
    if data.get("offload", False):
        cmd.append("--offload-to-cpu")
    return cmd


def decode_reference(value: str, directory: Path) -> Path | None:
    if not value:
        return None
    match = re.fullmatch(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)", value)
    if not match:
        raise ValueError("Reference must be a PNG, JPEG, or WebP image.")
    try:
        content = base64.b64decode(match.group(2), validate=True)
    except binascii.Error:
        raise ValueError("Invalid reference image.") from None
    if len(content) > 16 * 1024 * 1024:
        raise ValueError("Reference image exceeds 16 MB.")
    suffix = {"png": "png", "jpeg": "jpg", "webp": "webp"}[match.group(1)]
    path = directory / f"reference.{suffix}"
    path.write_bytes(content)
    return path


def resolve_reference(data: dict, directory: Path) -> Path | None:
    upload = str(data.get("reference") or "")
    history_id = str(data.get("source_history_id") or "")
    if upload and history_id:
        raise ValueError("Choose an uploaded image or a history image, not both.")
    if history_id:
        if not re.fullmatch(r"[a-f0-9]{32}", history_id):
            raise ValueError("Invalid history image ID.")
        source = OUTPUTS / history_id / "image.png"
        if not source.is_file():
            raise ValueError("The selected history image was not found.")
        target = directory / "reference.png"
        shutil.copyfile(source, target)
        return target
    return decode_reference(upload, directory)


def history_records() -> list[dict]:
    records = []
    if not OUTPUTS.exists():
        return records
    for directory in OUTPUTS.iterdir():
        if not directory.is_dir() or not re.fullmatch(r"[a-f0-9]{32}", directory.name):
            continue
        image = directory / "image.png"
        if not image.is_file():
            continue
        metadata_file = directory / "metadata.json"
        try:
            metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
        except (OSError, ValueError):
            metadata = {}
        records.append({
            "id": directory.name,
            "scene": metadata.get("scene") or "Earlier render",
            "prompt": metadata.get("prompt") or "",
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "steps": metadata.get("steps"),
            "input_mode": metadata.get("input_mode", "style"),
            "source_history_id": metadata.get("source_history_id"),
            "created_at": metadata.get("created_at") or image.stat().st_mtime,
            "image": f"/api/history/{directory.name}/image",
        })
    return sorted(records, key=lambda record: record["created_at"], reverse=True)[:50]


def complete_png(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            if file.read(8) != b"\x89PNG\r\n\x1a\n":
                return False
            file.seek(-8, os.SEEK_END)
            return file.read(4) == b"IEND"
    except (OSError, ValueError):
        return False


def latest_preview(directory: Path) -> tuple[Path, int] | None:
    for file in sorted(directory.glob("preview_*.png"), reverse=True):
        match = re.fullmatch(r"preview_(\d{3})\.png", file.name)
        if match and complete_png(file):
            return file, int(match.group(1)) + 1
    return None


def run_job(job_id: str, cmd: list[str]) -> None:
    global ACTIVE
    job = JOBS[job_id]
    try:
        with (Path(job["directory"]) / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
            with LOCK:
                job["process"] = process
                job["status"] = "running"
                cancelled = job.get("cancelled", False)
            if cancelled:
                process.terminate()
            code = process.wait()
        with LOCK:
            job["status"] = "cancelled" if job.get("cancelled") else ("done" if code == 0 and Path(job["output"]).is_file() else "error")
            job["returncode"] = code
    except OSError as exc:
        with LOCK:
            job["status"] = "error"
            job["error"] = str(exc)
    finally:
        with LOCK:
            ACTIVE = None
            job.pop("process", None)


class Handler(BaseHTTPRequestHandler):
    def authorized(self) -> bool:
        if not PASSWORD:
            return True
        value = self.headers.get("Authorization", "")
        try:
            scheme, encoded = value.split(" ", 1)
            credentials = base64.b64decode(encoded, validate=True).decode("utf-8") if scheme.lower() == "basic" else ""
        except (ValueError, UnicodeDecodeError, binascii.Error):
            credentials = ""
        if hmac.compare_digest(credentials, f"playground:{PASSWORD}"):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="qwen-image-2-1-abliterated-playground"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        size = int(self.headers.get("Content-Length", "0"))
        if size < 1 or size > MAX_BODY:
            raise ValueError("Request is empty or too large.")
        data = json.loads(self.rfile.read(size))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/healthz":
            return self.json_response(200, {"status": "ready", "backend": BACKEND})
        if not self.authorized():
            return
        path = urlparse(self.path).path
        if path == "/api/config":
            return self.json_response(200, {"defaults": DEFAULTS, "active": ACTIVE,
                                            "vae_cpu_default": sys.platform == "darwin",
                                            "backend": BACKEND, "rewriter": BACKEND == "diffusers" and os.environ.get("PLAYGROUND_REWRITER", "0") == "1"})
        if path == "/api/history":
            return self.json_response(200, {"items": history_records()})
        history_image = re.fullmatch(r"/api/history/([a-f0-9]{32})/image", path)
        if history_image:
            image = OUTPUTS / history_image.group(1) / "image.png"
            if not image.is_file():
                return self.json_response(404, {"error": "Image not found."})
            return self.send_file(image, "image/png")
        preview_match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/preview/(\d{3})\.png", path)
        if preview_match:
            job = JOBS.get(preview_match.group(1))
            if not job:
                return self.json_response(404, {"error": "Job not found."})
            file = Path(job["directory"]) / f"preview_{preview_match.group(2)}.png"
            if not complete_png(file):
                return self.json_response(404, {"error": "Preview is not ready."})
            return self.send_file(file, "image/png")
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})(?:/(image))?", path)
        if match:
            job = JOBS.get(match.group(1))
            if not job:
                return self.json_response(404, {"error": "Job not found."})
            if match.group(2):
                file = Path(job["output"])
                if job["status"] != "done" or not file.is_file():
                    return self.json_response(404, {"error": "Image is not ready."})
                return self.send_file(file, "image/png")
            log_path = Path(job["directory"]) / "run.log"
            log = log_path.read_text(errors="replace")[-5000:] if log_path.exists() else ""
            preview = latest_preview(Path(job["directory"]))
            return self.json_response(200, {"id": job["id"], "status": job["status"],
                                            "error": job.get("error"), "log": log,
                                            "preview": f"/api/jobs/{job['id']}/preview/{preview[0].name[8:]}" if preview else None,
                                            "preview_step": min(preview[1] * job.get("preview_interval", 4), job.get("steps", 30)) if preview else 0,
                                            "total_steps": job.get("steps"),
                                            "image": f"/api/jobs/{job['id']}/image" if job["status"] == "done" else None})
        file = STATIC / ("index.html" if path == "/" else path.lstrip("/"))
        if not file.resolve().is_relative_to(STATIC.resolve()) or not file.is_file():
            return self.send_error(404)
        self.send_file(file, mimetypes.guess_type(file.name)[0] or "application/octet-stream")

    def send_file(self, file: Path, content_type: str) -> None:
        content = file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        global ACTIVE
        if not self.authorized():
            return
        path = urlparse(self.path).path
        try:
            data = self.read_json() if path in {"/api/preview", "/api/jobs"} else {}
            if path == "/api/preview":
                return self.json_response(200, {"prompt": assembled_prompt(data)})
            if path == "/api/jobs":
                assembled_prompt(data)
                with LOCK:
                    if ACTIVE:
                        return self.json_response(409, {"error": "A generation is already running."})
                    ACTIVE = "preparing"
                job_id = uuid.uuid4().hex
                directory = OUTPUTS / job_id
                try:
                    directory.mkdir(parents=True, exist_ok=False)
                    reference = resolve_reference(data, directory)
                    output = directory / "image.png"
                    cmd = make_command(data, reference, output)
                    metadata = {"scene": str(data.get("scene", "")).strip(),
                                "prompt": assembled_prompt(data), "width": int(data.get("width", 768)),
                                "height": int(data.get("height", 768)), "steps": int(data.get("steps", 30)),
                                "input_mode": str(data.get("input_mode", "style")),
                                "source_history_id": str(data.get("source_history_id") or ""),
                                "seed": int(data.get("seed", 42)), "backend": BACKEND, "rewrite": bool(data.get("rewrite")),
                                "created_at": time.time()}
                    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
                    job = {"id": job_id, "directory": str(directory), "output": str(output),
                           "steps": int(data.get("steps", 30)),
                           "preview_interval": int(data.get("preview_interval", 4)) if data.get("live_preview", True) else 0,
                           "status": "starting"}
                    with LOCK:
                        JOBS[job_id] = job
                        ACTIVE = job_id
                    threading.Thread(target=run_job, args=(job_id, cmd), daemon=True).start()
                    return self.json_response(202, {"id": job_id})
                except Exception:
                    with LOCK:
                        ACTIVE = None
                    raise
            match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/cancel", path)
            if match:
                job = JOBS.get(match.group(1))
                if not job:
                    return self.json_response(404, {"error": "Job not found."})
                with LOCK:
                    job["cancelled"] = True
                    process = job.get("process")
                if process and process.poll() is None:
                    process.terminate()
                return self.json_response(200, {"status": "cancelling"})
            self.json_response(404, {"error": "Unknown endpoint."})
        except (ValueError, json.JSONDecodeError) as exc:
            self.json_response(400, {"error": str(exc)})


if __name__ == "__main__":
    if BACKEND not in {"sd-cpp", "diffusers"}:
        raise SystemExit("PLAYGROUND_BACKEND must be sd-cpp or diffusers")
    host = os.environ.get("PLAYGROUND_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost", "::1"} and not PASSWORD:
        raise SystemExit("Set PLAYGROUND_PASSWORD before binding qwen-image-2-1-abliterated-playground to a network interface.")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"qwen-image-2-1-abliterated-playground is ready at http://{host}:{port}", flush=True)
    server.serve_forever()
