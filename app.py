"""Small local UI for Qwen Image 2.1 through stable-diffusion.cpp."""

from __future__ import annotations

import atexit
import base64
import binascii
import hmac
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import select
import shutil
import subprocess
import sys
import threading
import time
import uuid
from lora_store import Store as LoraStore, CHUNK_SIZE
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
OUTPUTS = Path(os.environ.get("PLAYGROUND_OUTPUT_DIR", ROOT / "outputs"))
REFERENCES = OUTPUTS / "references"
MODEL_DIR = Path(os.environ.get("PLAYGROUND_MODEL_DIR", "models"))
BACKEND = os.environ.get("PLAYGROUND_BACKEND", "sd-cpp")
LORAS = LoraStore(os.environ.get("PLAYGROUND_LORA_DIR", str(MODEL_DIR / "loras")))
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
WARM_PROCESS = None
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
    seed = integer("seed", 0, 2**32 - 1, 0)
    if seed == 0:
        seed = secrets.randbelow(2**32 - 1) + 1
    data["seed"] = seed
    lora = LORAS.selection(data.get("lora_id"), data.get("lora_strength", 1))
    if BACKEND == "diffusers":
        if data.get("rewrite") and os.environ.get("PLAYGROUND_REWRITER", "0") != "1":
            raise ValueError("Prompt rewriting is disabled. Set PLAYGROUND_REWRITER=1 and restart the Pod.")
        if data.get("rewrite") and data.get("input_mode") == "edit":
            raise ValueError("Disable scene rewriting for image edits.")
        request = {"lora": lora, "prompt": assembled_prompt(data), "reference": str(reference) if reference else None,
                   "output": str(output), "width": width, "height": height, "steps": steps,
                   "cfg": cfg, "seed": seed, "negative": str(data.get("negative", "")),
                   "offload": bool(data.get("offload", True)), "rewrite": bool(data.get("rewrite")),
                   "input_mode": data.get("input_mode", "style"),
                   "live_preview": bool(data.get("live_preview", True)),
                   "preview_interval": integer("preview_interval", 1, 20, 1)}
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
    prompt = assembled_prompt(data)
    if lora and lora['strength'] != 0 and lora.get("format") == "comfy-dora":
        raise ValueError("This ComfyUI DoRA adapter requires the RunPod backend; the Mac engine does not support its magnitude tensors.")
    if lora and lora['strength'] != 0:
        cmd += ["--lora-model-dir", str(LORAS.root.resolve())]
        prompt += f" <lora:{lora['id']}:{lora['strength']}>"
    cmd += ["-p", prompt, "--seed", str(seed), "--cfg-scale", str(cfg), "--steps", str(steps),
            "--sampling-method", "euler", "--diffusion-fa", "-W", str(width),
            "-H", str(height), "-o", str(output)]
    if data.get("live_preview", True):
        interval = integer("preview_interval", 1, 20, 1)
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
    reference_id = str(data.get("reference_id") or "")
    if sum(bool(value) for value in (upload, history_id, reference_id)) > 1:
        raise ValueError("Choose an uploaded image or a history image, not both.")
    if reference_id:
        source = reference_path(reference_id)
        target = directory / source.name
        shutil.copyfile(source, target)
        return target
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


def reference_path(id):
    if not re.fullmatch(r"[a-f0-9]{32}", id):
        raise ValueError("Invalid reference ID.")
    for root in (REFERENCES, OUTPUTS):
        directory = root / id
        if directory.is_symlink(): continue
        for suffix in ('png', 'jpg', 'webp'):
            file = directory / ('reference.' + suffix)
            if file.is_file() and not file.is_symlink(): return file
    raise ValueError("Saved reference was not found.")


def reference_records():
    result = {}
    for root in (OUTPUTS, REFERENCES):
        if not root.exists(): continue
        for directory in root.iterdir():
            if not re.fullmatch(r'[a-f0-9]{32}', directory.name): continue
            try:
                file = reference_path(directory.name)
                info = directory / 'reference.json'
                name = json.loads(info.read_text()).get('name') if info.is_file() else None
                result[directory.name] = {'id': directory.name, 'name': name or ('Earlier reference ' + directory.name[:8]),
                    'image': f'/api/references/{directory.name}/image', 'created_at': file.stat().st_mtime}
            except (OSError, ValueError): continue
    return sorted(result.values(), key=lambda item: item['created_at'], reverse=True)


def save_reference(data):
    id = uuid.uuid4().hex
    directory = REFERENCES / id
    directory.mkdir(parents=True)
    try:
        file = decode_reference(str(data.get('reference') or ''), directory)
        if file is None: raise ValueError('Choose a reference image.')
        name = str(data.get('name') or 'Uploaded reference')[:200]
        (directory / 'reference.json').write_text(json.dumps({'name': name}))
        return {'id': id, 'name': name, 'image': f'/api/references/{id}/image'}
    except Exception:
        shutil.rmtree(directory)
        raise


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
        # Recover settings from older RunPod requests; omit paths and image payloads.
        request = {}
        try:
            request = json.loads((directory / 'request.json').read_text())
        except (OSError, ValueError): pass
        settings = {key: metadata.get(key, request.get(key)) for key in (
            'scene', 'prompt', 'negative', 'characters', 'width', 'height', 'steps', 'cfg', 'seed',
            'backend', 'precision', 'offload', 'vae_cpu', 'vae_tiling', 'kv_cache', 'rewrite',
            'live_preview', 'preview_interval', 'input_mode', 'source_history_id', 'reference_id', 'model_revisions')}
        lora = metadata.get('lora', request.get('lora'))
        settings['lora'] = {k: lora.get(k) for k in ('id', 'name', 'sha256', 'strength', 'format')} if lora else None
        settings['has_reference'] = any((directory / ('reference.' + ext)).is_file() for ext in ('png', 'jpg', 'webp'))
        records.append({
            "settings": settings,
            "id": directory.name,
            "scene": metadata.get("scene") or "Earlier render",
            "prompt": metadata.get("prompt") or "",
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "steps": metadata.get("steps"),
            "seed": metadata.get("seed"),
            "input_mode": metadata.get("input_mode", "style"),
            "source_history_id": metadata.get("source_history_id"),
            "created_at": metadata.get("created_at") or image.stat().st_mtime,
            "image": f"/api/history/{directory.name}/image",
        })
    return sorted(records, key=lambda record: record["created_at"], reverse=True)[:50]


def delete_history(history_id=None):
    if history_id is not None and not re.fullmatch(r"[a-f0-9]{32}", history_id):
        raise ValueError("Invalid history image ID.")
    with LOCK:
        if ACTIVE:
            raise ValueError("Wait for the current render to finish or cancel it before deleting history.")
        directories = [OUTPUTS / history_id] if history_id else list(OUTPUTS.glob("*"))
        deleted = []
        for directory in directories:
            if (not re.fullmatch(r"[a-f0-9]{32}", directory.name) or directory.is_symlink()
                    or not directory.is_dir() or not (directory / "image.png").is_file()):
                continue
            shutil.rmtree(directory)
            JOBS.pop(directory.name, None)
            deleted.append(directory.name)
        return deleted


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


def consume_progress(job: dict, line: str) -> None:
    line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
    event = None
    if "PLAYGROUND_EVENT " in line:
        try:
            event = json.loads(line.split("PLAYGROUND_EVENT ", 1)[1])
        except ValueError:
            return
    else:
        match = re.search(r"\|\s*(\d+)/(\d+)\s*-\s*[\d.]+(?:s/it|it/s)", line)
        if match and int(match[2]) == job.get("steps"):
            event = {"stage": "sampling", "step": int(match[1])}
        elif "decoding 1 latents" in line:
            event = {"stage": "decoding" if job.get("step", 0) >= job.get("steps", 1) else "preview"}
    if not isinstance(event, dict):
        return
    with LOCK:
        update = {key: event[key] for key in ("stage", "step", "preview_step") if key in event}
        if all(job.get(key) == value for key, value in update.items()):
            return
        job.update(update)
        job.setdefault("events", []).append({"stage": job.get("stage", "loading"),
            "step": job.get("step", 0), "preview_step": job.get("preview_step", 0), "status": "running"})


def job_snapshot(job: dict) -> dict:
    log_path = Path(job["directory"]) / "run.log"
    log = ""
    if log_path.exists():
        with log_path.open("rb") as stream:
            stream.seek(max(0, log_path.stat().st_size - 7000))
            log = stream.read().decode("utf-8", errors="replace")
        log = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log).replace("\r", "\n")
    preview = latest_preview(Path(job["directory"]))
    return {"id": job["id"], "status": job["status"], "stage": job.get("stage", "loading"),
            "step": job.get("step", 0), "total_steps": job.get("steps"),
            "error": job.get("error"), "log": log,
            "preview": f"/api/jobs/{job['id']}/preview/{preview[0].name[8:]}" if preview else None,
            "preview_step": min(preview[1] * job.get("preview_interval", 1), job.get("steps", 30)) if preview else 0,
            "image": f"/api/jobs/{job['id']}/image" if job["status"] == "done" else None}


def stop_warm_worker():
    global WARM_PROCESS
    process, WARM_PROCESS = WARM_PROCESS, None
    if process is not None:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        for stream in (process.stdin, process.stdout):
            if stream:
                stream.close()


atexit.register(stop_warm_worker)


def run_job(job_id: str, cmd: list[str]) -> None:
    global ACTIVE, WARM_PROCESS
    job = JOBS[job_id]
    warm = (BACKEND == "diffusers" and os.environ.get("PLAYGROUND_KEEP_WARM", "1") == "1"
            and len(cmd) > 3 and cmd[2] == str(ROOT / "inference_worker.py"))
    try:
        with (Path(job["directory"]) / "run.log").open("w", encoding="utf-8") as log:
            if warm:
                if WARM_PROCESS is None or WARM_PROCESS.poll() is not None:
                    stop_warm_worker()
                    WARM_PROCESS = subprocess.Popen(cmd[:-1] + ["--serve"], cwd=ROOT,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                process = WARM_PROCESS
            else:
                process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            with LOCK:
                job["process"] = process
                job["status"] = "running"
                cancelled = job.get("cancelled", False)
            if cancelled:
                process.terminate()
            if warm and not cancelled:
                process.stdin.write((json.dumps(cmd[-1]) + "\n").encode())
                process.stdin.flush()
            completed = False
            pending = b""
            while chunk := os.read(process.stdout.fileno(), 4096):
                log.write(chunk.decode("utf-8", errors="replace"))
                log.flush()
                pending += chunk
                lines = re.split(rb"[\r\n]", pending)
                pending = lines.pop()
                for line in lines:
                    if warm and line == b"PLAYGROUND_DONE 0":
                        completed = True
                        continue
                    consume_progress(job, line.decode("utf-8", errors="replace"))
                # Native progress bars start with CR and have no terminating newline.
                # Parse their partial line now instead of waiting for the next step.
                if pending:
                    consume_progress(job, pending.decode("utf-8", errors="replace"))
                if completed:
                    break
            if pending:
                consume_progress(job, pending.decode("utf-8", errors="replace"))
            if warm and completed:
                code = 0
            else:
                if warm:
                    stop_warm_worker()
                else:
                    process.stdout.close()
                code = process.wait()
        with LOCK:
            job["status"] = "cancelled" if job.get("cancelled") else ("done" if code == 0 and Path(job["output"]).is_file() else "error")
            job["returncode"] = code
    except OSError as exc:
        if warm:
            stop_warm_worker()
        with LOCK:
            job["status"] = "error"
            job["error"] = str(exc)
    finally:
        with LOCK:
            ACTIVE = None
            job.pop("process", None)


def websocket_origin_allowed(origin, host):
    if not origin:
        return True
    parsed = urlparse(origin)
    if parsed.scheme not in {"http", "https"} or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        return False
    if parsed.netloc == host:
        return True
    # RunPod's reverse proxy may replace Host with its upstream address.
    # Trust only an explicitly configured public origin or this Pod's exact URL.
    allowed = os.environ.get("PLAYGROUND_PUBLIC_ORIGIN", "").rstrip("/")
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    runpod_origin = f"https://{pod_id}-{os.environ.get('PORT', '8765')}.proxy.runpod.net" if pod_id else ""
    return origin.rstrip("/") in {value for value in (allowed, runpod_origin) if value}


class Handler(BaseHTTPRequestHandler):
    rbufsize = 0

    def stream_job(self, job: dict) -> None:
        # Same-origin Basic authentication has already run before the upgrade.
        origin = self.headers.get("Origin")
        if not websocket_origin_allowed(origin, self.headers.get("Host")):
            return self.json_response(403, {"error": "WebSocket origin does not match this host."})
        if self.headers.get("Upgrade", "").lower() != "websocket":
            return self.json_response(426, {"error": "WebSocket upgrade required."})
        try:
            from wsproto import WSConnection, ConnectionType
            from wsproto.events import Request, AcceptConnection, TextMessage, CloseConnection, Ping, Pong
            from wsproto.utilities import RemoteProtocolError, LocalProtocolError
        except ImportError:
            return self.json_response(503, {"error": "Install requirements-ui.txt for WebSocket streaming."})
        ws = WSConnection(ConnectionType.SERVER)
        self.close_connection = True
        upgraded = False
        try:
            handshake = self.requestline + "\r\n" + "".join(f"{k}: {v}\r\n" for k, v in self.headers.items()) + "\r\n"
            ws.receive_data(handshake.encode("latin-1"))
            if not any(isinstance(event, Request) for event in ws.events()):
                return self.json_response(400, {"error": "Invalid WebSocket handshake."})
            self.connection.sendall(ws.send(AcceptConnection()))
            upgraded = True
            cursor = len(job.get("events", []))
            previous = None
            last_ping = time.monotonic()
            last_pong = last_ping
            self.connection.settimeout(5)
            while True:
                with LOCK:
                    events = job.get("events", [])[cursor:]
                    cursor += len(events)
                snapshot = job_snapshot(job)
                # Preserve every sampled step even if several completed between sends.
                for event in events:
                    self.connection.sendall(ws.send(TextMessage(data=json.dumps({**snapshot, **event}))))
                payload = json.dumps(snapshot)
                if payload != previous:
                    self.connection.sendall(ws.send(TextMessage(data=payload)))
                    previous = payload
                if snapshot["status"] in {"done", "error", "cancelled"}:
                    self.connection.sendall(ws.send(CloseConnection(code=1000, reason="Job finished")))
                    return
                now = time.monotonic()
                if now - last_ping >= 20:
                    if now - last_pong > 60:
                        return
                    self.connection.sendall(ws.send(Ping(payload=b"progress")))
                    last_ping = now
                if select.select([self.connection], [], [], 0.15)[0]:
                    chunk = self.connection.recv(4096)
                    if not chunk:
                        return
                    ws.receive_data(chunk)
                    for event in ws.events():
                        if isinstance(event, CloseConnection):
                            self.connection.sendall(ws.send(event.response()))
                            return
                        if isinstance(event, Ping):
                            self.connection.sendall(ws.send(event.response()))
                        elif isinstance(event, Pong):
                            last_pong = now
                        elif isinstance(event, TextMessage):
                            self.connection.sendall(ws.send(CloseConnection(code=1008, reason="Progress stream is read-only")))
                            return
        except (OSError, RemoteProtocolError, LocalProtocolError, ValueError):
            if not upgraded:
                self.json_response(400, {"error": "Invalid WebSocket handshake."})

    def authorized(self) -> bool:
        if not PASSWORD:
            return True
        value = self.headers.get("Authorization", "")
        try:
            scheme, encoded = value.split(" ", 1)
            credentials = base64.b64decode(encoded, validate=True).decode("utf-8") if scheme.lower() == "basic" else ""
        except (ValueError, UnicodeDecodeError, binascii.Error):
            credentials = ""
        if hmac.compare_digest(credentials.encode("utf-8"), f"playground:{PASSWORD}".encode("utf-8")):
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
        # rbufsize=0 is required by WebSocket reads; raw HTTP reads may be short.
        body = bytearray()
        while len(body) < size:
            chunk = self.rfile.read(min(size - len(body), 65536))
            if not chunk:
                raise ValueError("Upload ended before the complete request arrived. Please retry.")
            body.extend(chunk)
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid JSON request. Please retry the upload.") from exc
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        return data

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/healthz":
            return self.json_response(200, {"status": "ready", "backend": BACKEND})
        if not self.authorized():
            return
        path = urlparse(self.path).path
        stream = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/events", path)
        if stream:
            job = JOBS.get(stream.group(1))
            if not job:
                return self.json_response(404, {"error": "Job not found."})
            return self.stream_job(job)
        if path == "/api/config":
            return self.json_response(200, {"defaults": DEFAULTS, "active": ACTIVE,
                                            "vae_cpu_default": sys.platform == "darwin",
                                            "backend": BACKEND, "rewriter": BACKEND == "diffusers" and os.environ.get("PLAYGROUND_REWRITER", "0") == "1"})
        if path == "/api/references":
            return self.json_response(200, {"items": reference_records()})
        reference_match = re.fullmatch(r"/api/references/([a-f0-9]{32})/image", path)
        if reference_match:
            try:
                file = reference_path(reference_match.group(1))
                return self.send_file(file, mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            except ValueError:
                return self.json_response(404, {"error": "Reference not found."})
        if path == "/api/loras":
            return self.json_response(200, {"items": LORAS.items()})
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
            return self.json_response(200, job_snapshot(job))
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

    def do_DELETE(self):
        if not self.authorized():
            return
        path = urlparse(self.path).path
        upload = re.fullmatch(r"/api/loras/uploads/([a-f0-9]{32})", path)
        if upload:
            LORAS.cancel(upload.group(1))
            return self.json_response(200, {"cancelled": True})
        match = re.fullmatch(r"/api/history(?:/([a-f0-9]{32}))?", path)
        if not match:
            return self.json_response(404, {"error": "Unknown endpoint."})
        try:
            deleted = delete_history(match.group(1))
            return self.json_response(200, {"deleted": deleted})
        except ValueError as exc:
            return self.json_response(409, {"error": str(exc)})
        except OSError:
            return self.json_response(500, {"error": "Could not delete history. Refresh and retry."})

    def do_POST(self) -> None:
        global ACTIVE
        if not self.authorized():
            return
        path = urlparse(self.path).path
        try:
            if path == "/api/references":
                return self.json_response(201, save_reference(self.read_json()))
            if path == "/api/loras/uploads":
                upload = self.read_json()
                return self.json_response(201, LORAS.begin(upload.get('name'), upload.get('size')))
            upload = re.fullmatch(r"/api/loras/uploads/([a-f0-9]{32})/(chunk|finish)", path)
            if upload:
                if upload.group(2) == 'finish':
                    return self.json_response(200, LORAS.finish(upload.group(1)))
                size = int(self.headers.get('Content-Length', '0'))
                offset = int(self.headers.get('X-Upload-Offset', '-1'))
                if not 0 < size <= CHUNK_SIZE:
                    raise ValueError('Upload chunks must be at most 8 MiB.')
                content = bytearray()
                while len(content) < size:
                    chunk = self.rfile.read(min(65536, size - len(content)))
                    if not chunk: raise ValueError('Upload was interrupted.')
                    content.extend(chunk)
                return self.json_response(200, LORAS.append(upload.group(1), offset, content))
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
                                "lora": LORAS.selection(data.get("lora_id"), data.get("lora_strength", 1)),
                                "created_at": time.time()}
                    metadata.update({key: data.get(key) for key in ('negative', 'characters', 'reference_id')})
                    metadata.update(cfg=float(data.get('cfg', 6)), offload=bool(data.get('offload', BACKEND == 'diffusers')),
                        vae_cpu=bool(data.get('vae_cpu', True)), live_preview=bool(data.get('live_preview', True)),
                        preview_interval=int(data.get('preview_interval', 1)),
                        kv_cache=os.environ.get('PLAYGROUND_KV_CACHE', '1') == '1')
                    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
                    job = {"id": job_id, "directory": str(directory), "output": str(output),
                           "steps": int(data.get("steps", 30)),
                           "preview_interval": int(data.get("preview_interval", 1)) if data.get("live_preview", True) else 0,
                           "status": "starting", "stage": "loading", "step": 0, "events": []}
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
                    if ACTIVE != match.group(1) or job["status"] in {"done", "error", "cancelled"}:
                        return self.json_response(409, {"error": "Job is no longer running."})
                    job["cancelled"] = True
                    process = job.get("process")
                if process and process.poll() is None:
                    process.terminate()
                return self.json_response(200, {"status": "cancelling"})
            self.json_response(404, {"error": "Unknown endpoint."})
        except OSError as exc:
            self.json_response(400, {"error": "File operation failed. Check storage space and retry the upload."})
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
