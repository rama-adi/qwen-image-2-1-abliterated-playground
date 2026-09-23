"""Download only the pinned, unquantized HF checkpoints used by RunPod."""
import argparse
import json
import os
from pathlib import Path

LOCK = json.loads((Path(__file__).parent / "models.lock.json").read_text())
HF_FILES = ["*.json", "*.jinja", "model-*.safetensors", "system_prompt.txt", "LICENSE", "NOTICE"]


def download(include_rewriter=False):
    from huggingface_hub import snapshot_download
    root = Path(os.environ.get("PLAYGROUND_MODEL_DIR", "/workspace/models"))
    for name in ("pipeline", "encoder", "rewriter"):
        if name == "rewriter" and not include_rewriter:
            continue
        entry = LOCK[name]
        patterns = (["model_index.json", "scheduler/*", "processor/*", "tokenizer/*",
                     "transformer/*", "vae/*", "LICENSE"] if name == "pipeline" else HF_FILES)
        print(f"Downloading {name}: {entry['repo']} @ {entry['revision']}", flush=True)
        snapshot_download(entry["repo"], revision=entry["revision"],
                          local_dir=root / name, allow_patterns=patterns)
    (root / "models.lock.json").write_text(json.dumps(LOCK, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewriter", action="store_true")
    download(parser.parse_args().rewriter)
