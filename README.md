# qwen-image-2-1-abliterated-playground

A browser playground for Qwen Image 2.1 with a Heretic text encoder: scene and character prompts, style references, image editing, reproducible seeds, cancellation, and persistent image history.

**Mac runs quantized GGUF on Metal. RunPod runs unquantized BF16 with Diffusers.** The RunPod image has no Q4, INT8, FP8, or NVFP4 profile and does not fall back to quantization on low memory.

## What runs where

| Component | Mac | RunPod |
| --- | --- | --- |
| Image generator (DiT) | Leejet Q4 GGUF | Official Qwen Image 2.1 BF16 |
| Text encoder | Heretic Q4 GGUF + matching F16 projector | Heretic BF16 HF checkpoint, including vision weights |
| VAE | Official Qwen Image 2.1 BF16 | Official Qwen Image 2.1 BF16 |
| Optional scene rewriter | Not integrated; use an external llama.cpp rewriter if needed | Heretic PE-T2I BF16; included in first-boot downloads |
| Engine | stable-diffusion.cpp / Metal | PyTorch / CUDA / Diffusers |
| Intermediate image preview | Every step by default | Every step by default using the BF16 VAE |

Heretic modifies the language components. The DiT and VAE remain official weights; they are not advertised here as abliterated derivatives. Reduced refusal scores in the model cards are not a guarantee about every output. BF16 is the native unquantized checkpoint precision, not FP32 arithmetic throughout; architecture-required FP32 parameters/operations remain full precision. This is an experimental single-user application. Users are responsible for their inputs and outputs.

The RunPod template title is **Qwen Image 2.1 Uncensored Quickstart**. A [paste-ready template description](docs/runpod-description.md) is included.

## RunPod: deploy the published image

1. Open [Actions](https://github.com/rama-adi/qwen-image-2-1-abliterated-playground/actions/workflows/container.yml). Wait for **Test and publish RunPod image** to finish successfully. Pushes to `main` publish `latest` and an immutable `sha-<full commit>` tag; `v*` tags also publish version tags. You can rerun it with **Run workflow**.
2. Make the container package public for anonymous RunPod pulls: [your GitHub packages](https://github.com/rama-adi?tab=packages) → `qwen-image-2-1-abliterated-playground` → **Package settings → Change visibility → Public**. GitHub initially creates packages as private even when the repository is public. Alternatively, configure a RunPod registry credential as described below.
3. In [RunPod](https://console.runpod.io/), create a **GPU Pod**, not a Serverless endpoint, with a custom template:

   | Setting | Value |
   | --- | --- |
   | Container image | `ghcr.io/rama-adi/qwen-image-2-1-abliterated-playground:latest` |
   | GPU | Start with one RTX A6000 48 GB, L40/L40S 48 GB, or A100 80 GB |
   | Host driver | Must support CUDA 12.8 (the image runtime) |
   | Host RAM | At least 64 GB; 96 GB gives more headroom |
   | Container disk | 30 GB |
   | Persistent/network volume | 120 GB or more, mounted at `/workspace` |
   | HTTP port | `8765` |
   | TCP ports | None required |
   | Container start command | Leave blank; use the image entrypoint |
   | Environment | Set `PLAYGROUND_PASSWORD`; see table below |

4. First startup checks CUDA/BF16 support and downloads pinned model snapshots. Expect roughly 50 GB of weights with the rewriter, plus cache/metadata overhead. Startup can take several minutes. Application code lives at `/app`, so the `/workspace` mount cannot hide it.
5. Open port 8765 through RunPod's **Connect** menu, or `https://POD_ID-8765.proxy.runpod.net/`. Sign in as **`playground`** with your `PLAYGROUND_PASSWORD`.
6. Start with 1024×1024, 40 steps, CFG 1, and **Keep model weights in RAM** enabled. That uses model CPU offloading without quantizing weights. For the first functional test, use 512×512 and 1 step; this is not a quality setting.

GPU sizing above is an estimate, not a measured peak guarantee. The BF16 image pipeline has about 32 GB of weights before activations/cache. The rewriter runs first and is unloaded before the image pipeline. Each render uses a fresh worker process so cancellation/completion releases CUDA allocations. Offloading trades speed for lower VRAM demand. If a render runs out of memory, lower the resolution or set `PLAYGROUND_KV_CACHE=0`; precision stays BF16.

### Environment variables and credentials

**GitHub Actions needs no manually created secrets.** It uses GitHub's automatic `GITHUB_TOKEN` with job-scoped `packages: write`. No Docker Hub account, Hugging Face token, or RunPod API key is required to build and publish this image. If Actions is disabled, enable it under repository **Settings → Actions → General**.

Set these under your **RunPod template/Pod → Environment Variables**. Use the **key icon / RunPod Secrets** for `PLAYGROUND_PASSWORD` and `HF_TOKEN`; ordinary environment-variable fields are not encrypted. [`.env.example`](.env.example) lists the common values; the application does not automatically load `.env` files.

| Variable | Required? / default | Where to get it / purpose |
| --- | --- | --- |
| `PLAYGROUND_PASSWORD` | **Required** | Generate a unique password in your password manager or with `openssl rand -hex 24`. Browser username is `playground`. |
| `HF_TOKEN` | Optional, unset | [Hugging Face → Settings → Access Tokens](https://huggingface.co/settings/tokens). Create a **read** token if anonymous downloads are throttled or access requirements change. Current public models normally need none. |
| `PLAYGROUND_REWRITER` | `1` | Set `0` to skip the optional BF16 prompt-rewriter download and hide its checkbox. This is a setting you choose, not a credential. |
| `PLAYGROUND_DOWNLOAD_MODELS` | `1` | Keep enabled. Set `0` only after the same pinned checkpoints are already present in `/workspace/models`. |
| `PLAYGROUND_KV_CACHE` | `1` | Set `0` to reduce inference cache memory. |
| `PLAYGROUND_REWRITE_TOKENS` | `6144` | Maximum generated rewrite tokens. Increase if the log reports an incomplete rewrite; this uses more memory/time. |
| `PLAYGROUND_MODEL_DIR` | `/workspace/models` | Usually leave unchanged. |
| `PLAYGROUND_OUTPUT_DIR` | `/workspace/outputs` | Usually leave unchanged. |
| `HF_HOME` | `/workspace/huggingface` | Usually leave unchanged; keeps Hugging Face cache on the volume. |
| `PORT` | `8765` | If changed, also change the exposed HTTP port and healthcheck. |

For a **private GHCR package**, create a [GitHub personal access token (classic)](https://github.com/settings/tokens/new) with `read:packages`. In RunPod's container registry credentials, set username `rama-adi`, password to that token, and attach the credential to the template. Do not put this token in the container's environment or commit it. A public package needs no registry credentials.

The workflow publishes the image; it does not rent a GPU or deploy paid Pods. No `RUNPOD_API_KEY` is needed for deployment through the console.

### Prompt rewriting and image editing

Enable **Expand scene with Heretic** to run the unquantized PE-T2I rewriter before generation. Its pinned `system_prompt.txt` is loaded automatically. The original prompt is retained in `request.json`, the actual render prompt is saved in `metadata.json`, and the rewrite JSON is saved beside the image. Your chosen width/height override its suggested ratio. Invalid rewrite JSON fails the job visibly instead of silently changing the prompt.

For **Edit / reprompt image**, use the original edit instruction with the BF16 Heretic image-pipeline encoder. The scene rewriter is disabled for editing: PE-T2I is not PE-I2I, and the linked PE-I2I GGUF is quantized. This project does not substitute that quantized model in the BF16 deployment. You can upload an image or select **Edit this image** in history.

Style references are approximate: the reference reaches both the vision and diffusion paths, so its composition or subject can leak into the result. Character cards become one structured natural-language prompt; they are not independently conditioned character regions.

### Check the deployed GPU

For an end-to-end check, run the included script from a checkout (locally or on the Pod):

```bash
PLAYGROUND_URL=https://POD_ID-8765.proxy.runpod.net \
PLAYGROUND_PASSWORD='your-password' python3 scripts/smoke_gpu.py
```

It generates a one-step 512×512 image, edits that result, and tests a BF16 rewrite plus generation if enabled. It polls jobs rather than holding a proxy request open. Inspect the job logs for verified unquantized parameters and peak CUDA allocation; use 40 steps for visual-quality evaluation. This script consumes GPU time on the Pod you already started.

## Mac: quantized GGUF / Metal

Install Xcode command-line tools, Git, Python 3, and CMake (`brew install cmake` if you use Homebrew). From the repository:

```bash
bash setup-macos.sh
.venv/bin/python app.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). No password is required while bound to localhost. Start at 768×768 on a 24 GB Apple Silicon Mac with **Decode VAE on CPU** enabled; reduce to 512×512 if needed. Setup downloads approximately 11 GB:

- `qwen_image_2.1-Q4_K.gguf` — [Leejet DiT](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF)
- `qwen3vl_8b_heretic-Q4_K_M.gguf` and `mmproj-qwen3vl_8b_heretic-f16.gguf` — [Heretic encoder](https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF)
- `qwen_image_2.1_vae_bf16.safetensors` — [official VAE](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae)

Setup creates a `.venv` with the small WebSocket transport dependency. If you already have the models, install it with `python3 -m venv .venv` followed by `.venv/bin/python -m pip install -r requirements-ui.txt`.

Model paths can be changed in **Model settings** on Mac. RunPod deliberately ignores browser-supplied model paths and uses its pinned BF16 checkpoint directories. Progress streams through an authenticated WebSocket on the same port as the UI, with heartbeat, reconnection, and HTTP polling fallback. The image area and log both show the current sampling step. Live previews default to every step on both engines; each preview adds a VAE decode and can slow down renders considerably. Increase the interval or disable previews for speed. RunPod previews use the same BF16 VAE, without a quantized preview model. Images and metadata persist under `outputs/` locally or `/workspace/outputs` on RunPod.

## Development and validation

```bash
.venv/bin/python -m unittest discover -s tests -v
node --check static/app.js
bash -n setup-macos.sh runpod/entrypoint.sh
```

Local Mac smoke tests on 2026-09-23 generated valid 512×512 PNGs with the quantized models, including a two-step run with a VAE preview after every step and browser WebSocket updates. This verifies loading and generation, not image quality.

CI tests request validation, backend separation, rejection of quantized checkpoint tensors before casting, rewrite parsing, worker cleanup, authenticated WebSocket streaming, per-step event ordering, and reconnection. The workflow builds the Linux AMD64 image, verifies model-class imports, and smoke-tests authentication/UI startup with a volume mounted at `/workspace` before publishing. GitHub-hosted CI has no NVIDIA GPU: its smoke test explicitly skips CUDA preflight and model downloads, and **does not certify GPU inference or image quality**. The GPU smoke script supplies that additional check on your deployed Pod.

Manual build:

```bash
docker buildx build --platform linux/amd64 -f runpod/Dockerfile \
  -t ghcr.io/rama-adi/qwen-image-2-1-abliterated-playground:local --load .
```

`models.lock.json` pins the HF checkpoint revisions; `runpod/requirements.txt` pins the main inference dependencies and Diffusers commit. Model weights, generated images, credentials, and the local stable-diffusion.cpp checkout are excluded from Git and Docker context. There is no application-level content filter or external prompt-rewriting service in this pipeline.

## Sources and model licenses

- [Original post](https://x.com/xiangxiang103/status/2101902711243542657) and [correction / prompt-rewriter update](https://x.com/xiangxiang103/status/2102704760122146992).
- [Official Qwen Image 2.1](https://github.com/QwenLM/Qwen-Image-2.1), [Heretic BF16 encoder](https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic), and [Heretic BF16 PE-T2I](https://huggingface.co/pottokao/Qwen-Image-2.1-PE-T2I-Heretic).
- [stable-diffusion.cpp Qwen Image 2.1 guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md).
- [RunPod templates](https://docs.runpod.io/pods/templates/create-custom-template) and [storage](https://docs.runpod.io/pods/storage/types).

The Qwen image model and PE rewriter use the Qwen Research License (non-commercial research/evaluation; other use may require a separate license). The Heretic text encoder derives from Apache-2.0 Qwen3-VL. Responsibility for output does not replace those model licenses; consult the licenses shipped with each checkpoint.
