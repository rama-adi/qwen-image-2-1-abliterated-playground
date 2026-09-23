# Qwen Image 2.1 Uncensored Quickstart

Generate and edit images in your browser with Qwen Image 2.1, a BF16 Heretic text encoder, and an optional BF16 Heretic prompt rewriter. Includes per-step image previews, live progress, character descriptions, style references, reproducible seeds, and saved render history.

## About

Built on [qwen-image-2-1-abliterated-playground](https://github.com/rama-adi/qwen-image-2-1-abliterated-playground), this single-user playground combines the official Qwen Image 2.1 diffusion model and VAE with Heretic language components. RunPod uses unquantized BF16 weights throughout, with CPU offloading to manage GPU memory. The repository also provides a separate quantized GGUF / Metal setup for Mac. Generation progress streams through WebSockets to both the image area and generation log, with automatic reconnection and HTTP fallback.

**Uncensored by design:** Heretic / abliterated behavior is intentional, and this application adds no content filter. This is an experimental tool that can generate explicit, sensitive, or offensive material. Exercise discretion. You are responsible for your prompts, generated content, and any use or sharing of the output.

## Setup

- GPU: NVIDIA with native BF16 support; **48 GB VRAM or more** recommended (RTX A6000, L40/L40S, A100 80 GB). Host driver must support **CUDA 12.8**.
- Host RAM: **64 GB or more**.
- Container disk: **30 GB**. Persistent volume: **120 GB**, mounted at `/workspace`.
- Expose **HTTP port 8765**. Leave the container start command blank.
- Container image:

```text
ghcr.io/rama-adi/qwen-image-2-1-abliterated-playground:latest
```

Set these in the RunPod template's **Environment Variables**:

```text
PLAYGROUND_PASSWORD=password
PLAYGROUND_REWRITER=1
```

Default login: username **`playground`**, password **`password`**. Change the shared default before exposing your Pod; use RunPod's key icon / Secrets for your replacement password. Generate one with:

```bash
openssl rand -hex 24
```

The password is a template setting; the container requires `PLAYGROUND_PASSWORD` to be set. Set `PLAYGROUND_REWRITER=0` to skip the optional scene rewriter.

Optional: add `HF_TOKEN` as a RunPod Secret using a read token from [Hugging Face settings](https://huggingface.co/settings/tokens). Public downloads work without it. No RunPod API key, GitHub token, or Docker Hub token is needed inside the Pod.

## Use

1. Deploy and watch startup logs. First boot downloads about **50 GB of weights** with rewriting enabled. Later starts reuse the persistent volume.
2. Open **Connect → HTTP service on port 8765** and sign in.
3. Enter a scene, optionally add characters or a style reference, then click **Generate image**.
4. Start at **1024×1024, 40 steps, CFG 1**, with **Keep model weights in RAM** enabled. For a quick loading test, use **512×512, 1 step**.
5. Enable **Expand scene with Heretic** for prompt rewriting. To edit an image, choose **Edit / reprompt image** and describe the changes; rewriting is disabled in edit mode.
6. Watch per-step previews and progress in the image area and log. Increase the preview interval or disable previews for faster rendering.

Outputs, prompts, seeds, and logs are saved to `/workspace/outputs`; weights to `/workspace/models`. Use a network volume to retain these after deleting a Pod.

## Troubleshooting

- **UI not ready:** First-boot downloads must finish before the UI starts. Check startup logs.
- **Container exits:** Check `PLAYGROUND_PASSWORD`, GPU availability, and CUDA/BF16 compatibility.
- **Out of memory:** Keep offloading enabled, reduce resolution, disable previews, or set `PLAYGROUND_KV_CACHE=0` and restart. Weights stay BF16.
- **Slow rendering:** Reduce preview frequency and disable scene rewriting when unused.
- **Download errors:** Check disk space; add `HF_TOKEN` and restart to resume downloads.
- **Progress disconnects:** Expose port 8765 as HTTP. The UI reconnects automatically; refresh to rejoin an active render.

From a local project checkout, test generation, editing, and enabled rewriting on your running Pod (uses GPU time):

```bash
PLAYGROUND_URL=https://POD_ID-8765.proxy.runpod.net \
PLAYGROUND_PASSWORD='your-password' \
python3 scripts/smoke_gpu.py
```

Downloader help inside the container:

```bash
python /app/download_models.py --help
```

## Contributors and support

Maintainer: [@rama-adi](https://github.com/rama-adi). Support: [GitHub Issues](https://github.com/rama-adi/qwen-image-2-1-abliterated-playground/issues). Models: Qwen and pottokao's Heretic checkpoints.

Image model and prompt rewriter: **Qwen Research License** (non-commercial research/evaluation). Heretic text encoder: **Apache-2.0**.

## Version history

- **0.1 — Initial release:** BF16 RunPod generation/editing, Heretic rewriting, per-step previews and progress, persistent history, password login, and separate quantized Mac setup.
