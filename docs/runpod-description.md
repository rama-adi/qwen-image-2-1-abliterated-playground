# Qwen Image 2.1 Uncensored Quickstart

Generate and edit images in your browser with Qwen Image 2.1, a BF16 Heretic text encoder, and an optional BF16 Heretic prompt rewriter. Includes per-step image previews, live progress, character descriptions, style references, reproducible seeds, and saved render history.

## About

Built on [qwen-image-2-1-abliterated-playground](https://github.com/rama-adi/qwen-image-2-1-abliterated-playground), this single-user playground combines the official Qwen Image 2.1 diffusion model and VAE with Heretic language components. RunPod uses unquantized BF16 weights throughout, with CPU offloading to manage GPU memory. The repository also provides a separate quantized GGUF / Metal setup for Mac. Generation progress streams through WebSockets to both the image area and generation log, with automatic reconnection and HTTP fallback.

**Uncensored by design:** Heretic / abliterated behavior is intentional, and this application adds no content filter. This is an experimental tool that can generate explicit, sensitive, or offensive material. Exercise discretion. You are responsible for your prompts, generated content, and any use or sharing of the output.

## Deploy the template

The container image, HTTP port, storage paths, and model settings are already configured.

1. Select an NVIDIA GPU with native BF16 support and **48 GB VRAM or more** recommended: RTX A6000, L40/L40S, or A100 80 GB. Choose a host with **64 GB RAM or more** and a driver compatible with **CUDA 12.8**.
2. In **Environment Variables**, change **`PLAYGROUND_PASSWORD`** from **`password`** to your own password. The login username is **`playground`**. Use RunPod's key icon / Secrets for your replacement password.
3. Optionally adjust the settings below, then deploy. Keep the template's **30 GB container disk** and **120 GB persistent storage** allocation, or increase them. Use a network volume if you want to keep your models and outputs after deleting the Pod.

Optional settings:

- **`PLAYGROUND_REWRITER=1`** enables the BF16 Heretic scene rewriter. Set it to **`0`** to skip it and save download/storage space.
- **`HF_TOKEN`** accepts a read token from [Hugging Face settings](https://huggingface.co/settings/tokens), added through RunPod Secrets. Public downloads work without it.
- **`PLAYGROUND_KV_CACHE=0`** reduces inference cache memory if needed; the default is **`1`**.

No RunPod API key, GitHub token, or Docker Hub token is needed. To generate a replacement password locally:

```bash
openssl rand -hex 24
```

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
