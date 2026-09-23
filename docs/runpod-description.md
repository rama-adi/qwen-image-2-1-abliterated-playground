# Qwen Image 2.1 Uncensored Quickstart

Generate and edit images in your browser with **Qwen Image 2.1 and an unquantized BF16 Heretic text encoder**. Includes an optional BF16 Heretic prompt rewriter, per-step image previews and progress, character descriptions, style references, and saved render history.

## About this template

This template runs [qwen-image-2-1-abliterated-playground](https://github.com/rama-adi/qwen-image-2-1-abliterated-playground), an experimental, single-user image playground. It combines the official Qwen Image 2.1 BF16 diffusion model and VAE with the Heretic BF16 text encoder. The optional Heretic PE-T2I model expands short scene descriptions before rendering. All RunPod model weights are unquantized; there is no Q4, INT8, FP8, or NVFP4 fallback. Heretic changes the language components, while the diffusion model and VAE retain their official weights. This setup has no additional application-level content filter. “Uncensored” describes the reduced-refusal setup, not a guarantee that every prompt will behave as intended. Users are responsible for their inputs and outputs.

## Before you deploy

- Choose an NVIDIA GPU with native BF16 support and preferably **48 GB VRAM or more**: RTX A6000, L40/L40S, or A100 80 GB. Use a host driver compatible with **CUDA 12.8**.
- Choose **64 GB host RAM or more**, **30 GB container disk**, and a **120 GB persistent volume** mounted at `/workspace`.
- Expose **HTTP port 8765**. Leave the container start command blank.
- Set **`PLAYGROUND_PASSWORD`** to a unique password. Use RunPod's **key icon / Secrets** for this value. You can generate one locally:

```bash
openssl rand -hex 24
```

- Optionally set **`HF_TOKEN`** using a RunPod Secret. Get a read token from [Hugging Face access-token settings](https://huggingface.co/settings/tokens). Public downloads normally work without it.
- Leave **`PLAYGROUND_REWRITER=1`** to download the BF16 scene rewriter. Set it to `0` if you do not need rewriting.

No RunPod API key, Docker Hub token, or GitHub token is required inside the Pod. The image is:

```text
ghcr.io/rama-adi/qwen-image-2-1-abliterated-playground:latest
```

The GHCR package must be public for anonymous pulls; if kept private, attach a RunPod registry credential with GitHub username and a classic PAT with `read:packages`.

## How to use

1. Deploy the Pod and watch its startup logs. First boot downloads approximately **50 GB of weights** with the rewriter enabled. Allow time for this download; subsequent starts reuse the persistent volume.
2. In **Connect**, open the HTTP service on **port 8765**.
3. Sign in with username **`playground`** and your **`PLAYGROUND_PASSWORD`**.
4. Enter a scene, optionally add character descriptions or a style reference, and click **Generate image**.
5. Start at **1024×1024, 40 steps, CFG 1** with **Keep model weights in RAM** enabled. For a quick loading test, use **512×512 and 1 step**; a one-step result is not representative of final quality.
6. Enable **Expand scene with Heretic** when you want automatic scene rewriting. For image edits, select **Edit / reprompt image** and provide the change you want; scene rewriting is disabled for this mode.
7. Watch the image preview and generation log for sampling-step updates. Progress streams through the same HTTP port using WebSockets, with reconnection and an HTTP fallback. Per-step VAE previews add render time; increase the preview interval or disable previews for speed.

Images, prompts, seeds, and render logs are stored in `/workspace/outputs`. Model files live in `/workspace/models`. Use persistent storage to retain them when replacing a Pod. Network volumes survive Pod deletion; ordinary Pod volume storage does not.

## Common issues

**The browser service is not ready:** First-boot model downloads must finish before the UI starts. Check the Pod logs for download progress or an authentication error.

**The container exits immediately:** Confirm that `PLAYGROUND_PASSWORD` is set and a compatible CUDA GPU is attached.

**Out of memory:** Keep model offloading enabled, lower the resolution, increase the preview interval, or disable live previews. You can also set `PLAYGROUND_KV_CACHE=0` and restart. The application does not switch to quantized weights.

**Slow rendering:** CPU offloading and frequent VAE previews add overhead. Disable previews for a speed comparison. Keep the rewriter disabled in the UI when you do not need it.

**Download errors:** Check persistent disk space. Add an optional Hugging Face read token as `HF_TOKEN` and restart; the downloader reuses its cache.

**Progress disconnects:** Keep port 8765 exposed as HTTP. The UI reconnects automatically and falls back to HTTP polling if WebSockets cannot connect. You can refresh the page to reconnect to an active render.

**Need a GPU smoke test?** From a local checkout, or a terminal with the project available:

```bash
PLAYGROUND_URL=https://POD_ID-8765.proxy.runpod.net \
PLAYGROUND_PASSWORD='your-password' \
python3 scripts/smoke_gpu.py
```

This tests text-to-image, editing, and enabled scene rewriting on your running Pod. It uses GPU time and saves its test renders in history. For downloader options:

```bash
python /app/download_models.py --help
```

## Contributors and support

- Maintainer: [@rama-adi](https://github.com/rama-adi)
- Questions and bug reports: [GitHub Issues](https://github.com/rama-adi/qwen-image-2-1-abliterated-playground/issues)
- Upstream models: [Qwen](https://github.com/QwenLM/Qwen-Image-2.1) and [pottokao's Heretic checkpoints](https://huggingface.co/pottokao)

The image model and prompt rewriter use the **Qwen Research License**, intended for non-commercial research/evaluation. The Heretic text encoder derives from Apache-2.0 Qwen3-VL. Check each checkpoint's license before other uses.

## Version history

- **0.1 — Initial release**
  - BF16 RunPod image generation and editing with a Heretic encoder.
  - Optional BF16 Heretic scene rewriting.
  - Per-step previews and streaming progress, persistent history, and password-protected access.
  - Separate quantized GGUF / Metal setup for Mac in the project repository.
