"""Reusable BF16 render worker. Cancellation/errors terminate its CUDA context."""
from adapter_compat import load_adapter
import gc
import json
import os
from pathlib import Path
import sys
import struct
import time

PIPELINE = None
PIPELINE_KEY = None

def release_pipeline():
    global PIPELINE, PIPELINE_KEY
    PIPELINE = None
    PIPELINE_KEY = None
    gc.collect()
    import torch
    torch.cuda.empty_cache()

def get_pipeline(root, request):
    global PIPELINE, PIPELINE_KEY
    lora = request.get("lora")
    adapter_key = (lora["id"], lora["strength"]) if lora and lora["strength"] != 0 else None
    key = (adapter_key, str(root), bool(request.get("offload", True)), bool(request.get("live_preview", True)), os.environ.get("PLAYGROUND_VAE_TILING", "0") == "1")
    if PIPELINE is not None and PIPELINE_KEY == key:
        print("Reusing warm BF16 image pipeline", flush=True)
        return PIPELINE
    release_pipeline()
    started = time.monotonic()
    PIPELINE = load_pipeline(root, request)
    print(f"Pipeline load completed in {time.monotonic() - started:.1f}s", flush=True)
    PIPELINE_KEY = key
    return PIPELINE



def emit_progress(**event):
    print("PLAYGROUND_EVENT " + json.dumps(event), flush=True)


def save_preview(pipe, packed_latents, request, step):
    """Decode target latents with the same BF16 VAE as the final image."""
    import torch
    latents = pipe._unpack_latents(packed_latents.detach(), request["height"], request["width"], pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    std = torch.tensor(pipe.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    decoded = pipe.vae.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    index = (step - 1) // request.get("preview_interval", 1)
    target = Path(request["output"]).parent / f"preview_{index:03d}.png"
    temporary = target.with_suffix(".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(target)


def verify_checkpoint(directory):
    """Check stored tensor dtypes before a loader can cast them to BF16."""
    files = sorted(Path(directory).glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors checkpoint found in {directory}")
    for file in files:
        with file.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > 100 * 1024 * 1024:
                raise ValueError(f"Invalid safetensors header in {file}")
            header = json.loads(stream.read(size))
        bad = {item["dtype"] for key, item in header.items()
               if key != "__metadata__" and item["dtype"] not in {"BF16", "F32", "I64", "BOOL"}}
        if bad:
            raise ValueError(f"{file.name} contains non-BF16/FP32 weights: {sorted(bad)}")


def parse_rewrite(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        raise ValueError("Rewriter exhausted its budget before producing an answer.")
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(result, dict) and isinstance(result.get("rewritten_prompt"), str) and result["rewritten_prompt"].strip():
            return result
    raise ValueError("Rewriter did not return valid rewritten_prompt JSON; original prompt was not silently replaced.")


def assert_bf16(model, name):
    import torch
    if getattr(model, "is_quantized", False) or getattr(model.config, "quantization_config", None):
        raise RuntimeError(f"{name} is quantized; this backend requires BF16 weights.")
    bad = {str(p.dtype) for p in model.parameters() if p.is_floating_point() and p.dtype not in {torch.bfloat16, torch.float32}}
    if bad:
        raise RuntimeError(f"{name} contains reduced-precision or unsupported parameters: {sorted(bad)}")
    print(f"Verified {name}: unquantized BF16 / architecture-required FP32 parameters", flush=True)


def rewrite_prompt(prompt, root):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessor, LogitsProcessorList

    class PresencePenalty(LogitsProcessor):
        def __init__(self, prompt_length):
            self.prompt_length = prompt_length

        def __call__(self, ids, scores):
            for row in range(ids.shape[0]):
                tokens = ids[row, self.prompt_length:]
                if tokens.numel():
                    scores[row, tokens.unique()] -= 1.5
            return scores

    checkpoint = root / "rewriter"
    verify_checkpoint(checkpoint)
    system_prompt = (checkpoint / "system_prompt.txt").read_text()
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    assert_bf16(model, "Heretic prompt rewriter")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=True).to("cuda")
    if "mm_token_type_ids" not in inputs and hasattr(processor, "create_mm_token_type_ids"):
        inputs["mm_token_type_ids"] = processor.create_mm_token_type_ids(inputs["input_ids"])
    length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=int(os.environ.get("PLAYGROUND_REWRITE_TOKENS", "6144")),
            do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
            logits_processor=LogitsProcessorList([PresencePenalty(length)]),
            pad_token_id=processor.tokenizer.eos_token_id)
    result = parse_rewrite(processor.tokenizer.decode(output[0, length:], skip_special_tokens=True))
    del inputs, output, model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return result


def load_pipeline(root, request):
    import torch
    from diffusers import QwenImage21Pipeline
    from transformers import Qwen3VLForConditionalGeneration
    print("Loading BF16 Heretic encoder and official BF16 DiT / VAE…", flush=True)
    emit_progress(stage="loading", step=0)
    for directory in (root / "encoder", root / "pipeline" / "transformer", root / "pipeline" / "vae"):
        verify_checkpoint(directory)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        root / "encoder", dtype=torch.bfloat16, local_files_only=True)
    pipe = QwenImage21Pipeline.from_pretrained(root / "pipeline", text_encoder=encoder,
        torch_dtype=torch.bfloat16, local_files_only=True)
    for name in ("text_encoder", "transformer", "vae"):
        assert_bf16(getattr(pipe, name), name)
    lora = request.get("lora")
    if lora and lora["strength"] != 0:
        print(f"Loading LoRA: {lora['name']} at strength {lora['strength']}", flush=True)
        try:
            load_adapter(pipe, lora["path"])
            pipe.set_adapters(["uploaded"], adapter_weights=[lora["strength"]])
            # Adapter files may be F16/F32; execution stays BF16 on RunPod.
            for name, parameter in pipe.transformer.named_parameters():
                if "lora_" in name:
                    parameter.data = parameter.data.to(dtype=torch.bfloat16)
            assert_bf16(pipe.transformer, "transformer with LoRA")
        except Exception as exc:
            raise RuntimeError("Could not load this LoRA. Use a standard Qwen Image 2.1 transformer LoRA compatible with Diffusers; older Qwen Image/Flux/SDXL adapters are not interchangeable. " + str(exc)) from exc
    tiled = os.environ.get("PLAYGROUND_VAE_TILING", "0") == "1"
    if tiled:
        pipe.vae.enable_tiling()
    else:
        pipe.vae.disable_tiling()
    print(f"VAE tiling: {'enabled' if tiled else 'disabled'} (BF16)", flush=True)
    if request.get("offload", True):
        if request.get("live_preview", True):
            # A VAE preview must not evict the transformer on every sampling step.
            pipe.model_cpu_offload_seq = "text_encoder->transformer"
            pipe._exclude_from_cpu_offload = [*pipe._exclude_from_cpu_offload, "vae"]
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe


def render(request):
    import torch
    from diffusers import QwenImage21Pipeline
    from transformers import Qwen3VLForConditionalGeneration
    from PIL import Image

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("RunPod requires an NVIDIA GPU with native BF16 support (Ampere or newer).")
    root = Path(os.environ.get("PLAYGROUND_MODEL_DIR", "/workspace/models"))
    output = Path(request["output"])
    prompt = request["prompt"]
    seed = request["seed"]
    torch.manual_seed(seed)
    if request.get("rewrite"):
        if request.get("input_mode") == "edit":
            raise ValueError("BF16 PE-T2I rewrites new scenes only. Disable rewriting for image edits.")
        release_pipeline()
        print("Rewriting prompt with BF16 Heretic PE-T2I…", flush=True)
        emit_progress(stage="rewriting", step=0)
        result = rewrite_prompt(prompt, root)
        (output.parent / "rewrite.json").write_text(json.dumps(result, indent=2))
        prompt = result["rewritten_prompt"]
        # The user's chosen canvas wins over the rewriter's aspect-ratio suggestion.
    metadata_path = output.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(prompt=prompt, seed=seed, precision="bfloat16", backend="diffusers",
                    vae_tiling=os.environ.get("PLAYGROUND_VAE_TILING", "0") == "1")
    metadata["model_revisions"] = json.loads((Path(__file__).parent / "models.lock.json").read_text())
    metadata_path.write_text(json.dumps(metadata, indent=2))

    pipe = get_pipeline(root, request)
    torch.cuda.reset_peak_memory_stats()
    print("Encoding prompt and running first sampling step…", flush=True)
    emit_progress(stage="encoding", step=0)
    sampling_started = time.monotonic()

    def progress(_pipe, step, _timestep, tensors):
        current = step + 1
        print(f"Sampling {step + 1}/{request['steps']} · {time.monotonic() - sampling_started:.1f}s since encoding started", flush=True)
        emit_progress(stage="sampling", step=current)
        if request.get("live_preview", True) and (current % request.get("preview_interval", 1) == 0 or current == request["steps"]):
            emit_progress(stage="preview", step=current)
            save_preview(_pipe, tensors["latents"], request, current)
            emit_progress(stage="sampling", step=current, preview_step=current)
        if current == request["steps"]:
            emit_progress(stage="decoding", step=current)
        return tensors

    image = Image.open(request["reference"]).convert("RGBA") if request.get("reference") else None
    result = pipe(prompt=prompt, image=image, negative_prompt=request.get("negative", ""),
        true_cfg_scale=request["cfg"], width=request["width"], height=request["height"],
        num_inference_steps=request["steps"], generator=torch.Generator("cuda").manual_seed(seed),
        callback_on_step_end=progress, use_kv_cache=os.environ.get("PLAYGROUND_KV_CACHE", "1") == "1")
    result.images[0].save(output)
    print(f"Saved {output}; peak CUDA allocation: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


def serve():
    # One request at a time. An exception exits the worker, releasing all GPU state.
    for line in sys.stdin:
        request_path = json.loads(line)
        render(json.loads(Path(request_path).read_text()))
        print("PLAYGROUND_DONE 0", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--serve":
        serve()
    else:
        render(json.loads(Path(sys.argv[1]).read_text()))
