"""Translate ComfyUI Qwen 2.1 DoRA tensors to Diffusers/PEFT without dropping scales."""

def convert_comfy_dora(state):
    if not any(key.endswith('.dora_scale') for key in state):
        raise ValueError('Expected a ComfyUI DoRA adapter.')
    groups = {}
    suffixes = ('.lora_A.weight', '.lora_B.weight', '.dora_scale')
    for key, value in state.items():
        suffix = next((suffix for suffix in suffixes if key.endswith(suffix)), None)
        if suffix is None or not key.startswith('diffusion_model.transformer_blocks.'):
            raise ValueError(f'Unsupported ComfyUI DoRA tensor: {key}')
        module = key[len('diffusion_model.'):-len(suffix)]
        groups.setdefault(module, {})[suffix] = value
    result = {}
    for module, tensors in groups.items():
        if set(tensors) != set(suffixes):
            raise ValueError(f'Incomplete DoRA A/B/magnitude tensors: {module}')
        a, b, magnitude = (tensors[suffix] for suffix in suffixes)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or magnitude.numel() != b.shape[0]:
            raise ValueError(f'Invalid DoRA tensor dimensions: {module}')
        magnitude = magnitude.reshape(-1)
        if module.endswith('.img_mlp.gate_up'):
            if b.shape[0] % 2:
                raise ValueError('Fused gate/up output must have an even dimension.')
            # ComfyUI packs [gate; up]; Diffusers calls up `proj`.
            for index, name in enumerate(('gate_layer', 'proj')):
                target = module.removesuffix('gate_up') + name
                result[f'transformer.{target}.lora_A.weight'] = a
                result[f'transformer.{target}.lora_B.weight'] = b.chunk(2, dim=0)[index].contiguous()
                result[f'transformer.{target}.lora_magnitude_vector'] = magnitude.chunk(2, dim=0)[index].contiguous()
        else:
            result[f'transformer.{module}.lora_A.weight'] = a
            result[f'transformer.{module}.lora_B.weight'] = b
            result[f'transformer.{module}.lora_magnitude_vector'] = magnitude
    return result


def load_adapter(pipe, path):
    from safetensors import safe_open
    with safe_open(path, framework='pt', device='cpu') as file:
        is_dora = any(key.endswith('.dora_scale') for key in file.keys())
    if not is_dora:
        pipe.load_lora_weights(path, adapter_name='uploaded', local_files_only=True, use_safetensors=True)
        return
    from safetensors.torch import load_file
    state = convert_comfy_dora(load_file(path, device='cpu'))
    # Verify every target before PEFT can warn and ignore unexpected keys.
    modules = dict(pipe.transformer.named_modules())
    for key, a in state.items():
        if not key.endswith('.lora_A.weight'):
            continue
        target = key.removeprefix('transformer.').removesuffix('.lora_A.weight')
        weight = getattr(modules.get(target), 'weight', None)
        b = state[key.replace('.lora_A.weight', '.lora_B.weight')]
        if weight is None or tuple(weight.shape) != (b.shape[0], a.shape[1]):
            raise ValueError(f'DoRA target does not match Qwen Image 2.1: {target}')
    print(f'Loading ComfyUI DoRA: {len(state)//3} target layers, preserving all magnitude tensors', flush=True)
    pipe.load_lora_weights(state, adapter_name='uploaded')
    if not pipe.transformer.peft_config['uploaded'].use_dora:
        raise RuntimeError('DoRA was not enabled; refusing to ignore magnitude tensors.')
