"""Optional APG -> FreSca guidance in denoised space for Qwen 2.1 flow matching.

Equations follow ComfyUI's nodes_apg.py and nodes_fresca.py pre-CFG ordering.
Reference: https://github.com/Comfy-Org/ComfyUI/tree/master/comfy_extras
The pinned Diffusers pipeline emits cond then uncond; hooks never alter its model.
"""
from contextlib import contextmanager
import math

DEFAULTS = {'apg': False, 'apg_eta': 1.0, 'apg_norm': 10.0, 'apg_momentum': 0.3,
            'fresca': False, 'fresca_low': 1.0, 'fresca_high': 2.0, 'fresca_cutoff': 8}


def settings(data, backend='diffusers', cfg=3):
    result = {}
    for name in ('apg', 'fresca'):
        value = data.get(name, False)
        if not isinstance(value, bool): raise ValueError(f'{name} must be true or false.')
        result[name] = value
    ranges = {'apg_eta':(-10,10), 'apg_norm':(0,50), 'apg_momentum':(-5,1),
              'fresca_low':(0,10), 'fresca_high':(0,10), 'fresca_cutoff':(1,10000)}
    for name, (low, high) in ranges.items():
        try: value = float(data.get(name, DEFAULTS[name]))
        except (ValueError, TypeError): raise ValueError(f'Invalid {name}.') from None
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} must be between {low} and {high}.')
        if name == 'fresca_cutoff':
            if not value.is_integer(): raise ValueError('FreSca cutoff must be an integer.')
            value = int(value)
        result[name] = value
    if result['apg'] or result['fresca']:
        if backend != 'diffusers': raise ValueError('APG and FreSca currently require RunPod / Diffusers.')
        if cfg <= 1: raise ValueError('APG and FreSca require CFG greater than 1.')
    return result


class Guidance:
    def __init__(self, config):
        self.config = config
        self.momentum = None
        self.previous_sigma = None

    def combine(self, cond, uncond, scale, sigma):
        """B,C,H,W denoised predictions -> guided denoised prediction (FP32)."""
        import torch
        cond, uncond = cond.float(), uncond.float()
        cfg = self.config
        delta = cond - uncond
        if cfg['apg']:
            if self.previous_sigma is not None and sigma > self.previous_sigma:
                self.momentum = None
            self.previous_sigma = sigma
            direction = delta
            if cfg['apg_momentum']:
                self.momentum = direction if self.momentum is None else direction + cfg['apg_momentum'] * self.momentum
                direction = self.momentum
            axes = tuple(range(1, cond.ndim))
            if cfg['apg_norm'] > 0:
                norm = torch.linalg.vector_norm(direction, dim=axes, keepdim=True)
                direction = direction * (cfg['apg_norm'] / norm.clamp_min(1e-12)).clamp(max=1)
            unit = cond / torch.linalg.vector_norm(cond, dim=axes, keepdim=True).clamp_min(1e-12)
            parallel = (direction * unit).sum(dim=axes, keepdim=True) * unit
            # Comfy's pre-CFG APG adds the original delta / CFG before normal CFG.
            delta = direction + (cfg['apg_eta'] - 1) * parallel + delta / scale
        if cfg['fresca']:
            height, width = delta.shape[-2:]
            cy, cx = height // 2, width // 2
            ry, rx = min(cfg['fresca_cutoff'], cy), min(cfg['fresca_cutoff'], cx)
            mask = torch.full((height,width), cfg['fresca_high'], device=delta.device, dtype=torch.float32)
            mask[cy-ry:cy+ry, cx-rx:cx+rx] = cfg['fresca_low']
            spectrum = torch.fft.fftshift(torch.fft.fft2(delta), dim=(-2,-1))
            delta = torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask, dim=(-2,-1))).real
        return uncond + scale * delta


@contextmanager
def guided_sampling(pipe, request):
    config = settings(request, cfg=request['cfg'])
    if not (config['apg'] or config['fresca']):
        yield
        return
    from diffusers import FlowMatchEulerDiscreteScheduler
    if not isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler):
        raise ValueError('APG/FreSca integration requires the flow-matching Euler scheduler.')
    controller = Guidance(config)
    outputs = []
    height = 2 * (request['height'] // (pipe.vae_scale_factor * 2))
    width = 2 * (request['width'] // (pipe.vae_scale_factor * 2))
    count = height * width
    def capture(module, args, output):
        if not isinstance(output, tuple) or len(outputs) >= 2:
            raise RuntimeError('Unexpected Qwen transformer output/order for guidance.')
        outputs.append(output[0][:, -count:].detach())
    scheduler = pipe.scheduler
    original = scheduler.step
    had_override = 'step' in scheduler.__dict__
    previous_override = scheduler.__dict__.get('step')
    def step(model_output, timestep, sample, *args, **kwargs):
        if len(outputs) != 2 or sample.shape[1] != count:
            raise RuntimeError('APG/FreSca expected conditional and unconditional predictions for the target image.')
        if scheduler.step_index is None: scheduler._init_step_index(timestep)
        sigma = float(scheduler.sigmas[scheduler.step_index])
        if sigma <= 0: raise RuntimeError('Guidance requires a positive sampling sigma.')
        cond_velocity, uncond_velocity = outputs
        outputs.clear()
        def spatial(tensor): return tensor.transpose(1,2).reshape(tensor.shape[0],tensor.shape[2],height,width)
        x = spatial(sample.float())
        cond = x - sigma * spatial(cond_velocity.float())
        uncond = x - sigma * spatial(uncond_velocity.float())
        denoised = controller.combine(cond, uncond, request['cfg'], sigma)
        velocity = ((x - denoised) / sigma).flatten(2).transpose(1,2).to(model_output.dtype)
        return original(velocity, timestep, sample, *args, **kwargs)
    hook = pipe.transformer.register_forward_hook(capture)
    scheduler.step = step
    try:
        yield
    finally:
        hook.remove()
        outputs.clear()
        if had_override: scheduler.step = previous_override
        else: del scheduler.step
