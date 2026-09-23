"""SEEDS-2 (phi_1, eta=1, r=0.5), flow matching with Comfy's sgm_uniform schedule.

Method: https://arxiv.org/abs/2305.14267
Reference configuration: ComfyUI sample_seeds_2 and QwenImage21 sampling shift 0.69.
Each nonfinal logical step evaluates the denoiser twice; the last evaluates once.
"""
import math
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput


class Seeds2Scheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, seed=79):
        super().__init__()
        self.seed = seed
        self.completed_steps = 0
        self.noise_generator = None

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None, **kwargs):
        count = num_inference_steps or len(sigmas)
        shift = math.exp(.69)
        def shifted(t): return shift / (shift + 1/t - 1)
        minimum = shifted(torch.tensor(.0001, dtype=torch.float32))
        times = torch.linspace(1, minimum, count+1)[:-1]
        schedule = shifted(times)
        schedule[0] = shifted(torch.tensor(.9999))  # finite initial log-SNR
        self.schedule = torch.cat((schedule, torch.zeros(1)))
        evaluation = []
        self.stages = []
        for i in range(count):
            start, end = self.schedule[i:i+2]
            evaluation.append(start); self.stages.append((i, 0))
            if end > 0:
                midpoint = torch.sigmoid((torch.logit(start) + torch.logit(end)) / 2)
                evaluation.append(midpoint); self.stages.append((i, 1))
        self.sigmas = torch.cat((torch.stack(evaluation), torch.zeros(1)))
        self.timesteps = (self.sigmas[:-1] * 1000).to(device=device)
        self.num_inference_steps = count
        self._step_index = None
        self._begin_index = None
        self.completed_steps = 0
        self.noise_generator = None
        self.saved = None

    def step(self, model_output, timestep, sample, return_dict=True, **kwargs):
        if self.step_index is None: self._init_step_index(timestep)
        i, stage = self.stages[self.step_index]
        sigma = self.sigmas[self.step_index].to(sample.device)
        start, end = self.schedule[i:i+2].to(sample.device)
        x = sample.float()
        denoised = x - sigma * model_output.float()
        if self.noise_generator is None:
            self.noise_generator = torch.Generator(device=sample.device).manual_seed(self.seed + (1 if sample.device.type == 'cpu' else 0))
        # Random values are sampled in spatial layout, as in ComfyUI.
        def noise():
            batch, tokens, channels = sample.shape
            value = torch.randn((batch,channels,tokens), generator=self.noise_generator, device=sample.device, dtype=x.dtype)
            return value.transpose(1,2)
        if end == 0:
            next_sample = denoised
            self.completed_steps = i+1
        else:
            h = torch.logit(start) - torch.logit(end)
            middle = torch.sigmoid((torch.logit(start) + torch.logit(end)) / 2)
            if stage == 0:
                first_noise = (-torch.expm1(-h)).sqrt() * noise()
                intermediate = middle/start * torch.exp(-h/2)*x - (1-middle)*torch.expm1(-h)*denoised
                next_sample = intermediate + middle*first_noise
                self.saved = (x, first_noise)
            else:
                previous, first_noise = self.saved
                final_noise = first_noise*torch.exp(-h/2) + (-torch.expm1(-h)).sqrt()*noise()
                next_sample = end/start*torch.exp(-h)*previous - (1-end)*torch.expm1(-2*h)*denoised + end*final_noise
                self.saved = None
                self.completed_steps = i+1
        self._step_index += 1
        next_sample = next_sample.to(sample.dtype)
        if not return_dict: return (next_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=next_sample)
