"""CPU checks for denoised guidance and the pinned pipeline's scheduler adapter."""
import importlib.util
from types import SimpleNamespace
import unittest
from guidance import Guidance, settings, guided_sampling

class GuidanceValidationTests(unittest.TestCase):
    def test_invalid_settings_and_backend_rejected(self):
        for data in ({'apg_eta':float('nan')}, {'apg': 'false'}, {'fresca_cutoff':2.5}, {'fresca_high':11}):
            with self.assertRaises(ValueError): settings(data)
        with self.assertRaises(ValueError): settings({'apg':True}, backend='sd-cpp')
        with self.assertRaises(ValueError): settings({'fresca':True}, cfg=1)
        self.assertFalse(settings({})['apg'])

@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('diffusers'), 'Requires inference dependencies')
class GuidanceMathTests(unittest.TestCase):
    def test_apg_projection_norm_and_momentum(self):
        import torch
        cond=torch.tensor([[[[0., 4.]]]])
        uncond=torch.tensor([[[[-3., 0.]]]])
        control=Guidance(settings({'apg':True,'apg_eta':0,'apg_norm':2,'apg_momentum':.3}))
        # Delta [3,4] clips to [1.2,1.6]; projecting off cond leaves [1.2,0].
        # APG pre-CFG also adds delta / CFG, yielding cond + CFG * [1.2,0].
        torch.testing.assert_close(control.combine(cond,uncond,3,1),torch.tensor([[[[3.6,4.]]]]))
        torch.testing.assert_close(control.momentum,torch.tensor([[[[3.,4.]]]]))
        control.combine(cond,uncond,3,.5)
        torch.testing.assert_close(control.momentum,torch.tensor([[[[3.9,5.2]]]]))
        control.combine(cond,uncond,3,1)
        torch.testing.assert_close(control.momentum,torch.tensor([[[[3.,4.]]]]))
        zero=torch.zeros_like(cond)
        self.assertTrue(torch.isfinite(control.combine(zero,zero,3,.5)).all())

    def test_fresca_scales_spatial_low_and_high_frequencies(self):
        import torch
        control=Guidance(settings({'fresca':True,'fresca_low':1,'fresca_high':2,'fresca_cutoff':1}))
        y,x=torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
        checker=((x+y)%2*2-1).float()[None,None]
        cond=3+checker
        result=control.combine(cond,torch.zeros_like(cond),3,1)
        torch.testing.assert_close(result,3*(3+2*checker))
        # Batch items/channels must not be mixed by the spatial FFT.
        multi=cond.repeat(2,3,1,1)
        torch.testing.assert_close(control.combine(multi,multi*0,3,1),result.repeat(2,3,1,1))

    def test_adapter_slices_reference_tokens_and_restores_hooks_after_error(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        class Transformer(torch.nn.Module):
            def forward(self,value): return (value,)
        scheduler=FlowMatchEulerDiscreteScheduler(); scheduler.set_timesteps(2)
        pipe=SimpleNamespace(transformer=Transformer(),scheduler=scheduler,vae_scale_factor=16)
        request={'cfg':3,'height':32,'width':32,'fresca':True,'fresca_low':1,'fresca_high':1}
        sample=torch.ones(1,4,2)
        cond=torch.ones(1,4,2)*.2; uncond=torch.ones(1,4,2)*.1
        baseline=FlowMatchEulerDiscreteScheduler();baseline.set_timesteps(2)
        expected=baseline.step(uncond+3*(cond-uncond),baseline.timesteps[0],sample,return_dict=False)[0]
        original=scheduler.step
        with self.assertRaisesRegex(RuntimeError,'test cleanup'):
            with guided_sampling(pipe,request):
                pipe.transformer(torch.cat((torch.full((1,2,2),99.),cond),dim=1))
                pipe.transformer(torch.cat((torch.full((1,2,2),99.),uncond),dim=1))
                result=scheduler.step(uncond+3*(cond-uncond),scheduler.timesteps[0],sample,return_dict=False)[0]
                torch.testing.assert_close(result,expected)
                raise RuntimeError('test cleanup')
        self.assertEqual(scheduler.step,original)
        self.assertFalse(pipe.transformer._forward_hooks)
        with guided_sampling(pipe,{'cfg':1}):
            self.assertEqual(scheduler.step,original)

    def test_seeds_matches_two_stage_flow_equations_and_resets(self):
        import math
        import torch
        from seeds_scheduler import Seeds2Scheduler
        # Independent spatial reference for Comfy's SEEDS-2 phi_1, eta=1, r=.5.
        for count in (1, 4, 20):
            scheduler = Seeds2Scheduler(seed=79)
            scheduler.set_timesteps(count)
            spatial = torch.linspace(-1, 1, 48).reshape(1,3,4,4)
            reference = spatial.clone()
            rng = torch.Generator().manual_seed(80)
            shift = math.exp(.69)
            sigma_min = shift / (shift + 10000 - 1)
            times = torch.linspace(1, sigma_min, count+1)[:-1]
            sigmas = shift / (shift + 1/times - 1)
            sigmas[0] = shift / (shift + 1/.9999 - 1)
            sigmas = torch.cat((sigmas, torch.zeros(1)))
            def denoiser(x, sigma): return .2*x + .1*sigma
            for start, end in zip(sigmas[:-1], sigmas[1:]):
                clean = denoiser(reference, start)
                if end == 0:
                    reference = clean
                    continue
                ls, lt = -torch.logit(start), -torch.logit(end)
                h = lt-ls
                lm = torch.lerp(ls,lt,.5)
                mid = torch.sigmoid(-lm)
                first_noise = (-h).expm1().neg().sqrt()*torch.randn(reference.shape, generator=rng)
                midpoint = mid/start*(-.5*h).exp()*reference - mid*lm.exp()*(-h).expm1()*clean + mid*first_noise
                clean_mid = denoiser(midpoint,mid)
                noise = first_noise*(-.5*h).exp() + (-h).expm1().neg().sqrt()*torch.randn(reference.shape,generator=rng)
                reference = end/start*(-h).exp()*reference - end*lt.exp()*(-2*h).expm1()*clean_mid + end*noise
            def run():
                x = spatial.flatten(2).transpose(1,2).clone()
                for timestep in scheduler.timesteps:
                    sigma = timestep/1000
                    velocity = (x-denoiser(x,sigma))/sigma
                    x = scheduler.step(velocity,timestep,x,return_dict=False)[0]
                self.assertEqual(scheduler.completed_steps,count)
                return x.transpose(1,2).reshape_as(spatial)
            actual=run()
            torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
            self.assertEqual(len(scheduler.timesteps),2*count-1)
            scheduler.set_timesteps(sigmas=[1]*count)
            torch.testing.assert_close(run(),actual,atol=0,rtol=0)

    def test_workflow_expansion_template_and_seed_isolation(self):
        import torch
        from unittest.mock import patch
        from workflow_expand import expand, CONFIG, PREFIX
        seen = {}
        class Inputs(dict):
            @property
            def input_ids(self): return self['input_ids']
            def to(self, device):
                seen['device'] = device
                return self
        class Tokenizer:
            def __call__(self, text, **kwargs):
                seen['text'] = text
                return Inputs(input_ids=torch.tensor([[1,2,3]]))
            def decode(self, tokens, **kwargs):
                self_tokens=tokens.tolist()
                assert self_tokens == [4,5]
                return 'Expanded scene'
        class Encoder:
            def generate(self, **kwargs):
                seen['config'] = {k:v for k,v in kwargs.items() if k != 'input_ids'}
                seen['seed'] = torch.initial_seed()
                torch.rand(3)
                return torch.tensor([[1,2,3,4,5]])
        pipe=SimpleNamespace(processor=SimpleNamespace(tokenizer=Tokenizer()),text_encoder=Encoder())
        before=torch.random.get_rng_state().clone()
        original_fork=torch.random.fork_rng
        with patch.object(torch.cuda,'current_device',return_value=0), patch.object(torch.random,'fork_rng',side_effect=lambda **kw: original_fork(devices=[])):
            self.assertEqual(expand(pipe,'My scene'),'Expanded scene')
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertEqual(seen['seed'],0)
        self.assertEqual(seen['config'],CONFIG)
        self.assertIn(PREFIX+'My scene',seen['text'])
        self.assertTrue(seen['text'].endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
