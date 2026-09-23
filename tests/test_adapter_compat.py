"""CPU-only numerical regression; runs in the inference image with torch/PEFT."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('diffusers') is not None

@unittest.skipUnless(HAS_TORCH, 'Requires inference dependencies')
class DoraTests(unittest.TestCase):
    def test_fused_dora_loads_and_matches_normalized_weight(self):
        import torch
        from diffusers import QwenImage21Transformer2DModel
        from diffusers.loaders import QwenImageLoraLoaderMixin
        from safetensors.torch import save_file
        from adapter_compat import load_adapter, convert_comfy_dora
        torch.manual_seed(1)
        model = QwenImage21Transformer2DModel(num_layers=1, attention_head_dim=8,
            num_attention_heads=1, context_in_dim=8, in_channels=4, out_channels=4,
            mlp_ratio=2, axes_dims_rope=(2,2,4))
        class Pipe(QwenImageLoraLoaderMixin):
            hf_device_map = None
            def __init__(self): self.transformer = model
            @property
            def components(self): return {'transformer': self.transformer}
        pipe = Pipe()
        mlp = model.transformer_blocks[0].img_mlp
        base = torch.cat((mlp.gate_layer.weight.detach().clone(), mlp.proj.weight.detach().clone()))
        a, b = torch.randn(2,8)*.1, torch.randn(32,2)*.1
        magnitude = torch.rand(32,1)+.5
        prefix = 'diffusion_model.transformer_blocks.0.img_mlp.gate_up'
        source = {prefix+'.lora_A.weight':a, prefix+'.lora_B.weight':b, prefix+'.dora_scale':magnitude}
        converted = convert_comfy_dora(source)
        self.assertEqual(len(converted), 6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fix.safetensors'; save_file(source, str(path))
            load_adapter(pipe, str(path))
        self.assertTrue(model.peft_config['uploaded'].use_dora)
        expected_weight = base+b@a
        expected_weight = expected_weight/expected_weight.norm(dim=1,keepdim=True)*magnitude
        x = torch.randn(3,8)
        actual = torch.cat((mlp.gate_layer(x), mlp.proj(x)), dim=-1)
        torch.testing.assert_close(actual, x@expected_weight.T, atol=1e-5, rtol=1e-5)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            convert_comfy_dora({prefix+'.lora_A.weight':a, prefix+'.dora_scale':magnitude})
