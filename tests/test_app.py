import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import app
from inference_worker import parse_rewrite


class BackendTests(unittest.TestCase):
    def test_bf16_command_ignores_client_model_path_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, 'BACKEND', 'diffusers'):
            output = Path(directory) / 'image.png'
            command = app.make_command({'scene': 'A red sailboat', 'width': 512, 'height': 512,
                'seed': 123, 'offload': True, 'settings': {'sd_cli': '/untrusted/tool',
                'encoder': '/models/quantized.gguf'}}, None, output)
            self.assertEqual(command[2], str(app.ROOT / 'inference_worker.py'))
            request = json.loads(Path(command[-1]).read_text())
            self.assertNotIn('settings', request)
            self.assertEqual(request['seed'], 123)
            self.assertIn('A red sailboat', request['prompt'])
            self.assertTrue(request['offload'])

    def test_invalid_dimensions_and_seed_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, 'BACKEND', 'diffusers'):
            for extra in ({'width': 513}, {'seed': -1}, {'cfg': float('nan')}):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    app.make_command({'scene': 'A boat', **extra}, None, Path(directory) / 'image.png')

    def test_rewriter_cannot_silently_process_edit_or_disabled_configuration(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, 'BACKEND', 'diffusers'):
            for enabled, mode in [('0', 'style'), ('1', 'edit')]:
                with patch.dict(os.environ, {'PLAYGROUND_REWRITER': enabled}), self.assertRaises(ValueError):
                    app.make_command({'scene': 'A boat', 'rewrite': True, 'input_mode': mode},
                                     Path(directory) / 'reference.png', Path(directory) / 'image.png')

    def test_mac_command_uses_local_gguf_and_projector(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, 'BACKEND', 'sd-cpp'):
            settings = {}
            for key in ('sd_cli', 'diffusion', 'encoder', 'vision', 'vae'):
                file = Path(directory) / (key + '.gguf')
                file.touch()
                settings[key] = str(file)
            command = app.make_command({'scene': 'A boat', 'settings': settings, 'seed': 17},
                Path(directory) / 'reference.png', Path(directory) / 'image.png')
            self.assertIn('--llm_vision', command)
            self.assertEqual(command[command.index('--seed') + 1], '17')

    def test_rewriter_parsing_removes_reasoning_and_rejects_malformed_output(self):
        self.assertEqual(parse_rewrite('reasoning {"rewritten_prompt":"wrong"}</think>```json\n{"rewritten_prompt":"A boat {at sea}","wh_ratio":"1:1"}\n```')['rewritten_prompt'], 'A boat {at sea}')
        for text in ('<think>unfinished', '{"rewritten_prompt":""}', 'No JSON here'):
            with self.assertRaises(ValueError):
                parse_rewrite(text)

    def test_worker_failure_and_cancel_release_generation_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            for code, cancelled in [(1, False), (0, True)]:
                job = {'directory': directory, 'output': str(Path(directory) / 'missing.png'),
                       'cancelled': cancelled}
                with patch.dict(app.JOBS, {'test': job}), patch.object(app, 'ACTIVE', 'test'):
                    app.run_job('test', [os.sys.executable, '-c', f'raise SystemExit({code})'])
                    self.assertEqual(job['status'], 'cancelled' if cancelled else 'error')
                    self.assertIsNone(app.ACTIVE)
                    self.assertNotIn('process', job)


if __name__ == '__main__':
    unittest.main()
