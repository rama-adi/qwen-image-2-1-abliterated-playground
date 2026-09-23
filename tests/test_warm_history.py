import json
from pathlib import Path
import tempfile
import unittest
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import app
import inference_worker as worker


class WarmHistoryTests(unittest.TestCase):
    def test_pipeline_reused_until_settings_change(self):
        with patch.object(worker, 'PIPELINE', None), patch.object(worker, 'PIPELINE_KEY', None), patch.object(worker, 'release_pipeline'), patch.object(worker, 'load_pipeline', side_effect=[object(), object()]) as load:
            first = worker.get_pipeline(Path('/models'), {'offload': True})
            self.assertIs(first, worker.get_pipeline(Path('/models'), {'offload': True, 'seed': 99}))
            self.assertIsNot(first, worker.get_pipeline(Path('/models'), {'offload': False}))
            self.assertEqual(load.call_count, 2)

    def test_zero_seed_resolved_and_fixed_seed_preserved(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, 'BACKEND', 'diffusers'), patch.object(app.secrets, 'randbelow', return_value=123):
            for supplied, expected in [(0, 124), (42, 42)]:
                data = {'scene': 'boat', 'seed': supplied}
                cmd = app.make_command(data, None, Path(directory) / 'image.png')
                self.assertEqual(json.loads(Path(cmd[-1]).read_text())['seed'], expected)
                self.assertEqual(data['seed'], expected)

    def test_delete_one_all_and_active_protection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, 'OUTPUTS', Path(directory)), patch.object(app, 'ACTIVE', None):
            ids = ['a'*32, 'b'*32]
            for id in ids:
                target = Path(directory) / id
                target.mkdir(); (target / 'image.png').touch(); (target / 'run.log').touch()
            unrelated = Path(directory) / 'models'; unrelated.mkdir()
            unfinished = Path(directory) / ('c'*32); unfinished.mkdir()
            with patch.object(app, 'ACTIVE', ids[0]), self.assertRaises(ValueError):
                app.delete_history()
            with self.assertRaises(ValueError):
                app.delete_history('../models')
            self.assertEqual(app.delete_history(ids[0]), [ids[0]])
            self.assertEqual(app.delete_history(), [ids[1]])
            self.assertTrue(unrelated.exists()); self.assertTrue(unfinished.exists())

    def test_worker_process_reuse_failure_and_restart(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, 'ROOT', Path(directory)), patch.object(app, 'BACKEND', 'diffusers'), patch.dict(app.os.environ, {'PLAYGROUND_KEEP_WARM': '1'}):
            root = Path(directory)
            (root / 'inference_worker.py').write_text('''import json, sys, os
from pathlib import Path
for line in sys.stdin:
    request = json.loads(Path(json.loads(line)).read_text())
    if request.get('fail'): raise RuntimeError('test failure')
    Path(request['output']).write_text(str(os.getpid()))
    print('PLAYGROUND_EVENT {"stage":"sampling","step":1}', flush=True)
    print('PLAYGROUND_DONE 0', flush=True)
''')
            pids = []
            try:
                for i in range(4):
                    id = str(i); target = root / id; target.mkdir()
                    request = target / 'request.json'; output = target / 'image.png'
                    request.write_text(json.dumps({'output': str(output), 'fail': i == 2}))
                    job = {'directory': str(target), 'output': str(output), 'steps': 1}
                    with patch.dict(app.JOBS, {id: job}), patch.object(app, 'ACTIVE', id):
                        app.run_job(id, [app.sys.executable, '-u', str(root / 'inference_worker.py'), str(request)])
                    self.assertEqual(job['status'], 'error' if i == 2 else 'done')
                    if i != 2: pids.append(output.read_text())
                self.assertEqual(pids[0], pids[1]); self.assertNotEqual(pids[1], pids[2])
            finally:
                app.stop_warm_worker()

    def test_preview_does_not_chain_vae_to_transformer_offload(self):
        pipe = MagicMock()
        pipe._exclude_from_cpu_offload = []
        torch = MagicMock()
        diffusers = SimpleNamespace(QwenImage21Pipeline=MagicMock())
        diffusers.QwenImage21Pipeline.from_pretrained.return_value = pipe
        transformers = SimpleNamespace(Qwen3VLForConditionalGeneration=MagicMock())
        with patch.dict('sys.modules', {'torch': torch, 'diffusers': diffusers, 'transformers': transformers}), patch.object(worker, 'verify_checkpoint'), patch.object(worker, 'assert_bf16'), patch.dict(worker.os.environ, {'PLAYGROUND_VAE_TILING': '0'}):
            worker.load_pipeline(Path('/models'), {'offload': True, 'live_preview': True})
        self.assertEqual(pipe.model_cpu_offload_seq, 'text_encoder->transformer')
        self.assertIn('vae', pipe._exclude_from_cpu_offload)
        pipe.enable_model_cpu_offload.assert_called_once()
        pipe.vae.disable_tiling.assert_called_once()
        pipe.vae.enable_tiling.assert_not_called()
        pipe.reset_mock()
        with patch.dict('sys.modules', {'torch': torch, 'diffusers': diffusers, 'transformers': transformers}), patch.object(worker, 'verify_checkpoint'), patch.object(worker, 'assert_bf16'), patch.dict(worker.os.environ, {'PLAYGROUND_VAE_TILING': '1'}):
            worker.load_pipeline(Path('/models'), {'offload': False})
        pipe.vae.enable_tiling.assert_called_once()
        pipe.vae.disable_tiling.assert_not_called()

    def test_cancellation_terminates_warm_worker(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, 'ROOT', Path(directory)), patch.object(app, 'BACKEND', 'diffusers'), patch.dict(app.os.environ, {'PLAYGROUND_KEEP_WARM': '1'}):
            root = Path(directory)
            (root / 'inference_worker.py').write_text('import sys, time\nfor line in sys.stdin: time.sleep(60)\n')
            request = root / 'request.json'; request.write_text('{}')
            job = {'directory': str(root), 'output': str(root / 'image.png')}
            with patch.dict(app.JOBS, {'cancel-test': job}), patch.object(app, 'ACTIVE', 'cancel-test'):
                thread = threading.Thread(target=app.run_job, args=('cancel-test', [app.sys.executable, '-u', str(root / 'inference_worker.py'), str(request)]))
                thread.start()
                try:
                    deadline = time.monotonic() + 5
                    while 'process' not in job and time.monotonic() < deadline: time.sleep(.01)
                    self.assertIn('process', job)
                    job['cancelled'] = True; job['process'].terminate()
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(job['status'], 'cancelled')
                    self.assertIsNone(app.WARM_PROCESS)
                finally:
                    app.stop_warm_worker(); thread.join(timeout=5)

    def test_fragmented_reference_upload_reads_complete_json(self):
        payload = {'scene': 'boat', 'reference': 'data:image/png;base64,' + 'A' * 200000}
        raw = json.dumps(payload).encode()
        chunks = [raw[:184], raw[184:65000], raw[65000:130000], raw[130000:]]
        stream = MagicMock()
        stream.read.side_effect = chunks
        handler = SimpleNamespace(headers={'Content-Length': str(len(raw))}, rfile=stream)
        self.assertEqual(app.Handler.read_json(handler), payload)
        stream.read.side_effect = [raw[:184], b'']
        with self.assertRaisesRegex(ValueError, 'Upload ended'):
            app.Handler.read_json(handler)
