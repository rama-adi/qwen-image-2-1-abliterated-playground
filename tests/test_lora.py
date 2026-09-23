import base64
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import app
import inference_worker as worker
from lora_store import Store, MAX_SIZE


def adapter_bytes(name='transformer.test.lora_A.weight'):
    header = json.dumps({name: {'dtype': 'BF16', 'shape': [1, 2], 'data_offsets': [0, 4]}}).encode()
    header += b' ' * (-len(header) % 8)
    return struct.pack('<Q', len(header)) + header + b'\0' * 4


class LoraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / 'loras')

    def upload(self, raw=None):
        raw = adapter_bytes() if raw is None else raw
        id = self.store.begin('fix.safetensors', len(raw))['id']
        self.store.append(id, 0, raw)
        return self.store.finish(id)

    def test_chunked_upload_selection_and_disable(self):
        raw = adapter_bytes()
        id = self.store.begin('fix.safetensors', len(raw))['id']
        self.store.append(id, 0, raw[:13])
        self.assertEqual(self.store.items(), [])
        with self.assertRaises(ValueError): self.store.finish(id)
        with self.assertRaises(ValueError): self.store.append(id, 0, raw[13:])
        self.store.append(id, 13, raw[13:])
        result = self.store.finish(id)
        self.assertEqual(len(result['sha256']), 64)
        self.assertEqual(self.store.items(), [result])
        self.assertEqual(Path(self.store.selection(id, 1)['path']).read_bytes(), raw)
        self.assertEqual(self.store.selection(id, 0)['strength'], 0)
        self.assertIsNone(self.store.selection('', 1))
        for strength in ('nan', 'inf', -1, 3):
            with self.assertRaises(ValueError): self.store.selection(id, strength)

    def test_invalid_files_and_paths_rejected(self):
        for name, size in [('../fix.safetensors', 100), ('file.pkl', 100), ('fix.safetensors', MAX_SIZE+1)]:
            with self.assertRaises(ValueError): self.store.begin(name, size)
        for raw in (b'not a tensor file', adapter_bytes()[:-1], adapter_bytes('base.weight')):
            with self.assertRaises(ValueError): self.upload(raw)
        with self.assertRaises(ValueError): self.store.selection('../file', 1)
        id = self.store.begin('fix.safetensors', 100)['id']
        self.store.cancel(id)
        self.assertFalse((self.store.root / (id + '.part')).exists())

    def test_backend_commands_select_only_server_files(self):
        id = self.upload()['id']
        output = Path(self.temp.name) / 'image.png'
        settings = {}
        for key in ('sd_cli', 'diffusion', 'encoder', 'vision', 'vae'):
            path = Path(self.temp.name) / key; path.touch(); settings[key] = str(path)
        with patch.object(app, 'LORAS', self.store):
            for backend in ('diffusers', 'sd-cpp'):
                with patch.object(app, 'BACKEND', backend):
                    cmd = app.make_command({'scene': 'boat', 'seed': 79, 'lora_id': id,
                        'lora_strength': 1, 'settings': settings, 'lora': {'path': '/untrusted'}}, None, output)
                    if backend == 'diffusers':
                        request = json.loads(Path(cmd[-1]).read_text())
                        self.assertEqual(request['lora']['id'], id)
                        self.assertNotEqual(request['lora']['path'], '/untrusted')
                    else:
                        self.assertIn('--lora-model-dir', cmd)
                        self.assertTrue(any(f'<lora:{id}:1.0>' in part for part in cmd))

    def test_warm_pipeline_reloads_on_adapter_switch_and_removal(self):
        with patch.object(worker, 'PIPELINE', None), patch.object(worker, 'PIPELINE_KEY', None), patch.object(worker, 'release_pipeline'), patch.object(worker, 'load_pipeline', side_effect=lambda *args: object()) as load:
            request = {'lora': {'id': 'a'*32, 'strength': 1}}
            first = worker.get_pipeline('/models', request)
            self.assertIs(first, worker.get_pipeline('/models', request))
            changed = worker.get_pipeline('/models', {'lora': {'id': 'a'*32, 'strength': .5}})
            self.assertIsNot(first, changed)
            removed = worker.get_pipeline('/models', {})
            self.assertIsNot(changed, removed)
            self.assertIs(removed, worker.get_pipeline('/models', {'lora': {'id': 'a'*32, 'strength': 0}}))
            self.assertEqual(load.call_count, 3)

    def test_adapter_loader_preserves_base_precision_and_rejects_failure(self):
        pipe = MagicMock()
        base, adapter = MagicMock(), MagicMock()
        pipe.transformer.named_parameters.return_value = [('norm.weight', base), ('layer.lora_A.uploaded.weight', adapter)]
        torch = MagicMock()
        diffusers = SimpleNamespace(QwenImage21Pipeline=MagicMock())
        diffusers.QwenImage21Pipeline.from_pretrained.return_value = pipe
        transformers = SimpleNamespace(Qwen3VLForConditionalGeneration=MagicMock())
        request = {'offload': False, 'lora': {'path': '/models/fix.safetensors', 'name': 'fix', 'strength': 1}}
        with patch.dict('sys.modules', {'torch': torch, 'diffusers': diffusers, 'transformers': transformers}), patch.object(worker, 'verify_checkpoint'), patch.object(worker, 'assert_bf16'):
            worker.load_pipeline(Path('/models'), request)
            pipe.load_lora_weights.assert_called_once_with('/models/fix.safetensors', adapter_name='uploaded', local_files_only=True, use_safetensors=True)
            pipe.set_adapters.assert_called_once_with(['uploaded'], adapter_weights=[1])
            base.data.to.assert_not_called()
            pipe.load_lora_weights.side_effect = ValueError('wrong shape')
            with self.assertRaisesRegex(RuntimeError, 'Could not load this LoRA'):
                worker.load_pipeline(Path('/models'), request)

    def test_http_upload_authentication_and_roundtrip(self):
        with patch.object(app, 'LORAS', self.store), patch.object(app, 'PASSWORD', 'test'):
            server = ThreadingHTTPServer(('127.0.0.1', 0), app.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            def call(path, data=None, binary=False, auth=True):
                headers = {'Content-Type': 'application/octet-stream' if binary else 'application/json'}
                if auth: headers['Authorization'] = 'Basic ' + base64.b64encode(b'playground:test').decode()
                if binary: headers['X-Upload-Offset'] = '0'
                req = Request(f'http://127.0.0.1:{server.server_port}'+path,
                    data if binary else json.dumps(data).encode() if data is not None else None, headers)
                with urlopen(req, timeout=5) as response: return json.load(response)
            try:
                with self.assertRaises(HTTPError) as caught: call('/api/loras', auth=False)
                self.assertEqual(caught.exception.code, 401)
                raw = adapter_bytes()
                id = call('/api/loras/uploads', {'name': 'fix.safetensors', 'size': len(raw)})['id']
                call(f'/api/loras/uploads/{id}/chunk', raw, binary=True)
                call(f'/api/loras/uploads/{id}/finish', {})
                self.assertEqual(call('/api/loras')['items'][0]['id'], id)
            finally:
                server.shutdown(); server.server_close(); thread.join()
