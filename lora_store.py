"""Bounded, chunked safetensors uploads; no pickle loading or client paths."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import threading
import uuid

MAX_SIZE = 2 * 1024**3
CHUNK_SIZE = 8 * 1024**2
LOCK = threading.Lock()


def checked_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{32}', value):
        raise ValueError('Invalid LoRA ID.')
    return value


def validate_file(path):
    total = path.stat().st_size
    with path.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('Invalid safetensors file.')
        length = struct.unpack('<Q', raw)[0]
        if not 2 <= length <= min(8 * 1024**2, total - 8):
            raise ValueError('Invalid safetensors header length.')
        try:
            header = json.loads(stream.read(length))
        except (ValueError, UnicodeError) as exc:
            raise ValueError('Invalid safetensors header.') from exc
    if not isinstance(header, dict):
        raise ValueError('Invalid safetensors header.')
    tensors = {k:v for k,v in header.items() if k != '__metadata__'}
    if not tensors or not any('lora' in k.lower() for k in tensors):
        raise ValueError('This file does not contain LoRA tensors. Upload a Qwen Image 2.1 LoRA, not a base model.')
    intervals = []
    for name, tensor in tensors.items():
        if not ('lora' in name.lower() or name.endswith('.alpha')) or not isinstance(tensor, dict):
            raise ValueError('Unsupported adapter: expected standard LoRA tensors only.')
        dtype = tensor.get('dtype')
        shape = tensor.get('shape')
        offsets = tensor.get('data_offsets')
        if dtype not in {'BF16', 'F16', 'F32'} or not isinstance(shape, list) or not all(type(n) is int and n > 0 for n in shape):
            raise ValueError('LoRA tensors must use BF16, F16, or F32 floating-point weights.')
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(n) is int for n in offsets):
            raise ValueError('Invalid tensor offsets.')
        start, end = offsets
        if start < 0 or end - start != math.prod(shape) * (4 if dtype == 'F32' else 2):
            raise ValueError('Invalid tensor size.')
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError('Invalid or overlapping safetensors data.')
        cursor = end
    if cursor != total - 8 - length:
        raise ValueError('Truncated or invalid safetensors data.')


class Store:
    def __init__(self, root):
        self.root = Path(root)

    def begin(self, name, size):
        if not isinstance(name, str) or '/' in name or '\\' in name or not name.lower().endswith('.safetensors') or len(name) > 200:
            raise ValueError('Choose a .safetensors LoRA file.')
        if type(size) is not int or not 8 < size <= MAX_SIZE:
            raise ValueError('LoRA size must be at most 2 GiB.')
        self.root.mkdir(parents=True, exist_ok=True)
        id = uuid.uuid4().hex
        (self.root / (id + '.upload.json')).write_text(json.dumps({'id': id, 'name': name, 'size': size}))
        (self.root / (id + '.part')).touch()
        return {'id': id, 'chunk_size': CHUNK_SIZE}

    def append(self, id, offset, content):
        checked_id(id)
        with LOCK:
            info = json.loads((self.root / (id + '.upload.json')).read_text())
            path = self.root / (id + '.part')
            if offset != path.stat().st_size:
                raise ValueError('Upload offset mismatch. Select the file again to retry.')
            if not 0 < len(content) <= CHUNK_SIZE or offset + len(content) > info['size']:
                raise ValueError('Invalid upload chunk size.')
            with path.open('ab') as stream:
                stream.write(content)
            return {'offset': path.stat().st_size}

    def finish(self, id):
        checked_id(id)
        with LOCK:
            info_path = self.root / (id + '.upload.json')
            info = json.loads(info_path.read_text())
            path = self.root / (id + '.part')
            if path.stat().st_size != info['size']:
                raise ValueError('Upload is incomplete.')
            validate_file(path)
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                while chunk := stream.read(CHUNK_SIZE): digest.update(chunk)
            info['sha256'] = digest.hexdigest()
            path.replace(self.root / (id + '.safetensors'))
            (self.root / (id + '.json')).write_text(json.dumps(info))
            info_path.unlink()
            return info

    def cancel(self, id):
        checked_id(id)
        with LOCK:
            for suffix in ('.part', '.upload.json'):
                (self.root / (id + suffix)).unlink(missing_ok=True)

    def items(self):
        result = []
        for file in self.root.glob('*.json'):
            if not re.fullmatch(r'[a-f0-9]{32}\.json', file.name): continue
            try:
                info = json.loads(file.read_text())
                if (self.root / (file.stem + '.safetensors')).is_file(): result.append(info)
            except (OSError, ValueError): continue
        return sorted(result, key=lambda item: item['name'].lower())

    def selection(self, id, strength):
        if not id: return None
        checked_id(id)
        try: strength = float(strength)
        except (ValueError, TypeError): raise ValueError('Invalid LoRA strength.') from None
        if not math.isfinite(strength) or not 0 <= strength <= 2:
            raise ValueError('LoRA strength must be between 0 and 2.')
        path = self.root / (id + '.safetensors')
        if not path.is_file() or path.is_symlink(): raise ValueError('LoRA file was not found. Upload it again.')
        info = json.loads((self.root / (id + '.json')).read_text())
        return {**info, 'path': str(path.resolve()), 'strength': strength}
