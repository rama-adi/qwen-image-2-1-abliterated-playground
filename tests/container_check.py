"""Build-time inference imports, adapter math, and authenticated entrypoint smoke test."""
import os
from pathlib import Path
import subprocess
import sys

import peft
import torch
import wsproto
from diffusers import QwenImage21Pipeline
from transformers import Qwen3VLForConditionalGeneration, AutoModelForImageTextToText

assert torch.version.cuda, 'Expected CUDA-enabled PyTorch'
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', '/checks', '-p', 'test_adapter_compat.py', '-v'], check=True)
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', '/checks', '-p', 'test_guidance_math.py', '-v'], check=True)
env = {**os.environ, 'PLAYGROUND_PASSWORD': 'ci-test-only', 'PLAYGROUND_DOWNLOAD_MODELS': '0',
       'PLAYGROUND_REQUIRE_CUDA': '0', 'PORT': '8765'}
process = subprocess.Popen(['bash', '/usr/local/bin/playground-entrypoint'], env=env)
try:
    subprocess.run([sys.executable, '/checks/smoke_http.py'], check=True, timeout=90)
finally:
    process.terminate()
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait()
Path('/tmp/playground-validated').write_text('imports, DoRA math, entrypoint, authentication and HTTP passed\n')
