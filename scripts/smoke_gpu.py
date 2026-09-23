"""Exercise a running Pod's real GPU pipeline; no provisioning or rental."""
import base64
import json
import os
import time
from urllib.request import Request, urlopen

base = os.environ.get('PLAYGROUND_URL', 'http://127.0.0.1:8765').rstrip('/')
password = os.environ['PLAYGROUND_PASSWORD']
headers = {'Authorization': 'Basic ' + base64.b64encode(('playground:' + password).encode()).decode(),
           'Content-Type': 'application/json'}


def api(path, body=None):
    request = Request(base + path, headers=headers,
                      data=json.dumps(body).encode() if body is not None else None)
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def render(extra):
    body = dict(scene='A small red sailboat on a calm blue lake, watercolor illustration.',
                width=512, height=512, steps=1, cfg=1, seed=42, offload=True,
                live_preview=False, input_mode='style')
    body.update(extra)
    job_id = api('/api/jobs', body)['id']
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        job = api('/api/jobs/' + job_id)
        if job['status'] in ('error', 'cancelled'):
            raise RuntimeError(str(job.get('error')) + '\n' + job['log'])
        if job['status'] == 'done':
            with urlopen(Request(base + job['image'], headers=headers), timeout=60) as response:
                assert response.read(8) == b'\x89PNG\r\n\x1a\n'
            print(job['log'])
            print('Passed:', job_id, flush=True)
            return job_id
        time.sleep(5)
    raise TimeoutError('Generation timed out; inspect the Pod log and cancel the job in the UI.')


config = api('/api/config')
assert config['backend'] == 'diffusers', 'This smoke check requires the BF16 RunPod backend'
source = render({})
render(dict(input_mode='edit', source_history_id=source, scene='Change the sail to yellow.'))
if config['rewriter']:
    render(dict(rewrite=True))
print('GPU text-to-image, editing, and enabled rewriting checks passed (one step; not a quality benchmark).')
