"""Run against the container's published port; no weights or GPU required."""
import base64
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

url = 'http://127.0.0.1:8765'
for attempt in range(60):
    try:
        with urlopen(url + '/healthz', timeout=2) as response:
            assert json.load(response)['backend'] == 'diffusers'
        break
    except (URLError, TimeoutError):
        time.sleep(1)
else:
    raise RuntimeError('Container did not become ready')
try:
    urlopen(url + '/api/config')
    raise AssertionError('UI must require authentication')
except HTTPError as error:
    assert error.code == 401
headers = {'Authorization': 'Basic ' + base64.b64encode(b'playground:ci-test-only').decode()}
with urlopen(Request(url + '/api/config', headers=headers)) as response:
    assert json.load(response)['backend'] == 'diffusers'
with urlopen(Request(url + '/', headers=headers)) as response:
    assert b'qwen-image-2-1-abliterated-playground' in response.read()
print('Container health, authentication, backend selection, and UI passed.')
