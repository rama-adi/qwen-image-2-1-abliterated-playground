import base64
import json
from pathlib import Path
import socket
import tempfile
import threading
import sys
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request as HTTPRequest, urlopen
from http.server import ThreadingHTTPServer

from wsproto import WSConnection, ConnectionType
from wsproto.events import Request, TextMessage, Ping, Pong, CloseConnection
import app


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.job_id = 'a' * 32
        self.job = {'id': self.job_id, 'directory': self.directory.name,
                    'output': str(Path(self.directory.name) / 'image.png'),
                    'steps': 3, 'step': 0, 'stage': 'loading', 'status': 'running', 'events': []}
        self.password = patch.object(app, 'PASSWORD', 'test-password')
        self.password.start()
        self.jobs = patch.dict(app.JOBS, {self.job_id: self.job})
        self.jobs.start()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), app.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host = f'127.0.0.1:{self.server.server_port}'
        self.authorization = 'Basic ' + base64.b64encode(b'playground:test-password').decode()
        self.connections = []

    def tearDown(self):
        for connection in self.connections:
            connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.jobs.stop()
        self.password.stop()
        self.directory.cleanup()

    def connect(self):
        connection = socket.create_connection(('127.0.0.1', self.server.server_port), timeout=3)
        self.connections.append(connection)
        ws = WSConnection(ConnectionType.CLIENT)
        connection.sendall(ws.send(Request(host=self.host, target=f'/api/jobs/{self.job_id}/events',
            extra_headers=[(b'authorization', self.authorization.encode()),
                           (b'origin', ('http://' + self.host).encode())])))
        return connection, ws

    def messages_until(self, connection, ws, predicate):
        messages = []
        while True:
            chunk = connection.recv(65536)
            self.assertTrue(chunk, 'Stream closed before the expected update')
            ws.receive_data(chunk)
            for event in ws.events():
                if isinstance(event, TextMessage):
                    payload = json.loads(event.data)
                    messages.append(payload)
                    if predicate(payload):
                        return messages
                elif isinstance(event, Ping):
                    connection.sendall(ws.send(event.response()))

    def test_stream_preserves_each_step_and_reconnects_to_current_state(self):
        connection, ws = self.connect()
        self.messages_until(connection, ws, lambda message: message['step'] == 0)
        for step in (1, 2, 3):
            app.consume_progress(self.job, f'PLAYGROUND_EVENT {{"stage":"sampling","step":{step}}}')
        messages = self.messages_until(connection, ws, lambda message: message['step'] == 3)
        self.assertEqual([message['step'] for message in messages], [1, 2, 3])
        connection.close()
        connection, ws = self.connect()
        current = self.messages_until(connection, ws, lambda message: message['step'] == 3)
        self.assertEqual(current[-1]['status'], 'running')
        self.job['status'] = 'done'
        finished = self.messages_until(connection, ws, lambda message: message['status'] == 'done')
        self.assertEqual(finished[-1]['image'], f'/api/jobs/{self.job_id}/image')

    def test_http_reference_request_arrives_in_multiple_network_reads(self):
        raw = json.dumps({'scene': 'boat', 'reference': 'data:image/png;base64,' + 'A'*150000}).encode()
        connection = socket.create_connection(('127.0.0.1', self.server.server_port), timeout=3)
        self.connections.append(connection)
        headers = (f'POST /api/preview HTTP/1.1\r\nHost: {self.host}\r\n'
                   f'Authorization: {self.authorization}\r\nContent-Length: {len(raw)}\r\n'
                   'Content-Type: application/json\r\nConnection: close\r\n\r\n').encode()
        connection.sendall(headers + raw[:184])
        time.sleep(.05)
        connection.sendall(raw[184:65000])
        time.sleep(.05)
        connection.sendall(raw[65000:])
        response = b''
        while chunk := connection.recv(65536):
            response += chunk
        self.assertIn(b'200 OK', response.split(b'\r\n', 1)[0])
        self.assertIn('boat', json.loads(response.split(b'\r\n\r\n', 1)[1])['prompt'])

    def test_stream_rejects_missing_auth_and_cross_origin(self):
        path = f'http://{self.host}/api/jobs/{self.job_id}/events'
        for headers, expected in [({}, 401), ({'Authorization': self.authorization,
                                               'Origin': 'https://unrelated.example'}, 403)]:
            with self.assertRaises(HTTPError) as raised:
                urlopen(HTTPRequest(path, headers=headers))
            self.assertEqual(raised.exception.code, expected)

    def test_carriage_return_progress_arrives_before_next_line(self):
        code = "import sys,time; sys.stdout.write('\\r |===| 1/3 - 2.0s/it'); sys.stdout.flush(); time.sleep(1)"
        worker = threading.Thread(target=app.run_job, args=(self.job_id, [sys.executable, '-c', code]))
        worker.start()
        deadline = time.monotonic() + 0.8
        while self.job['step'] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        observed = self.job['step']
        worker.join(timeout=3)
        self.assertEqual(observed, 1, 'Progress was buffered until the next newline')

    def test_native_progress_excludes_weight_loading_bars(self):
        app.consume_progress(self.job, '|###| 2/3 - 700.5MB/s')
        self.assertEqual(self.job['step'], 0)
        app.consume_progress(self.job, '\x1b[32m  |===| 1/3 - 2.35s/it\x1b[K')
        self.assertEqual(self.job['step'], 1)
        self.assertEqual(self.job['stage'], 'sampling')
