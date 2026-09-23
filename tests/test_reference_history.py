import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import app

class ReferenceHistoryTests(unittest.TestCase):
    def test_saved_reference_reuse_and_legacy_reference(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(app, 'OUTPUTS', Path(temp)), patch.object(app, 'REFERENCES', Path(temp)/'references'):
            data = 'data:image/png;base64,' + base64.b64encode(b'fixture-image').decode()
            saved = app.save_reference({'name': 'study.png', 'reference': data})
            self.assertEqual(app.reference_records()[0]['name'], 'study.png')
            render = Path(temp)/('a'*32); render.mkdir()
            result = app.resolve_reference({'reference_id': saved['id']}, render)
            self.assertEqual(result.read_bytes(), b'fixture-image')
            self.assertEqual(len(app.reference_records()), 2)
            with self.assertRaises(ValueError): app.reference_path('../secrets')
            with self.assertRaises(ValueError): app.resolve_reference({'reference_id': saved['id'], 'reference':data}, render)
            with self.assertRaises(ValueError): app.save_reference({'reference':'invalid'})
            self.assertEqual(len(list(app.REFERENCES.iterdir())), 1)

    def test_history_settings_recover_old_request_without_exposing_paths(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(app, 'OUTPUTS', Path(temp)):
            render = Path(temp)/('a'*32); render.mkdir(); (render/'image.png').touch()
            (render/'request.json').write_text(json.dumps({'cfg':3, 'seed':79, 'negative':'blur', 'reference':'/private/reference.png', 'lora':{'path':'/private/adapter', 'name':'fix', 'strength':1}}))
            (render/'metadata.json').write_text(json.dumps({'scene':'boat', 'prompt':'actual expanded boat', 'seed':80}))
            record = app.history_records()[0]
            self.assertEqual(record['settings']['seed'],80)
            self.assertEqual(record['settings']['cfg'],3)
            self.assertEqual(record['settings']['negative'],'blur')
            self.assertNotIn('/private',json.dumps(record))
            self.assertIsNone(record['settings']['characters'])
