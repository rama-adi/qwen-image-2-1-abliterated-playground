import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import app
from reference_roles import selected_references, reference_instructions

class MultipleReferenceTests(unittest.TestCase):
    def test_ordered_images_roles_and_backend_arguments(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, 'OUTPUTS', Path(directory)), patch.object(app, 'REFERENCES', Path(directory)/'references'):
            ids=[]
            for color in (b'first', b'second'):
                ids.append(app.save_reference({'reference':'data:image/png;base64,'+base64.b64encode(color).decode()})['id'])
            data={'scene':'A boat', 'seed':79, 'references':[
                {'id':ids[1], 'role':'pose','note':'Person on the left'}, {'id':ids[0], 'role':'style'}]}
            render=Path(directory)/'render';render.mkdir()
            paths=app.resolve_reference(data,render)
            self.assertEqual([p.read_bytes() for p in paths],[b'second',b'first'])
            prompt=app.assembled_prompt(data)
            self.assertIn('Image 1: Borrow the body pose',prompt)
            self.assertIn('Image 2: Borrow the visual style',prompt)
            self.assertIn('Person on the left',prompt)
            with patch.object(app,'BACKEND','diffusers'):
                command=app.make_command(data,paths,render/'image.png')
                request=json.loads(Path(command[-1]).read_text())
                self.assertEqual(request['references'],[str(p) for p in paths])
                self.assertIn('Image 1:',request['reference_instructions'])
            data['settings']={}
            for key in ('sd_cli','diffusion','encoder','vae','vision'):
                file=Path(directory)/key;file.touch();data['settings'][key]=str(file)
            with patch.object(app,'BACKEND','sd-cpp'):
                command=app.make_command(data,paths,render/'image.png')
                self.assertEqual([command[i+1] for i,v in enumerate(command) if v=='-r'],[str(p) for p in paths])
            data['input_mode']='edit'
            self.assertIn('Image 1: This is the source image to edit',app.assembled_prompt(data))

    def test_limits_and_invalid_inputs(self):
        ref={'id':'a'*32,'role':'pose'}
        for data in ({'references':[ref]*5}, {'references':[ref],'reference_id':'b'*32},
                     {'references':[{'id':'../x'}]}, {'references':[{'id':'a'*32,'role':{}}]},
                     {'references':[{'id':'a'*32,'role':'unknown'}]}, {'references':[None]}):
            with self.subTest(data=data), self.assertRaises(ValueError): selected_references(data)
        with self.assertRaises(ValueError): app.assembled_prompt({'scene':'boat','input_mode':'edit'})
        self.assertIn('Edit the supplied image', app.assembled_prompt({'scene':'boat','input_mode':'edit','reference_id':'a'*32}))
