import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reader', ROOT/'scripts/prepare_memory_reader.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class ReaderContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/'configs/memory_reader_v1.json').read_text())
        self.ids = [f'C{i:02d}' for i in range(1, 7)]
        self.item = dict(user='private-user', trajectory='private-session', cutoff_ns=1000000000,
            context=[dict(poi='raw-a', category='Cafe', time_ns=0), dict(poi='raw-outside', category='Park', time_ns=1000000000)],
            candidates=[dict(poi='raw-a' if i==1 else f'raw-{i}', candidate_id=f'C{i:02d}', category='Shop', distance_km=i) for i in range(1,7)],
            shared_sha256='unused-in-render-unit-test', memory={k:dict(text='' if k=='context' else k+' evidence') for k in self.cfg['conditions']})

    def test_strict_parser_keeps_order(self):
        ids = ['C06','C02','C01','C04','C03']
        self.assertEqual(reader.parse_predictions(' '+json.dumps(ids)+'\n', self.ids), dict(valid=True,predictions=ids,reason=None))

    def test_parser_rejects_unrecoverable_outputs(self):
        bad = ['```json\n["C01","C02","C03","C04","C05"]\n```',
               '["C01","C01","C03","C04","C05"]', '["C01","C02","C03","C04","H01"]',
               '["C01"]', '{"predictions":["C01","C02","C03","C04","C05"]}',
               '[1,2,3,4,5]', '["C01","C02","C03","C04","C05"] explanation']
        for raw in bad:
            with self.subTest(raw=raw):
                parsed=reader.parse_predictions(raw,self.ids)
                self.assertFalse(parsed['valid']); self.assertEqual(parsed['predictions'],[])

    def test_render_excludes_identity_and_raw_poi(self):
        original=reader.messages(self.item,'personal',self.cfg)
        changed=copy.deepcopy(self.item);changed.update(user='another',trajectory='another')
        self.assertEqual(original,reader.messages(changed,'personal',self.cfg))
        text=json.dumps(original)
        self.assertNotIn('private-user',text);self.assertNotIn('raw-',text)
        self.assertIn('C01',text);self.assertIn('H01',text)

    def test_shared_fields_invariant_and_target_rejected(self):
        rendered=[reader.messages(self.item,k,self.cfg) for k in self.cfg['conditions']]
        self.assertTrue(all(x[0]==rendered[0][0] for x in rendered))
        prefixes=[x[1]['content'].split('\nAdditional memory:\n')[0] for x in rendered]
        self.assertEqual(len(set(prefixes)),1)
        modified=copy.deepcopy(self.item);modified['target']='C01'
        with self.assertRaises(AssertionError):reader.messages(modified,'personal',self.cfg)
        modified=copy.deepcopy(self.item);modified['context'][0]['time_ns']=2000000000
        with self.assertRaises(AssertionError):reader.messages(modified,'personal',self.cfg)


if __name__=='__main__':unittest.main()
