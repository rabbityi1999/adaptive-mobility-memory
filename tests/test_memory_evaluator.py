import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('eval_reader',ROOT/'scripts/evaluate_memory_reader.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


class EvaluatorTests(unittest.TestCase):
    def test_rank_and_absent_target(self):
        ids=['C01','C02','C03','C04','C05']
        self.assertEqual(e.scores(ids,'C01'),[1.,1.,1.])
        self.assertEqual(e.scores(ids,'C05'),[0.,1.,1/math.log2(6)])
        self.assertEqual(e.scores(ids,None),[0.,0.,0.])
        self.assertEqual(e.scores([],'C01'),[0.,0.,0.])

    def test_strict_parser_independent_failures(self):
        allowed=['C01','C02','C03','C04','C05']
        self.assertTrue(e.strict_parse(json.dumps(allowed),allowed)['valid'])
        for value in [json.dumps(allowed)+' text',json.dumps(allowed[:4]),json.dumps(['C01']*5),json.dumps(allowed[:4]+['H01'])]:
            self.assertEqual(e.strict_parse(value,allowed)['predictions'],[])

    def test_refuses_unfinished_or_preflight_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with self.assertRaises(AssertionError):e.require_completed(p)
            (p/'exit.json').write_text('{"exit_code":0}')
            (p/'status.json').write_text('{"stage":"preflight_pass"}')
            (p/'summary.json').write_text('{"status":"runner_cpu_preflight_pass"}')
            with self.assertRaises(AssertionError):e.require_completed(p)

    def test_denominator_oracle_and_pairs(self):
        vectors={'context':[[0,0,0]]*4,'personal':[[1,1,1],[1,1,1],[0,0,0],[0,0,0]],
                 'population':[[0,0,0],[0,0,0],[1,1,1],[0,0,0]],'fixed_fusion':[[0,0,0]]*4}
        valid={k:[True,True,True,False] for k in vectors}
        r=e.summarize(vectors,valid,[True,True,True,False])
        self.assertEqual(r['conditions']['personal']['metrics']['Acc1'],.5)
        self.assertEqual(r['conditions']['personal']['conditional_metrics']['Acc1'],2/3)
        self.assertEqual(r['oracle_personal_population']['acc1'],.75)
        self.assertEqual(r['oracle_personal_population']['neither'],1)
        self.assertEqual(e.holm([.01,.04,.03]),[.03,.06,.06])
        self.assertEqual(e.paired([[1,1,1]]*6,[[0,0,0]]*6)['acc1_exact_p'],.03125)


if __name__=='__main__':unittest.main()
