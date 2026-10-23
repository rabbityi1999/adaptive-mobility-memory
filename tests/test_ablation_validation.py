"""Synthetic contracts for scoring, fixed denominators and reuse provenance.

No real trajectories, model calls, remote connections or benchmark scores.
The fixture has analytically known rank-one hits rather than model outputs.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ablation_adapter import MASKS, SOURCE_CONDITION, digest, sha
from validate_ablation_results import validate

LABEL = 'a' * 24
MISS = 'b' * 24
HITS = dict(recon_base=20, recon_stm=30, recon_wkg=10, recon_cke=50,
            recon_stm_wkg=35, recon_full=40)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ValidationContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.run, self.source = root/'run', root/'source'
        self.out = root/'summary.json'
        self.args = SimpleNamespace(run=self.run, output=self.out)
        self.selected = [[str(i), '1'] for i in range(200)]
        save(self.run/'selected_queries.json', self.selected)
        save(self.run/'status.json', dict(stage='completed', finished_queries=200, model_calls=999))
        save(self.run/'exit.json', dict(exit_code=0))
        save(self.source/'run_metadata.json', dict(conditions={SOURCE_CONDITION: 'agent_move_v6'}))
        (self.source/'wkg_cache.jsonl').write_text('{}\n')
        manifests, calls, source_hashes = [], [], {}
        for u,t in self.selected:
            query = u+':'+t
            name = f'llama3-8B-local_agent_move_v6_{u}_{t}_False.json'
            for condition in MASKS:
                prompt = f'synthetic {query} {condition}'
                failed = condition == 'recon_cke' and u == '199'
                pred = [] if failed else [LABEL if int(u) < HITS[condition] else MISS]
                result = dict(input='' if failed else prompt, output={}, prediction=pred,
                              true=LABEL, execution_status='error' if failed else 'completed')
                save(self.run/'results'/condition/name, result)
                manifests.append(dict(query=query, condition=condition, original_sha256=digest(prompt),
                    effective_sha256=digest(prompt), input_tokens=10, clipped=False, retained={}))
                if condition == 'recon_full':
                    source_path = self.source/'results'/SOURCE_CONDITION/name
                    save(source_path, result)
                    source_hashes[SOURCE_CONDITION+'/'+name] = sha(source_path)
                elif not failed:
                    calls.append(dict(query=query, stage=condition, prompt_original=prompt,
                        prompt_effective=prompt, prompt_sha256=digest(prompt), input_tokens=10,
                        output_tokens=5, seconds=0.1, clipped=False, retained_payloads={}))
        save(self.run/'ablation_prompt_manifest.json', manifests)
        plan = dict(status='reconstructed_ablation_preflight_pass',
            prompt_manifest_sha256=digest(json.dumps(manifests, sort_keys=True)),
            source_run_metadata_sha256=sha(self.source/'run_metadata.json'),
            source_wkg_cache_sha256=sha(self.source/'wkg_cache.jsonl'),
            source_result_tree_sha256=digest(json.dumps(source_hashes, sort_keys=True)),
            source_WKG_cost={'calls': 400}, source_full_prediction_cost={'calls': 200},
            adapter_sha256='synthetic-fixture', retention_counts={})
        save(self.run/'ablation_preflight.json', plan)
        save(self.run/'run_metadata.json', dict(ablation=plan, conditions={c:'agent_move_v6' for c in MASKS},
            selection_sha256=sha(self.run/'selected_queries.json'), ablation_source=str(self.source),
            model='synthetic-no-model', script_sha256='synthetic-fixture'))
        save(self.run/'evaluation.json', dict(complete=True, denominator=200, model_calls=999,
            conditions={c:dict(Acc1=n/200, Acc5=n/200, NDCG5=n/200,
                               official_ndcg_untruncated=n/200) for c,n in HITS.items()}))
        (self.run/'calls.jsonl').write_text('\n'.join(json.dumps(c) for c in calls)+'\n')

    def check(self):
        with redirect_stdout(io.StringIO()):
            validate(self.args)

    def test_known_scores_failure_denominator_and_six_contrasts(self):
        self.check()
        result = json.loads(self.out.read_text())
        for condition, hits in HITS.items():
            self.assertAlmostEqual(result['conditions'][condition]['Acc1'], hits/200)
            self.assertEqual(result['conditions'][condition]['n'], 200)
        self.assertEqual(result['conditions']['recon_cke']['execution_errors'], 1)
        self.assertEqual(result['conditions']['recon_cke']['empty_predictions'], 1)
        self.assertEqual(result['integrity']['reused_full_files'], 200)
        self.assertEqual(sum(c['calls'] for c in result['new_execution_costs'].values()), 999)
        self.assertEqual(len(result['comparisons']), 6)
        stm = next(c for c in result['comparisons'] if c['a'] == 'recon_stm')
        self.assertEqual((stm['a_only_correct'], stm['b_only_correct']), (10, 0))
        self.assertAlmostEqual(stm['metrics']['Acc1']['difference'], .05)
        self.assertAlmostEqual(stm['acc1_exact_p'], 2/1024)
        self.assertAlmostEqual(stm['acc1_holm_p_six_contrasts'], 6/1024)

    def test_unfinished_gate_precedes_manifest_and_scores(self):
        save(self.run/'status.json', dict(stage='recon_base'))
        (self.run/'run_metadata.json').unlink()
        (self.run/'evaluation.json').unlink()
        with self.assertRaisesRegex(SystemExit, 'unfinished'):
            self.check()
        self.assertFalse(self.out.exists())

    def test_changed_label_rejected_without_summary(self):
        path = self.run/'results/recon_stm/llama3-8B-local_agent_move_v6_0_1_False.json'
        result = json.loads(path.read_text())
        result['true'] = MISS
        save(path, result)
        with self.assertRaises(AssertionError):
            self.check()
        self.assertFalse(self.out.exists())

    def test_reused_output_change_rejected_even_when_score_unchanged(self):
        path = self.run/'results/recon_full/llama3-8B-local_agent_move_v6_0_1_False.json'
        result = json.loads(path.read_text())
        result['output'] = {'reason': 'changed but same score'}
        save(path, result)
        with self.assertRaises(AssertionError):
            self.check()
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
