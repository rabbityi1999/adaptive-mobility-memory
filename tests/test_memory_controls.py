import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_memory_controls import render, transform


class Controls(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/memory_controls_v1.json').read_text())
        p=[dict(source='personal',candidate_id=f'C{i:02}',support_count=20-i,source_share=.01*i,days_since_latest_support=i*2.) for i in range(1,9)]
        g=[dict(source='population',candidate_id=f'C{i:02}',support_count=10-i,source_share=.02*i,days_since_latest_support=i*3.) for i in range(1,4)]
        f=[]
        for i in range(6):
            f.append(p[i])
            if i<len(g): f.append(g[i])
        self.item=dict(user=7,trajectory=19,cutoff_ns=123,context=[],candidates=[dict(candidate_id=f'C{i:02}') for i in range(1,11)],shared_sha256='test',
                       memory={c:dict(rows=rows,text=render(rows)) for c,rows in [('personal',p),('fixed_fusion',f)]})

    def test_personal_budget_is_fusion_personal(self):
        x,_,count,_=transform(self.item,'personal6',self.cfg)
        self.assertEqual(count,2)
        self.assertEqual(x['memory']['personal6']['rows'],self.item['memory']['personal']['rows'][:6])

    def test_rebinding_preserves_all_non_id_fields_and_original(self):
        old=copy.deepcopy(self.item)
        for c in ['personal_rebound','fusion_population_rebound']:
            x,origin,count,_=transform(self.item,c,self.cfg)
            rows=x['memory'][c]['rows']; prior=self.item['memory'][origin]['rows']
            self.assertEqual(count,8 if c=='personal_rebound' else 3)
            for a,b in zip(rows,prior):
                self.assertEqual({k:v for k,v in a.items() if k!='candidate_id'},{k:v for k,v in b.items() if k!='candidate_id'})
                self.assertEqual(a['candidate_id']==b['candidate_id'],c=='fusion_population_rebound' and a['source']=='personal')
            self.assertEqual(transform(self.item,c,self.cfg)[0],x)
        self.assertEqual(self.item,old)

    def test_null_source_and_target_rejected(self):
        self.item['memory']['fixed_fusion']['rows']=self.item['memory']['personal']['rows'][:6]
        self.item['memory']['fixed_fusion']['text']=render(self.item['memory']['fixed_fusion']['rows'])
        x,_,n,_=transform(self.item,'fusion_population_rebound',self.cfg)
        self.assertEqual(n,0)
        self.assertEqual(x['memory']['fusion_population_rebound']['text'],self.item['memory']['fixed_fusion']['text'])
        self.item['target']='forbidden'
        with self.assertRaises(AssertionError):transform(self.item,'personal_rebound',self.cfg)


if __name__=='__main__': unittest.main()
