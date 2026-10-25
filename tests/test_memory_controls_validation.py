import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_memory_controls import HEADER,START,END,validate_control


class Validation(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/memory_controls_v1.json').read_text())
        ids=[f'C{i:02}' for i in range(1,9)]
        rows=[dict(source='personal',candidate_id=ids[0],support_count=3,source_share=.5,days_since_latest_support=2.)]
        self.item=dict(user=1,trajectory=2,shared_sha256='shared',candidates=[dict(candidate_id=i) for i in ids],memory={'personal':{'rows':rows}})
        msg=[dict(role='system',content='system'),dict(role='user',content='shared'+START+HEADER+'\npersonal: C01 | 3 | 0.500000 | 2.000000'+END)]
        self.old={'personal':dict(messages=msg,source_shared_sha256='shared',valid_ids=ids,input_tokens=100,input_token_ids_sha256='old')}
        offset=1+int(hashlib.sha256(b'memory-controls-v1-20261026:1:2:personal_rebound').hexdigest(),16)%7
        changed=copy.deepcopy(msg);changed[1]['content']=changed[1]['content'].replace('personal: C01','personal: '+ids[offset])
        self.row=dict(user=1,trajectory=2,condition='personal_rebound',original_condition='personal',source_shared_sha256='shared',valid_ids=ids,
                      messages=changed,input_tokens=100,input_token_ids_sha256='new',offset=offset,changed_rows=1,action='generate',reuse_condition=None)

    def test_prescribed_rebinding_accepted(self):
        self.assertEqual(validate_control(self.item,self.row,self.old,self.cfg),1)

    def test_shared_context_and_numeric_tampering_rejected(self):
        for before,after in [('shared','leaked target'),(' | 3 |',' | 4 |')]:
            r=copy.deepcopy(self.row);r['messages'][1]['content']=r['messages'][1]['content'].replace(before,after)
            with self.assertRaises(AssertionError):validate_control(self.item,r,self.old,self.cfg)

    def test_cross_user_false_cache_and_length_rejected(self):
        for changes in [dict(user=2),dict(action='reuse',reuse_condition='personal'),dict(input_tokens=101),dict(changed_rows=0)]:
            r=dict(self.row,**changes)
            with self.assertRaises(AssertionError):validate_control(self.item,r,self.old,self.cfg)

    def test_exact_prompt_cache_accepted_and_wrong_reference_rejected(self):
        self.item['memory']['fixed_fusion']={'rows':self.item['memory']['personal']['rows']}
        self.old['fixed_fusion']=copy.deepcopy(self.old['personal'])
        r=dict(self.row,condition='personal6',offset=None,changed_rows=0,action='reuse',reuse_condition='personal',
               messages=copy.deepcopy(self.old['personal']['messages']),input_token_ids_sha256='old')
        self.assertEqual(validate_control(self.item,r,self.old,self.cfg),0)
        r['reuse_condition']='fixed_fusion'
        with self.assertRaises(AssertionError):validate_control(self.item,r,self.old,self.cfg)


if __name__=='__main__':unittest.main()
