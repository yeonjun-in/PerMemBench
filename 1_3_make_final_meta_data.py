# %%
from LLM import UnifiedLLM
import json
import os
from glob import glob
from tqdm import tqdm
import argparse
from collections import defaultdict
from copy import deepcopy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='judgment_results_v3')
parser.add_argument('--persona_metadata_dir', type=str, default='persona_metadata_domains_v3')
parser.add_argument('--domain_list_file', type=str, default='domain_list_v3.txt')
parser.add_argument('--output_dir', type=str, default='final_persona_metadata_v3')
args = parser.parse_args()

domain_list = open(args.domain_list_file, 'r', encoding='utf-8').read()

domain_list = [a.split('. ')[1] for a in domain_list.split('\n')]

judge1 = json.load(open(os.path.join(args.input_dir, 'openai_gpt-5.1_judgments.json'), 'r', encoding='utf-8'))
judge2 = json.load(open(os.path.join(args.input_dir, 'openai_o3-mini_judgments.json'), 'r', encoding='utf-8'))
judge3 = json.load(open(os.path.join(args.input_dir, 'openai_o3-mini_judgments.json'), 'r', encoding='utf-8'))

# %%
valid_domains_per_uid = {}
for inst1, inst2, inst3 in zip(judge1['results'], judge2['results'], judge3['results']):
    assert inst1['uuid'] == inst2['uuid'] == inst3['uuid']
    uid = inst1['uuid']
    for a in inst1['domain_judgments']:
        assert a['domain_name'] in domain_list
    for a in inst2['domain_judgments']:
        assert a['domain_name'] in domain_list
    for a in inst3['domain_judgments']:
        assert a['domain_name'] in domain_list

    ### use_decision 
    successful_domain1 = []
    temp = defaultdict(list)
    for a in inst1['domain_judgments']:
        try:
            temp[a['domain_name']].append(a['use_decision']['reasonable'] if a['use_decision']['reasonable'] is not None else False)
        except:
            temp[a['domain_name']].append(False)
    for a in inst2['domain_judgments']:
        try:
            temp[a['domain_name']].append(a['use_decision']['reasonable'] if a['use_decision']['reasonable'] is not None else False)
        except:
            temp[a['domain_name']].append(False)
    for a in inst3['domain_judgments']:
        try:
            temp[a['domain_name']].append(a['use_decision']['reasonable'] if a['use_decision']['reasonable'] is not None else False)
        except:
            temp[a['domain_name']].append(False)
    for k,v in temp.items():
        try:
            if sum(v) == 3:
                successful_domain1.append(k)
        except:
            continue
    ### memory_required
    successful_domain2 = []
    temp = defaultdict(list)
    for a in inst1['domain_judgments']:
        try:
            temp[a['domain_name']].append(a['memory_required_decision']['reasonable'] if a['memory_required_decision']['reasonable'] is not None else False)
        except:
            temp[a['domain_name']].append(False)
    for a in inst2['domain_judgments']:
        try:
            temp[a['domain_name']].append(a['memory_required_decision']['reasonable'] if a['memory_required_decision']['reasonable'] is not None else False)
        except:
            temp[a['domain_name']].append(False)
    for a in inst3['domain_judgments']:
        try:
            temp[a['domain_name']].append(a['memory_required_decision']['reasonable'] if a['memory_required_decision']['reasonable'] is not None else False)
        except:
            temp[a['domain_name']].append(False)
    for k,v in temp.items():
        try:
            if sum(v) == 3:
                successful_domain2.append(k)
        except:
            continue
    valid_domains =sorted(set(successful_domain1).intersection(set(successful_domain2)))
    if len(valid_domains) >= 10:
        valid_domains_per_uid[uid] = valid_domains

os.makedirs(os.path.join(args.output_dir), exist_ok=True)

# %%
for key in valid_domains_per_uid:
    meta_data = json.load(open(os.path.join(args.persona_metadata_dir, f'{key}.json'), 'r', encoding='utf-8'))
    if meta_data['persona']['age'] < 10 or meta_data['persona']['age'] > 80:
        continue
    for a in meta_data['domains']:
        assert a['domain_name'] in domain_list
    
    v_domains = valid_domains_per_uid[key]
    new_meta_data = deepcopy(meta_data)
    new_meta_data['domains'] = [a for a in meta_data['domains'] if a['domain_name'] in v_domains and a['use']]
    
    ### 모든 domain 이 memory required 이면 뺌
    if sum([a['memory_required'] for a in new_meta_data['domains']]) == len(domain_list):
        continue

    json.dump(new_meta_data, open(os.path.join(args.output_dir, f'{key}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)



