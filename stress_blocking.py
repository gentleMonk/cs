"""Adversarial candidate generation tests: show why perfect synthetic blocking is fragile."""
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import json,pandas as pd
BASE=Path(__file__).parent
FIX=BASE/'fixtures';OUT=BASE/'results'

def block_keys(r):
    return ["region_sector:"+r['region']+'|'+r['sector'],
            "bank:"+r['bank_suffix'],
            "sectorname:"+r['sector']+'|'+r['name'].replace(r['region'],'')[:1],
            "prefix:"+r['name'][:4]]

def stats(df,ground_truth):
    data=df.set_index('record_id').to_dict('index')
    groups=defaultdict(list)
    for recid,r in data.items():
        for k in block_keys(r):
            if k:groups[k].append(recid)
    candidate=set()
    for ids in groups.values():
        for a,b in combinations(ids,2):candidate.add(tuple(sorted((a,b))))
    ent=defaultdict(list)
    for i,v in ground_truth.items():ent[v].append(i)
    actual={tuple(sorted((a,b))) for ids in ent.values() for a,b in combinations(ids,2)}
    found=actual & candidate
    return {'candidate_pairs':len(candidate),'true_pairs':len(actual),'found_true_pairs':len(found),
            'recall':round(len(found)/len(actual),4),
            'false_candidate_pairs':len(candidate-found),
            'candidate_reduction_vs_all_pairs':round(1-len(candidate)/(len(df)*(len(df)-1)/2),4)}

def main():
    df=pd.read_csv(FIX/'synthetic_entities_harder.csv',dtype=str,keep_default_na=False)
    truth=json.loads((FIX/'synthetic_entities_harder_truth.json').read_text())
    baseline=stats(df,truth)
    renamed=df.copy()
    dupeids=sorted([x for x in df.record_id if x.startswith('D')])[:12]
    for id in dupeids:
        loc=renamed.record_id.eq(id)
        # No overlapping observable blocking keys with original entity after relocation,
        # rebranding, and payment account change. Ground truth still says same entity.
        renamed.loc[loc,'name']='新集团'+id+'控股有限公司'
        renamed.loc[loc,'region']='迁址'+id
        renamed.loc[loc,'sector']='新行业'+id
        renamed.loc[loc,'bank_suffix']='新账号'+id
    broken=stats(renamed,truth)
    common=df.copy()
    # 100 unrelated records share a generic organisational prefix.
    base_rows=common[common.record_id.str.startswith('B')].head(100).record_id
    mask=common.record_id.isin(base_rows)
    common.loc[mask,'name']='中国企业'+common.loc[mask,'name']
    prefix=stats(common,truth)
    report={'baseline':baseline,'identity_signal_erasure_12_true_pairs':broken,
            'shared_name_prefix_100_records':prefix,
            'warning':'Synthetic adversarial changes. Not Splink library. Deliberately removing all shared features makes true duplicates unidentifiable from the chosen blocking features.'}
    (OUT/'stress_blocking.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
