"""OPTIONAL: Original Splink 5 library test. Requires 'pip install splink'.
Matches on observable name, sector, region, bank; entity truth ID is NEVER given to Splink.
2026-11-02: API structure checked against official V5 docs; execution remains unverified.
"""
import json, time
from pathlib import Path
import pandas as pd

BASE=Path(__file__).parent.parent; FIX=BASE/'fixtures';RESULTS=BASE/'results'

def main():
    import splink.comparison_library as cl
    from splink import DuckDBAPI,Linker,SettingsCreator,block_on

    source=pd.read_csv(FIX/'synthetic_entities_harder.csv',dtype=str,keep_default_na=False)
    truth=json.loads((FIX/'synthetic_entities_harder_truth.json').read_text())
    data=source.rename(columns={'record_id':'unique_id'})
    db=DuckDBAPI()
    table=db.register(data,dataset_display_name='synthetic_companies')
    settings=SettingsCreator(
        link_type='dedupe_only',
        comparisons=[
            cl.JaroWinklerAtThresholds('name',[0.93,0.80]),
            cl.ExactMatch('region'),
            cl.ExactMatch('sector'),
            cl.ExactMatch('bank_suffix')
        ],
        blocking_rules_to_generate_predictions=[
            block_on('region','sector'),
            block_on('bank_suffix'),
        ],
    )
    linker=Linker(table,settings)
    start=time.perf_counter()
    linker.training.estimate_probability_two_random_records_match(
        [block_on('bank_suffix','sector')],recall=0.6
    )
    linker.training.estimate_u_using_random_sampling(max_pairs=100_000)
    linker.training.estimate_parameters_using_expectation_maximisation(block_on('bank_suffix'))
    linker.training.estimate_parameters_using_expectation_maximisation(block_on('region','sector'))
    predictions=linker.inference.predict(threshold_match_probability=0.5)
    scored=predictions.as_pandas_dataframe()
    found=set()
    for row in scored.itertuples(index=False):
        a=str(getattr(row,'unique_id_l'));b=str(getattr(row,'unique_id_r'))
        found.add(tuple(sorted((a,b))))
    real={}
    for rid,eid in truth.items(): real.setdefault(eid,[]).append(rid)
    gt=set()
    from itertools import combinations
    for rids in real.values():
        gt|={tuple(sorted((a,b))) for a,b in combinations(rids,2)}
    tp=len(found & gt)
    report={'library':'Splink_5','records':len(data),'predicted_pairs':len(found),'true_pairs':len(gt),
            'tp':tp,'fp':len(found-gt),'fn':len(gt-found),'precision':round(tp/len(found),4) if found else 0,
            'recall_all_pairs':round(tp/len(gt),4) if gt else 0,'seconds':round(time.perf_counter()-start,3),
            'warning':'Evaluate blocking recall separately; training parameters not tuned, only a smoke test.'}
    (RESULTS/'splink5_library_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
