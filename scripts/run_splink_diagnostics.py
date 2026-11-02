"""Runs original Splink 5.0 API. Separates blocking coverage from model decisions.
Retains the already-successful V5 model definition and training calls.
"""
import json,time
from pathlib import Path
import pandas as pd
from audit_fixture import truth_and_candidates,pair_key

ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/'fixtures';OUT=ROOT/'results';OUT.mkdir(exist_ok=True)

def summarize(pairs,gt):
    tp=len(pairs&gt);fp=len(pairs-gt);fn=len(gt-pairs)
    p=tp/(tp+fp) if tp+fp else 0
    r=tp/(tp+fn) if tp+fn else 0
    return {'predicted':len(pairs),'tp':tp,'fp':fp,'fn':fn,'precision':round(p,5),
            'recall_all_pairs':round(r,5),'f1':round(2*p*r/(p+r),5) if p+r else 0}

def to_pairs(df, minimum_probability):
    assert 'match_probability' in df.columns, f'missing match_probability; found {list(df.columns)}'
    m=df[df.match_probability >= minimum_probability]
    return {pair_key(a,b) for a,b in zip(m.unique_id_l,m.unique_id_r)}

def main():
    from splink import DuckDBAPI,Linker,SettingsCreator,block_on
    import splink.comparison_library as cl

    source=pd.read_csv(FIX/'synthetic_entities_harder.csv',dtype=str,keep_default_na=False)
    rows,gt,candidates=truth_and_candidates()
    data=source.rename(columns={'record_id':'unique_id'})
    print('START', json.dumps({'records':len(data),'truth':len(gt),'blocking_candidates':len(candidates),'blocking_hits':len(gt&candidates)},ensure_ascii=False),flush=True)
    db=DuckDBAPI()
    table=db.register(data,dataset_display_name='synthetic_companies')
    settings=SettingsCreator(link_type='dedupe_only',comparisons=[
        cl.JaroWinklerAtThresholds('name',[0.93,0.80]),
        cl.ExactMatch('region'),cl.ExactMatch('sector'),cl.ExactMatch('bank_suffix')],
        blocking_rules_to_generate_predictions=[block_on('region','sector'),block_on('bank_suffix')])
    linker=Linker(table,settings)
    start=time.perf_counter()
    linker.training.estimate_probability_two_random_records_match([block_on('bank_suffix','sector')],recall=0.6)
    linker.training.estimate_u_using_random_sampling(max_pairs=100_000)
    linker.training.estimate_parameters_using_expectation_maximisation(block_on('bank_suffix'))
    linker.training.estimate_parameters_using_expectation_maximisation(block_on('region','sector'))

    # Fixed threshold from the verified previous CI run.
    original_df=linker.inference.predict(threshold_match_probability=0.5).as_pandas_dataframe()
    half_pairs={pair_key(a,b) for a,b in zip(original_df.unique_id_l, original_df.unique_id_r)}
    report={'library':'Splink_5','records':len(data),'true_pairs':len(gt),
            'blocking':{'candidate_pairs':len(candidates),'possible_pairs':len(data)*(len(data)-1)//2,
                        'true_pairs_reached':len(candidates&gt),'true_pairs_lost':len(gt-candidates),
                        'candidate_recall':round(len(candidates&gt)/len(gt),5)},
            'threshold_0_5':summarize(half_pairs,gt),
            'threshold_0_5_misses':{'blocked_out':len(gt-candidates),'model_or_threshold':len((gt&candidates)-half_pairs)},
            'elapsed_seconds':round(time.perf_counter()-start,3),
            'note':'V5 model warnings must be checked in splink_original_diagnostics.log; not all m probabilities may be trained.'}
    # Run an optional lower-threshold sweep using the same estimated model.
    # This does not fix training defects; it merely reveals threshold tradeoffs.
    try:
        scored=linker.inference.predict(threshold_match_probability=0.01).as_pandas_dataframe()
        report['threshold_sweep']={str(t):summarize(to_pairs(scored,t),gt) for t in [0.01,0.1,0.3,0.5,0.7,0.9]}
        report['threshold_sweep']['0.5']['consistency_with_first_run']=(report['threshold_sweep']['0.5']==report['threshold_0_5'])
    except Exception as exc:
        report['threshold_sweep_error']={'exception_type':type(exc).__name__,'message':str(exc)}
    (OUT/'splink_diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
