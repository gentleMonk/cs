"""Runs the ORIGINAL Raha Detection.run API for multiple artificial label budgets.
The clean CSV provides an oracle for Raha's simulated human-labeling step.
Do not interpret as blind anomaly detection or label-free model performance.
"""
import json, time, traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT/'fixtures'; OUT = ROOT/'results'; OUT.mkdir(exist_ok=True)

def main():
    from raha.dataset import Dataset
    from raha.detection import Detection
    from audit_fixture import read_rows

    original = read_rows('raha_dirty.csv')
    clean = read_rows('raha_clean.csv')
    col_names = list(original[0]); true_errors = {(i,j) for i,(a,b) in enumerate(zip(original,clean))
                                                 for j,k in enumerate(col_names) if a[k] != b[k]}
    budgets = (10,20,40)
    results = []
    for budget in budgets:
        job_name = f'raha_budget_{budget}'
        dataset = {'name':job_name,'path':str((FIX/'raha_dirty.csv').resolve()),
                   'clean_path':str((FIX/'raha_clean.csv').resolve())}
        algo = Detection()
        algo.LABELING_BUDGET=budget
        algo.SAVE_RESULTS=False
        # Only PVD/RVD: exact families used by the verified previous CI run.
        algo.ERROR_DETECTION_ALGORITHMS=['PVD','RVD']
        start=time.perf_counter()
        predicted=algo.run(dataset)
        scores=Dataset(dataset).get_data_cleaning_evaluation(predicted)
        pred_coords=set()
        for k in predicted.keys():
            if isinstance(k,tuple) and len(k)==2:
                try: pred_coords.add((int(k[0]),int(k[1])))
                except (TypeError,ValueError): pass
        recognized_keys = len(pred_coords)==len(predicted)
        if recognized_keys:
            tp=len(pred_coords & true_errors);fp=len(pred_coords-true_errors);fn=len(true_errors-pred_coords)
            per_field={col:{'true':sum((i,j) in true_errors for i in range(len(original))),
                            'predicted':sum((i,j) in pred_coords for i in range(len(original))),
                            'tp':sum((i,j) in pred_coords and (i,j) in true_errors for i in range(len(original)))}
                       for j,col in enumerate(col_names)}
        else:
            tp=fp=fn=None
            per_field={}
        item={'label_budget_rows':budget,'algorithm_families':['PVD','RVD'],
              'ground_truth_cells':len(true_errors),'predicted_cells':len(predicted),
              'precision_library':float(scores[0]),'recall_library':float(scores[1]),'f1_library':float(scores[2]),
              'tp_independently_computed':tp,'fp_independently_computed':fp,'fn_independently_computed':fn,
              'coordinates_recognized':recognized_keys,'per_column':per_field,'elapsed_seconds':round(time.perf_counter()-start,4),
              'oracle_labeling':True}
        results.append(item)
        print(json.dumps(item,ensure_ascii=False,indent=2),flush=True)
    (OUT/'raha_diagnostics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':
    main()
