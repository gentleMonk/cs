"""OPTIONAL: real Raha original library test. Requires separate 'pip install raha'.
Not executed in the hosted environment. Runs the real Raha Detection API with a truth-oracle
for simulated human labels ONLY in this test. The live platform must use human review.
"""
import json, time
from pathlib import Path

BASE=Path(__file__).parent.parent; FIX=BASE/'fixtures'; RESULTS=BASE/'results'

def main():
    from raha.detection import Detection
    from raha.dataset import Dataset
    dd={'name':'synthetic_raha_v2', 'path':str((FIX/'raha_dirty.csv').resolve()), 'clean_path':str((FIX/'raha_clean.csv').resolve())}
    detector=Detection()
    detector.LABELING_BUDGET=20
    detector.SAVE_RESULTS=False
    # Limit heavy strategy families for local integration test.
    # Original Raha default uses OD, PVD, RVD and KBVD.
    detector.ERROR_DETECTION_ALGORITHMS=['PVD','RVD']
    start=time.perf_counter()
    predicted=detector.run(dd)
    clean=Dataset(dd)
    prec,recall,f1=clean.get_data_cleaning_evaluation(predicted)[:3]
    report={'library':'original_Raha','algorithm_families':['PVD','RVD'],'labeled_rows':20,'predicted_error_cells':len(predicted),'precision':prec,'recall':recall,'f1':f1,'seconds':time.perf_counter()-start,'note':'Ground-truth oracle supplies synthetic human labels; not a blind production inference.'}
    (RESULTS/'raha_library_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
