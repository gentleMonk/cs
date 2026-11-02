"""V3: reproducible, dependency-light STRESS TEST of the V2 results.
Does NOT run Raha or Splink: neither library can be installed in this environment.
Uses only numpy, pandas, scikit-learn, rapidfuzz. Synthetic fixtures only.
"""
from __future__ import annotations
import json, time, collections, statistics
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from rapidfuzz.fuzz import ratio

HERE=Path(__file__).resolve().parent
FIX=HERE/'fixtures';OUT=HERE/'results';OUT.mkdir(exist_ok=True)

def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=int)
    tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.}

def entity_block_tests():
    df=pd.read_csv(FIX/'synthetic_entities_harder.csv',dtype=str,keep_default_na=False)
    truth=json.loads((FIX/'synthetic_entities_harder_truth.json').read_text())
    df['truth_entity']=df['record_id'].map(truth)
    rows=df.set_index('record_id').to_dict('index')
    all_entities=sorted(df.truth_entity.unique());out=[]
    def keys(row,mode):
        keys0=[]
        if mode in ('region_sector','base','base_plus_prefix'):
            keys0+=['region_sector:'+row['region']+'|'+row['sector']]
        if mode in ('bank','base','base_plus_prefix') and row['bank_suffix']:
            keys0+=['bank:'+row['bank_suffix']]
        if mode in ('base','base_plus_prefix'):
            # Exactly reproduce V2's base candidate mechanism
            keys0+=['sectorname:'+row['sector']+'|'+(row['name'].replace(row['region'],'')[:1] if row['name'] else '')]
        if mode=='base_plus_prefix':
            keys0+=['prefix:'+row['name'][:4]]
        return keys0
    def candidate_pairs(ids,mode):
        groups=collections.defaultdict(list)
        for id in ids:
            for key in keys(rows[id],mode):
                if key: groups[key].append(id)
        matches=set()
        for members in groups.values():
            for a,b in combinations(members,2):matches.add(tuple(sorted((a,b))))
        return matches
    for seed in range(30):
        rng=np.random.default_rng(2026+seed)
        ents=set(rng.choice(all_entities,size=95,replace=False))
        ids=[id for id,r in rows.items() if r['truth_entity'] in ents]
        real={tuple(sorted((a,b))) for a,b in combinations(ids,2) if rows[a]['truth_entity']==rows[b]['truth_entity']}
        all_count=len(ids)*(len(ids)-1)//2
        for mode in ('region_sector','bank','base','base_plus_prefix'):
            candidate=candidate_pairs(ids,mode)
            good=len(real & candidate)
            out.append({'seed':seed,'method':mode,'records':len(ids),'true_pairs':len(real),
                 'candidate_pairs':len(candidate),'candidate_fraction':len(candidate)/all_count,
                 'found_true_pairs':good,'blocking_recall':good/len(real) if real else None,
                 'missed_true_pairs':len(real)-good,
                 'candidate_precision':good/len(candidate) if candidate else 0})
    dd=pd.DataFrame(out);dd.to_csv(OUT/'blocking_30splits.csv',index=False)
    summary=[]
    for name,g in dd.groupby('method'):
        summary.append({'method':name,'splits':len(g), 'mean_blocking_recall':round(g.blocking_recall.mean(),4),
           'min_recall':round(g.blocking_recall.min(),4),'max_recall':round(g.blocking_recall.max(),4),
           'mean_candidates':round(g.candidate_pairs.mean(),1),'mean_candidate_fraction':round(g.candidate_fraction.mean(),5),
           'sum_true_pairs':int(g.true_pairs.sum()),'sum_missed_true_pairs':int(g.missed_true_pairs.sum())})
    pd.DataFrame(summary).to_csv(OUT/'blocking_summary.csv',index=False)
    return {'dataset_records':len(df),'real_entities':len(all_entities),'full_pairs':len(df)*(len(df)-1)//2,
            'grouped_holdout_splits':30,'summary':summary}

def active_cold_start():
    # Important: no human truth labels are picked on purpose; initial labels are a BLIND random sample.
    df=pd.read_csv(FIX/'synthetic_orders.csv',dtype=str,keep_default_na=False)
    gt=json.loads((FIX/'ground_truth.json').read_text())
    classes=['missing','date_format','amount_mismatch','category_inconsistent','numeric_outlier']
    y=np.zeros(len(df),dtype=int)
    for kind in classes:y[np.array(gt[kind],dtype=int)]=1
    # Explicit rule flags are allowed as unsupervised detector features but make in-domain supervised results optimistic.
    sku=df.sku.fillna('')
    date=df.order_date.fillna('')
    quantity=pd.to_numeric(df.quantity,errors='coerce')
    price=pd.to_numeric(df.unit_price,errors='coerce')
    amount=pd.to_numeric(df.amount,errors='coerce')
    mismatch=((quantity*price-amount).abs()>0.02).fillna(False)
    catmode=df.groupby('sku')['category'].agg(lambda a:a.mode().iloc[0] if len(a.mode()) else '')
    mismatch_cat=(df.category!=df.sku.map(catmode)).fillna(False)
    feature_names=['sku_missing','date_invalid','amount_ineq','price_gt500','category_disagree','price_log','sku_len','name_len']
    X=np.array([
        sku.str.strip().eq('').astype(float).values,
        (~date.str.match(r'^20\d\d-\d\d-\d\d$')).astype(float).values,
        mismatch.astype(float).values,
        (price>500).fillna(False).astype(float).values,
        mismatch_cat.astype(float).values,
        np.log1p(price.fillna(0).clip(lower=0).values),
        sku.str.len().values.astype(float),
        df.product_name.fillna('').str.len().values.astype(float)
    ]).T
    # Frozen, shared holdout for all trials. Evaluation truth cannot drive sample selection.
    train_idx,test_idx=train_test_split(np.arange(len(df)),test_size=.30,stratify=y,random_state=2027)
    pool=train_idx
    trials=[]
    first8_positive_count=[]
    for seed in range(20):
        rng=np.random.default_rng(9000+seed)
        labeled=list(map(int,rng.choice(pool,size=8,replace=False)))
        npos=int(y[labeled].sum());first8_positive_count.append(npos)
        for strategy in ('random','uncertainty'):
            select_rng=np.random.default_rng(9000+seed+123)
            picked=labeled.copy()
            while len(picked)<32:
                remaining=np.array(sorted(set(pool)-set(picked)),dtype=int)
                # A model needing both classes cannot be trained when all labels are negative.
                if strategy=='random' or len(set(y[picked]))<2:
                    chosen=int(select_rng.choice(remaining))
                else:
                    m=LogisticRegression(class_weight='balanced',max_iter=150,random_state=11)
                    m.fit(X[picked],y[picked])
                    probs=m.predict_proba(X[remaining])[:,1]
                    chosen=int(remaining[np.argmin(abs(probs-0.5))])
                picked.append(chosen)
            if len(set(y[picked]))<2:
                # All-negative blind labels: no valid binary supervised classifier; conservative predict none.
                pred=np.zeros(len(test_idx),dtype=int);model_available=False
            else:
                m=LogisticRegression(class_weight='balanced',max_iter=150,random_state=11)
                m.fit(X[picked],y[picked]);pred=(m.predict_proba(X[test_idx])[:,1]>=.5).astype(int)
                model_available=True
            trials.append({'seed':seed,'strategy':strategy,'initial_positive_labels':npos,'total_positive_labels':int(y[picked].sum()),
                           'can_train_model':model_available,**metrics(y[test_idx],pred)})
    trials_df=pd.DataFrame(trials);trials_df.to_csv(OUT/'cold_start_trials.csv',index=False)
    s=[]
    for strat,g in trials_df.groupby('strategy'):
        s.append({'strategy':strat,'trials':len(g),'mean_precision':round(g.precision.mean(),4),
                  'mean_recall':round(g.recall.mean(),4),'min_recall':round(g.recall.min(),4),
                  'max_recall':round(g.recall.max(),4),'mean_positive_labels':round(g.total_positive_labels.mean(),2),
                  'unable_to_train_trials':int((~g.can_train_model).sum())})
    pd.DataFrame(s).to_csv(OUT/'cold_start_summary.csv',index=False)
    # Simulate a genuinely novel issue: train without one type, evaluate only those unseen errors.
    unseen=np.array(gt['category_inconsistent'],dtype=int)
    unseen_test=np.intersect1d(unseen,test_idx)
    # Exclude rows with *any* category problem from training, even if they have other errors.
    known_train=np.setdiff1d(train_idx,unseen)
    m=LogisticRegression(class_weight='balanced',max_iter=150,random_state=4)
    m.fit(X[known_train],y[known_train])
    unseen_prediction=(m.predict_proba(X[unseen_test])[:,1]>=.5).astype(int)
    return {'rows':len(df),'train_rows':len(train_idx),'test_rows':len(test_idx),
            'observable_dirty_rows':int(y.sum()),'known_business_fact_errors_not_detectable_without_reference':len(gt['unverifiable_fact']),
            'trials':20,'initial_blind_8_no_positive_trials':sum(n==0 for n in first8_positive_count),
            'initial_blind_8_mean_positive':round(np.mean(first8_positive_count),2),'results':s,
            'unseen_category_test_cases':len(unseen_test),'unseen_category_predicted_positive':int(unseen_prediction.sum()),
            'unseen_category_recall':round(float(unseen_prediction.mean()),4) if len(unseen_test) else None,
            'warning':'Cross-row statistics computed on one original dataset. Unseen-type experiment can include incidental overlapping anomalies; not a proof of semantic generalization.'}

def main():
    t=time.perf_counter(); a=entity_block_tests(); print('blocking done',round(time.perf_counter()-t,2),flush=True)
    b=active_cold_start(); print('active done',round(time.perf_counter()-t,2),flush=True)
    report={'toolchain':{'raha':'NOT INSTALLED','splink':'NOT INSTALLED','simulation':'rapidfuzz scikit-learn pandas numpy'},
       'entity_blocking':a,'active_learning_cold_start':b,
       'run_seconds':round(time.perf_counter()-t,3),
       'limitations':['Synthetic generated data only; labels generated by synthetic fixtures',
        'Splink and Raha originals were not run; package installation DNS fails',
        'Same underlying dataset used for multiple splits; 30 splits are not independent datasets',
        'No automatic repair should occur based on these candidate detectors']}
    (OUT/'v3_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
