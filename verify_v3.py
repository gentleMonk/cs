import json
from pathlib import Path
import pandas as pd
R=Path(__file__).parent/'results'
b=json.loads((R/'v3_report.json').read_text())
s=json.loads((R/'stress_blocking.json').read_text())
assert b['entity_blocking']['dataset_records']==400
assert b['entity_blocking']['grouped_holdout_splits']==30
assert len(pd.read_csv(R/'blocking_30splits.csv'))==120
assert len(pd.read_csv(R/'cold_start_trials.csv'))==40
assert s['baseline']['true_pairs']==80
assert s['identity_signal_erasure_12_true_pairs']['found_true_pairs']==68
assert s['shared_name_prefix_100_records']['candidate_pairs']>s['baseline']['candidate_pairs']
assert b['active_learning_cold_start']['initial_blind_8_no_positive_trials']>=0
print('Verification checks passed. Raha and Splink library execution NOT asserted.')
