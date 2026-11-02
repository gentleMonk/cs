"""Independent fixture audit. No Raha or Splink dependency required."""
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'fixtures'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)


def read_rows(name):
    with (FIX / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def pair_key(a, b):
    return tuple(sorted((str(a), str(b))))


def truth_and_candidates():
    rows = read_rows('synthetic_entities_harder.csv')
    truth_map = json.loads((FIX / 'synthetic_entities_harder_truth.json').read_text(encoding='utf8'))
    groups = defaultdict(list)
    for rid, eid in truth_map.items():
        groups[eid].append(rid)
    gt = {pair_key(a,b) for rids in groups.values() for a,b in combinations(rids,2)}
    candidates = set()
    for cols in [('region', 'sector'), ('bank_suffix',)]:
        blocks = defaultdict(list)
        for row in rows:
            blocks[tuple(row[c] for c in cols)].append(row['record_id'])
        for ids in blocks.values():
            candidates.update(pair_key(a,b) for a,b in combinations(ids,2))
    return rows, gt, candidates


def audit():
    dirty, clean = read_rows('raha_dirty.csv'), read_rows('raha_clean.csv')
    assert len(dirty) == len(clean)
    assert [r.keys() for r in dirty] == [r.keys() for r in clean]
    columns = list(dirty[0]); errors = {k: sum(d[k] != c[k] for d,c in zip(dirty,clean)) for k in columns}
    rows, gt, candidates = truth_and_candidates()
    result = {
        'raha': {'records':len(dirty), 'columns':len(columns), 'true_error_cells':sum(errors.values()),'error_cells_by_column':errors},
        'splink': {'records':len(rows), 'possible_pairs': len(rows)*(len(rows)-1)//2,
                   'true_pairs':len(gt),'blocking_candidates':len(candidates),
                   'true_pairs_in_candidates':len(gt & candidates),
                   'blocking_false_negative_pairs':len(gt-candidates),
                   'blocking_recall':len(gt & candidates)/len(gt) if gt else None}
    }
    (OUT/'fixture_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__ == '__main__':
    audit()
