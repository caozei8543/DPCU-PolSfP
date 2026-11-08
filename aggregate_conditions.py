#!/usr/bin/env python3
"""Summarize FIVE acquisition sequences PER condition, then macro average by C1-C9.

CSV with columns: condition,sequence,baseline_mae,ours_mae,oracle_mae,
selection_error,invalid_pct. Each sequence is one repeated acquisition of the
SAME specimen; mean±SD across sequences is NOT a confidence interval across
independent specimens.
"""
from collections import defaultdict
from pathlib import Path
import argparse,csv,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',required=True)
p.add_argument('--out',default='results/condition_aggregate.json');a=p.parse_args()
with open(a.input,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
cols=('baseline_mae','ours_mae','oracle_mae','selection_error','invalid_pct')
if not rows or not {'condition','sequence',*cols}.issubset(rows[0]):
    raise SystemExit('Input CSV needs condition,sequence and '+','.join(cols))
groups=defaultdict(list)
for r in rows:groups[r['condition']].append(r)
out=[]
for cond,vals in sorted(groups.items()):
    if len({v['sequence'] for v in vals})!=len(vals):raise ValueError('Duplicated sequence ID in '+cond)
    row={'condition':cond,'n_sequences':len(vals)}
    for c in cols:
        x=np.array([float(v[c]) for v in vals],dtype=float)
        row[c+'_mean']=float(np.mean(x));row[c+'_sd']=float(np.std(x,ddof=1)) if len(x)>1 else None
    out.append(row)
macro={c:float(np.mean([v[c+'_mean'] for v in out])) for c in cols}
result={'conditions':out,'equally_weighted_macro_mean':macro,
        'warning':'Repeated sequences on the same physical configuration are not independent target specimens.'}
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
print('Saved:',a.out)
