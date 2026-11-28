"""Validate global exchange on all twelve retained four-parameter capacities."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from exchange_pilot import run_exchange
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reruns'/'capacity_p4')
    args=parser.parse_args()
    directory=args.output.resolve()
    directory.mkdir(parents=True,exist_ok=False)
    output=directory/'results.json'
    data=json.loads((ROOT.parent/'results/balanced_reproduction/runs.json').read_text())
    arrays=np.load(ROOT.parent/'results/balanced_reproduction/instances.npz')
    rows=[r for r in data['rows'] if r['kind']=='capacity' and r['p']==4 and r['budget']==513 and r['repetition']==0]
    results=[]
    for row in rows:
        name=row['instance']
        A,b,c,G=[arrays[name+'_'+k] for k in ['A','b','c','G']]
        run=run_exchange(A,b,c,G,label='p4validation_'+name,time_limit=20,iterations=10,
                         capacity_bounds=True,log_directory=directory)
        run['baseline']=dict(queries=row['queries'],upper=row['upper'],variance=row['variance'],
            original_gap=row['original_gap'],adaptive_gap=row['adaptive_gap'],
            adaptive_total_seconds=row['adaptive_total_seconds'])
        results.append(run)
        output.write_text(json.dumps({'complete':len(results)==len(rows),'cases':results},indent=2),encoding='utf-8')
        print('COMPLETE',name,'detected',run['detected'],'bound',run['certificate'],'time',run['total_seconds'],flush=True)


if __name__=='__main__':
    main()
