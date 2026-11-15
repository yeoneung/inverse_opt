"""Small-instance tests of the mixed/proper computational separation."""
import argparse
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.consistency_complexity import (boolean_solutions, clause_matrix,
    solve_consistent_center)


def instances():
    yield 'single_clause', 3, [(0,1,2)]
    yield 'repeated_unsatisfiable', 1, [(0,0,0)]
    yield 'distinct_unsatisfiable', 4, [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    for count in [3,5,7]:
        for seed in range(4):
            rng = np.random.default_rng(20261115+100*count+seed)
            clause_count = (2*count+2)//3
            planted = np.zeros(count,dtype=int)
            planted[rng.choice(count,size=max(1,count//3),replace=False)] = 1
            pos, neg = np.flatnonzero(planted), np.flatnonzero(1-planted)
            for _ in range(10000):
                if seed % 2 == 0:
                    clauses = [tuple(map(int,[rng.choice(pos),rng.choice(neg),rng.choice(neg)]))
                               for _ in range(clause_count)]
                else:
                    clauses = [tuple(map(int,rng.integers(0,count,3))) for _ in range(clause_count)]
                if {i for clause in clauses for i in clause} == set(range(count)):
                    break
            else:
                raise RuntimeError('Could not generate a formula covering every variable.')
            yield f'random_{count}_{seed}', count, clauses


def run(output):
    destination = ROOT/'results'/output
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, count, clauses in instances():
        matrix = clause_matrix(count, clauses)
        # The Boolean labels are computed independently of all convex solves.
        assignments = boolean_solutions(count, clauses)
        result = solve_consistent_center(matrix)
        baseline = result['baseline']
        if assignments:
            assert abs(result['proper_upper']-baseline) < 2e-7
        else:
            assert result['proper_lower']-baseline >= 1/6-2e-7
        assert result['proper_upper']-result['proper_lower'] < 2e-7
        assert result['max_equality_residual'] < 1e-8
        assert result['max_box_violation'] < 1e-8
        rows.append(dict(name=name, variables=count, clauses=clauses,
                         source_nullity=int(matrix.shape[1]-np.linalg.matrix_rank(matrix)),
                         satisfying_assignments=len(assignments), **result))
    summary = dict(instances=len(rows), satisfiable=sum(r['satisfying_assignments']>0 for r in rows),
                   unsatisfiable=sum(r['satisfying_assignments']==0 for r in rows),
                   all_labels_agree=True,
                   max_risk_primal_dual_gap=max(r['proper_upper']-r['proper_lower'] for r in rows),
                   max_equality_residual=max(r['max_equality_residual'] for r in rows),
                   max_box_violation=max(r['max_box_violation'] for r in rows),
                   min_unsatisfiable_excess=min(r['proper_lower']-r['baseline'] for r in rows if not r['satisfying_assignments']),
                   max_satisfiable_excess=max(abs(r['proper_upper']-r['baseline']) for r in rows if r['satisfying_assignments']),
                   random_source_nullities=sorted({r['source_nullity'] for r in rows if r['name'].startswith('random_')}),
                   scope='Exhaustive small-instance verification, not empirical worst-case complexity estimation.')
    (destination/'runs.json').write_text(json.dumps(rows,indent=2), encoding='utf-8')
    (destination/'summary.json').write_text(json.dumps(summary,indent=2), encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='consistency_separation')
    run(parser.parse_args().output)
