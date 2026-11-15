"""Independent Fraction-only face enumeration for three tiny quadratic controls.

No scientific module, CVXPY solver or reduction helper is imported.
The algorithm enumerates all box faces and sign pairs and solves each affine
least-squares projection by rational row reduction.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import json


def reduced_rows(matrix, rhs, columns):
    rows = [[F(x) for x in row]+[F(b)] for row,b in zip(matrix,rhs)]
    pivot = 0
    for column in range(columns):
        found = next((i for i in range(pivot,len(rows)) if rows[i][column]),None)
        if found is None:
            continue
        rows[pivot],rows[found] = rows[found],rows[pivot]
        scale = rows[pivot][column]
        rows[pivot] = [x/scale for x in rows[pivot]]
        for i in range(len(rows)):
            if i != pivot:
                scale = rows[i][column]
                rows[i] = [x-scale*y for x,y in zip(rows[i],rows[pivot])]
        pivot += 1
    if any(all(x==0 for x in row[:columns]) and row[-1] for row in rows):
        return None
    return [row for row in rows if any(row[:columns])]


def project_on_face(matrix, target, face):
    k = len(target)
    fixed = [i for i,s in enumerate(face) if s]
    free = [i for i,s in enumerate(face) if not s]
    value = [F(s,4) for s in face]
    rhs = [-sum(F(row[i])*value[i] for i in fixed) for row in matrix]
    rows = reduced_rows([[row[i] for i in free] for row in matrix],rhs,len(free))
    if rows is None:
        return None
    if rows:
        gram = [[sum(a*b for a,b in zip(left[:-1],right[:-1])) for right in rows] for left in rows]
        residual = [row[-1]-sum(a*target[i] for a,i in zip(row[:-1],free)) for row in rows]
        solution = reduced_rows(gram,residual,len(rows))
        multiplier = [row[-1] for row in solution]
    else:
        multiplier=[]
    for j,i in enumerate(free):
        value[i] = target[i]+sum(rows[l][j]*multiplier[l] for l in range(len(rows)))
    if any(abs(x)>F(1,4) for x in value):
        return None
    assert all(sum(F(a)*b for a,b in zip(row,value))==0 for row in matrix)
    return value


def exact_value(matrix):
    k=len(matrix[0]); best=None; best_u=None
    for tail in product((-1,1),repeat=k-1):
        target=[F(s,2) for s in (1,)+tail]
        for face in product((-1,0,1),repeat=k):
            value=project_on_face(matrix,target,face)
            if value is None:
                continue
            objective=F(3*k,2)+2*sum((x-y)**2 for x,y in zip(value,target))
            if best is None or objective<best:
                best,best_u=objective,value
    return best,best_u


def run(destination):
    controls=[('single_clause',[[1,1,1,1]],F(13,2)),
              ('repeated_unsatisfiable',[[1,3]],F(125,36)),
              ('distinct_unsatisfiable',[[1,1,1,1,0],[1,1,1,0,1],
                                         [1,1,0,1,1],[1,0,1,1,1]],F(649,72))]
    report=[]
    for name,matrix,expected in controls:
        value,u=exact_value(matrix)
        assert value==expected,(name,value,expected)
        report.append(dict(name=name,matrix=matrix,proper_risk=str(value),
                           parameter=[str(x) for x in u],passed=True))
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(Path(__file__).resolve().parents[1]/'results/consistency_separation/exact_controls.json'))
    run(parser.parse_args().output)
